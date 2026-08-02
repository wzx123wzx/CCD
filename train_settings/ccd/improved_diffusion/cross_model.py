import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
import config.settings
import torchvision.models as models
from utils_transform.tps_utils import upsample_tps, tps2flow
import torch.nn.functional as F




def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio,
            separate_cross_attn,
            proj=False,
            **block_kwargs,
    ):
        super().__init__()
        # hidden_size = hidden_size*3
        self.proj = proj
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.cross_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, x, t, update_cond=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=1)
        )

        conditioned, _ = self.cross_attn(
            query=self.cross_norm(x),  # (N, T, D)
            key=update_cond,  # (N, steps, D)
            value=update_cond,  # (N, steps, D)
            need_weights=False)
        x = x + conditioned  # (N, T, D)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa))

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x




class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_res18_FeatureMap(resnet18_model):
    # suppose input resolution to be    512*384
    # the output resolution should be   32*24

    layers_list = []

    # layers_list.append(resnet18_model.conv1)    #stride 2*2     H/2
    layers_list.append(nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  # stride 2       H/4

    layers_list.append(resnet18_model.layer1)  # H/4
    layers_list.append(resnet18_model.layer2)  # H/8
    # layers_list.append(resnet18_model.layer3)  # H/16

    feature_extractor = nn.Sequential(*layers_list)
    # exit(0)

    return feature_extractor



class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            settings,
            input_size,
            embedding_size,
            patch_size,
            input_patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio=4.0,
            time_frequency_embedding_size=256,
            learn_sigma=False,
            use_pretrain_VGG=False,
            resume=False,
    ):
        # input_size -> tps resolution
        # embedding_size -> embedding condition (VGG feature, line mask, seg mask) resolution
        # patch_size -> patch embedding size
        # in_channels -> 2 (for flow and tps motion)
        # hidden_size -> embedding channel number
        # depth -> DiT depth (number of transformer blocks)
        # num_heads -> for transformer
        super().__init__()
        self.settings = settings
        self.input_size = input_size
        self.cascade_num = 3
        self.in_channels = in_channels  # 2
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_patch_size = input_patch_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_pretrain_VGG = use_pretrain_VGG
        self.separate_cross_attn = self.settings.model.separate_cross_attn
        self.num_timesteps = self.settings.model.diffusion_steps
        self.rescale_timesteps = self.settings.model.rescale_timesteps
        self.scale = 1000 / self.num_timesteps

        # define the res18 backbone
        resnet18_model = models.resnet.resnet18(pretrained=True)
        # resnet34_model = models.resnet.resnet34(pretrained=True)

        # print(resnet18_model)
        if torch.cuda.is_available():
            resnet18_model = resnet18_model.cuda()
        # resnet34_model = resnet34_model.cuda()

        self.feature_extractor = get_res18_FeatureMap(resnet18_model)

        self.input_embedder_tps0 = PatchEmbed(self.input_size[0], input_patch_size, in_channels, hidden_size, bias=True)
        self.input_embedder_tps1 = PatchEmbed(self.input_size[1], input_patch_size, in_channels, hidden_size, bias=True)
        self.input_embedder_grid = PatchEmbed(
            [self.input_size[2], self.input_size[3]], input_patch_size, in_channels, hidden_size, bias=True
        )

        num_input_patches_tps0 = self.input_embedder_tps0.num_patches
        self.noised_input_pos_embed_tps0 = nn.Parameter(
            torch.zeros(1, num_input_patches_tps0, hidden_size), requires_grad=False
        )
        num_input_patches_tps1 = self.input_embedder_tps1.num_patches
        self.noised_input_pos_embed_tps1 = nn.Parameter(
            torch.zeros(1, num_input_patches_tps1, hidden_size), requires_grad=False
        )
        num_input_patches_grid = self.input_embedder_grid.num_patches
        self.noised_input_pos_embed_grid = nn.Parameter(
            torch.zeros(1, num_input_patches_grid, hidden_size), requires_grad=False
        )
        self.num_input_patches = [num_input_patches_tps0, num_input_patches_tps1, num_input_patches_grid]

        self.update_embedder_tps0 = PatchEmbed(
            embedding_size, patch_size, in_chans=128, embed_dim=hidden_size, bias=True  # 256
        )
        self.update_embedder_tps1 = PatchEmbed(
            embedding_size, patch_size, in_chans=128, embed_dim=hidden_size, bias=True  # 256
        )
        self.update_embedder_grid = PatchEmbed(
            embedding_size, patch_size, in_chans=128, embed_dim=hidden_size, bias=True  # 256
        )

        self.t_embedder = TimestepEmbedder(hidden_size, time_frequency_embedding_size)

        num_cond_patches = self.update_embedder_tps0.num_patches
        self.num_cond_patches = num_cond_patches
        self.noised_cond_pos_embed = nn.Parameter(
            torch.zeros(1, num_cond_patches, hidden_size), requires_grad=False
        )

        self.blocks_stage1 = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    separate_cross_attn=self.separate_cross_attn,
                    proj=False,
                )
                for _ in range(depth)
            ]
        )
        self.blocks_stage2 = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    separate_cross_attn=self.separate_cross_attn,
                    proj=False,
                )
                for _ in range(depth)
            ]
        )
        self.blocks_stage3 = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    separate_cross_attn=self.separate_cross_attn,
                    proj=False,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer_stage1 = FinalLayer(hidden_size, input_patch_size, self.out_channels)
        self.final_layer_stage2 = FinalLayer(hidden_size, input_patch_size, self.out_channels)
        self.final_layer_stage3 = FinalLayer(hidden_size, input_patch_size, self.out_channels)

        if resume == False:
            self.initialize_weights()
        else:
            pass


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        noised_input_pos_embed_tps0 = get_2d_sincos_pos_embed(
            self.noised_input_pos_embed_tps0.shape[-1],
            self.input_size[0])
        self.noised_input_pos_embed_tps0.data.copy_(
            torch.from_numpy(noised_input_pos_embed_tps0).float().unsqueeze(0))

        noised_input_pos_embed_tps1 = get_2d_sincos_pos_embed(
            self.noised_input_pos_embed_tps1.shape[-1],
            self.input_size[1])
        self.noised_input_pos_embed_tps1.data.copy_(
            torch.from_numpy(noised_input_pos_embed_tps1).float().unsqueeze(0))

        noised_input_pos_embed_grid = get_2d_sincos_pos_embed(
            self.noised_input_pos_embed_grid.shape[-1],
            [self.input_size[2], self.input_size[3]])
        self.noised_input_pos_embed_grid.data.copy_(
            torch.from_numpy(noised_input_pos_embed_grid).float().unsqueeze(0))

        noised_cond_pos_embed = get_2d_sincos_pos_embed(
            self.noised_cond_pos_embed.shape[-1],
            int(self.num_cond_patches ** 0.5),
        )
        self.noised_cond_pos_embed.data.copy_(
            torch.from_numpy(noised_cond_pos_embed).float().unsqueeze(0)
        )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.input_embedder_tps0.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.input_embedder_tps0.proj.bias, 0)

        w = self.input_embedder_tps1.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.input_embedder_tps1.proj.bias, 0)

        w = self.input_embedder_grid.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.input_embedder_grid.proj.bias, 0)

        w = self.update_embedder_tps0.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.update_embedder_tps0.proj.bias, 0)

        w = self.update_embedder_tps1.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.update_embedder_tps1.proj.bias, 0)

        w = self.update_embedder_grid.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.update_embedder_grid.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks_stage1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.blocks_stage2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.blocks_stage3:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer_stage1.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_stage1.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_stage1.linear.weight, 0)
        nn.init.constant_(self.final_layer_stage1.linear.bias, 0)

        nn.init.constant_(self.final_layer_stage2.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_stage2.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_stage2.linear.weight, 0)
        nn.init.constant_(self.final_layer_stage2.linear.bias, 0)

        nn.init.constant_(self.final_layer_stage3.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_stage3.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_stage3.linear.weight, 0)
        nn.init.constant_(self.final_layer_stage3.linear.bias, 0)

    def unpatchify(self, x, input_size):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        if isinstance(input_size, list):
            c = self.out_channels
            p = self.input_embedder_grid.patch_size[0]
            h = input_size[0] // p
            w = input_size[1] // p

            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum("nhwpqc->nchpwq", x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
            return imgs

        c = self.out_channels
        p = self.input_embedder_tps0.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, source=None, target=None, seg_feat=None, textline_feat=None,
                textline_mask=None, mask=None, update_feat=None,
                mode=None, tmode=None, path=None,
                seg_cond=None, feat=None, feat_cond=None, line_cond=None, norm_target_mesh=None):
        # x: [1,2,16,16], tps motion
        # t: time step
        # source: source image [1,3,512,512], [0,1]
        # seg mask: [1,384,64,64]
        # line mask: [1,64,64,64]
        # mask: [1,1,512,512], [0,1]
        # init_flow, update_flow: [1,2,512,512]
        # init_feat, update_feat: [1,256,64,64]
        # update_tps_motion: [1,2,16,16]

        # time_step = num_timestep-1
        batch_size = x.shape[0]
        x = [x[:, :, :self.input_size[0] ** 2].reshape([batch_size, 2, self.input_size[0], self.input_size[0]]),
             x[:, :, self.input_size[0] ** 2:self.input_size[0] ** 2 + self.input_size[1] ** 2].reshape(
                 [batch_size, 2, self.input_size[1], self.input_size[1]]),
             x[:, :, -self.input_size[2] * self.input_size[3]:].reshape(
                 [batch_size, 2, self.input_size[2], self.input_size[3]])]

        x_emb = [self.input_embedder_tps0(x[0]) + self.noised_input_pos_embed_tps0,
                 self.input_embedder_tps1(x[1]) + self.noised_input_pos_embed_tps1,
                 self.input_embedder_grid(x[2]) + self.noised_input_pos_embed_grid]

        t = self.t_embedder(t)  # [1, 384]

        if mask is not None:  # [16, 1, 512, 512]  # false
            source_cat = torch.cat([source, mask, textline_mask], dim=1)  # [1, 4, 512, 512]
        else:
            source_cat = source

        feat_ori = self.feature_extractor(source_cat)  # [1, 256, 32, 32]
        update_feat = feat_ori.clone()
        update_cond = self.update_embedder_tps0(update_feat) + self.noised_cond_pos_embed  # [1, 1024, 384]

        tps = []
        tps_up = []

        for idx in range(self.cascade_num):
            pre = x_emb[idx].clone()
            if idx == 0:
                for block in self.blocks_stage1:
                    pre = block(pre, t, update_cond)  # x [b, num_patches, 384]
            elif idx == 1:
                for block in self.blocks_stage2:
                    pre = block(pre, t, update_cond)  # x [b, num
            elif idx == 2:
                for block in self.blocks_stage3:
                    pre = block(pre, t, update_cond)  # x [b, num
            n, _, d = pre.size()
            if self.settings.model.use_decoder:
                pre = pre.transpose(1, 2).contiguous().view(n, d,
                                                            int(self.input_size[idx] / self.input_patch_size),
                                                            int(self.input_size[
                                                                    idx] / self.input_patch_size))  # [b, num_patches, 384] -> [b, 384, num_patches_x, num_patches_y]
                if idx == 0:
                    pre = self.decoder_1(pre)  # [b, 384, num_patches_x, num_patches_y] -> [b, num_patches, 384]
                elif idx == 1:
                    pre = self.decoder_2(pre)  # [b, 384, num_patches_x, num_patches_y] -> [b, num_patches, 384]
                elif idx == 2:
                    pre = self.decoder_3(pre)  # [b, 384, num_patches_x, num_patches_y] -> [b, num_patches, 384]
            else:
                pass

            if idx == 0:
                pre = self.final_layer_stage1(pre, t)  #
                pre = self.unpatchify(pre, self.input_size[0])
            elif idx == 1:
                pre = self.final_layer_stage2(pre, t)  #
                pre = self.unpatchify(pre, self.input_size[1])
            elif idx == 2:
                pre = self.final_layer_stage3(pre, t)  #
                pre = self.unpatchify(pre, [self.input_size[2], self.input_size[3]])

            if idx == 0:
                tps.append(pre)
            else:
                tps.append(pre + tps_up[-1])
            if idx < self.cascade_num - 1:
                if idx == 0:
                    tps_up.append(upsample_tps(tps[idx], self.input_size[idx], self.input_size[idx],
                                               self.input_size[idx + 1], self.input_size[idx + 1],
                                               self.settings.train.image_size, norm_target_mesh[idx],
                                               norm_target_mesh[idx + 1]))
                elif idx == 1:
                    tps_up.append(upsample_tps(tps[idx], self.input_size[idx], self.input_size[idx],
                                               self.input_size[idx + 1], self.input_size[idx + 2],
                                               self.settings.train.image_size, norm_target_mesh[idx],
                                               norm_target_mesh[idx + 1]))
                update_tps_flow, *_ = tps2flow(tps[-1], [feat_ori.shape[-2], feat_ori.shape[-1]],
                                               norm_target=norm_target_mesh[idx], solve_inv=False)
                update_feat = F.grid_sample(feat_ori.to(update_tps_flow.device), update_tps_flow.permute(0, 2, 3, 1),
                                            mode=self.settings.model.interpolation_mode, align_corners=True)
                # update feat of source image
                if idx == 0:
                    update_cond = self.update_embedder_tps1(update_feat) + self.noised_cond_pos_embed  # [1, 1024, 384]
                elif idx == 1:
                    update_cond = self.update_embedder_grid(update_feat) + self.noised_cond_pos_embed  # [1, 1024, 384]

        x = torch.concatenate(
            [tps[0].reshape(batch_size, 2, -1), tps[1].reshape(batch_size, 2, -1), tps[2].reshape(batch_size, 2, -1)],
            dim=-1)
        return x

    @property
    def device(self):
        return next(self.parameters()).device



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, list):
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_S_2_new(**kwargs):
    return DiT(settings=config.settings, depth=6,
               embedding_size=64, hidden_size=384, input_patch_size=1, patch_size=2, num_heads=6, **kwargs)


DiT_models2 = {
    'DiT-S/2_new': DiT_S_2_new
}
