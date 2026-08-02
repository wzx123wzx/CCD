# Python standard library imports
import copy
import functools
import os

from torch.utils.tensorboard import SummaryWriter
import datetime
import blobfile as bf
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ..evaluation import prepare_train_data
from . import logger
from .fp16_util import (make_master_params, master_params_to_model_params,
                        model_grads_to_master_grads, unflatten_master_params,
                        zero_grad)
from utils_transform.tps_utils import draw_mesh_on_warp
from .nn import update_ema
from .resample import UniformSampler
from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
            self,
            *,
            model,
            pretrained_dewarp_model,
            pretrained_line_seg_model,
            diffusion,
            settings,
            batch_preprocessing,
            data,
            schedule_sampler=None,
    ):
        self.model = model
        self.pretrained_dewarp_model = pretrained_dewarp_model
        self.pretrained_line_seg_model = pretrained_line_seg_model
        self.diffusion = diffusion
        self.settings = settings
        self.data = data
        self.batch_size = self.settings.train.batch_size
        self.microbatch = self.settings.train.microbatch if self.settings.train.microbatch > 0 else self.batch_size
        self.lr = self.settings.train.lr
        self.ema_rate = (
            [self.settings.train.ema_rate]
            if isinstance(self.settings.train.ema_rate, float)
            else [float(x) for x in self.settings.train.ema_rate.split(",")]
        )
        self.log_interval = self.settings.log.log_interval
        self.save_interval = self.settings.log.save_interval
        self.tb_interval = self.settings.log.tb_interval
        self.resume_checkpoint = self.settings.resume.resume_checkpoint
        self.use_fp16 = self.settings.train.use_fp16
        self.fp16_scale_growth = self.settings.train.fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = self.settings.train.weight_decay
        self.lr_anneal_steps = self.settings.train.lr_anneal_steps

        self.step = 0
        self.resume_step = self.settings.log.resume_step
        self.use_gt_mask = self.settings.train.use_gt_mask
        self.use_init_flow = self.settings.train.use_init_flow
        self.use_line_mask = self.settings.train.use_line_mask
        self.train_mode = self.settings.train.train_mode

        self.global_batch = self.batch_size
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print(f"Resuming from checkpoint with optimizer at step {self.resume_step}")
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            print("Initializing training from scratch (no resume)")
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model

        self.batch_processing = batch_preprocessing
        list_time = list(str(datetime.datetime.now()))[:16]
        list_time[10] = '-'
        str_time = ''.join(list_time)
        self.writer = SummaryWriter(log_dir='{}/{}'.format(bf.join(get_blob_logdir()), str_time))
        # self.writer = SummaryWriter(log_dir='{}'.format(bf.join(get_blob_logdir())))

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            state_dict = self._load_state_dict(resume_checkpoint)
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.settings.train.device)  # Move to local device

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"Loading EMA parameters from {ema_checkpoint}")
            state_dict = self._load_state_dict(ema_checkpoint)
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(opt_checkpoint, map_location=self.settings.train.device)
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def coords_grid_tensor(self, perturbed_img_shape):
        im_x, im_y = np.mgrid[0:perturbed_img_shape[0] - 1:complex(perturbed_img_shape[0]),
                     0:perturbed_img_shape[1] - 1:complex(perturbed_img_shape[1])]
        coords = np.stack((im_y, im_x), axis=2)  
        coords = torch.from_numpy(coords).float().permute(2, 0, 1).to(self.settings.train.device)  # (2, 512, 512)
        return coords.unsqueeze(0)  

    def run_loop_dewarping(self):
        self.embedding_cond_size = self.settings.model.embedding_cond_size
        self.grid_resolution = self.settings.model.grid_resolution
        self.radius = self.settings.model.radius

        self.base64 = self.coords_grid_tensor((64, 64)) / 63 * 2 - 1    # [1,2,64,64]

        while True:
            if self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
                break

            indices = list(range(0, len(self.data)))
            SIZE = None

            batch_preprocessing = None
            for i, data in zip(indices, self.data):
                # data['doc_image']:        [b,3,512,512],   [0,1]  - document image
                # data['doc_mask']:         [b,1,512,512],   [0,1]  - foreground mask
                # data['doc_target_image']: [b,3,512,512],   [0,1]  - flat target image
                # data['doc_grid2d']:       [b,45,31,2],            - 2D GT mesh, pixel coords [0,511]
                # data['doc_grid3d']:       [b,45,31,3],            - 3D GT mesh
                # data['doc_grid_bm']:      [b,2,512,512],          - normalized backward mapping [-1,1]
                # data['doc_path']:         list of str             - source image paths
                (
                    data,
                    H_ori,
                    W_ori,
                    source,
                    grid_bm,
                    grid2d,
                    grid3d,
                    mask
                ) = prepare_train_data(self.settings, batch_preprocessing, SIZE, data)
                with torch.no_grad():
                    if self.use_gt_mask == False:
                        mskx, d0, *_ = self.pretrained_dewarp_model(source)
                        # [b,384,512,512]
                        if self.use_line_mask:
                            # [b,64,512,512]
                            _, textline_mask = self.pretrained_line_seg_model(mskx)
                            textline_mask = F.interpolate(textline_mask, size=512,
                                                         mode='bilinear',
                                                         align_corners=False)
                    else:
                        seg_feat = None
                        textline_feat = None

                cond = {'source': source,
                        'tmode': self.train_mode,
                        'mask': data['doc_mask'].to(source.device),
                        'textline_mask': textline_mask,
                        'mode': "train", 'path': data['doc_path']}
                self.run_step(grid_bm, grid2d, grid3d, mask, cond)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def get_gpu_memory_usage(self, device_id=None):
        allocated_bytes = torch.cuda.memory_allocated(0)
        return allocated_bytes

    def run_step(self, grid_bm, grid2d, grid3d, mask, cond):
        self.forward_backward(grid_bm, grid2d, grid3d, mask, cond)

        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, grid_bm, grid2d, grid3d, mask, cond):
        # grid_bm: [b,2,512,512]  - normalized backward mapping [-1,1]
        # mask:    [1,512,512]
        # cond:    dict
        # cond['source']:        [b,3,512,512], [0,1]
        # cond['mask']:          [b,1,512,512], [0,1]
        # cond['textline_mask']: [b,1,512,512]
        zero_grad(self.model_params)
        for i in range(0, cond['source'].shape[0], self.microbatch):
            if grid_bm is not None:
                micro_grid_bm = grid_bm[i: i + self.microbatch].to(self.settings.train.device)
            else:
                micro_grid_bm = None

            if grid2d is not None:
                micro_grid2d = grid2d[i: i + self.microbatch].to(self.settings.train.device)
            else:
                micro_grid2d = None

            if grid3d is not None:
                micro_grid3d = grid3d[i: i + self.microbatch].to(self.settings.train.device)
            else:
                micro_grid3d = None

            # sample time step (0~T-1) for each batch
            last_batch = (i + self.microbatch) >= grid2d.shape[0]
            t, weights = self.schedule_sampler.sample(grid2d.shape[0], self.settings.train.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_grid_bm,
                micro_grid2d,
                micro_grid3d,
                mask,
                t,
                model_kwargs=cond,
            )

            if last_batch or not self.use_ddp:  # true
                losses, tb_terms = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses, tb_terms = compute_losses()

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)

            if self.step % self.tb_interval == 0:
                if self.step % (5 * self.tb_interval) == 0:
                    self.write_tensorboard(losses, imgs=tb_terms)
                    # self.write_tensorboard(losses, imgs=None)
                else:
                    self.write_tensorboard(losses, imgs=None)

    def optimize_fp16(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is None:
                continue
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def write_tensorboard(self, losses, imgs):
        if losses is not None:
            self.writer.add_scalar('total_loss', losses['loss'].item(), self.step + self.resume_step)
            if self.settings.train.use_mesh_motion_l2_loss:
                self.writer.add_scalar('mesh_motion_l2_stage1_loss', losses['mesh_motion_l2_stage1'].item(),
                                       self.step + self.resume_step)
                self.writer.add_scalar('mesh_motion_l2_stage2_loss', losses['mesh_motion_l2_stage2'].item(),
                                       self.step + self.resume_step)
                self.writer.add_scalar('mesh_motion_l2_stage3_loss', losses['mesh_motion_l2_stage3'].item(),
                                       self.step + self.resume_step)
            if self.settings.train.use_axis_loss:
                self.writer.add_scalar('axis_loss_stage1_loss', losses['axis_stage1'].item(),
                                       self.step + self.resume_step)
                self.writer.add_scalar('axis_loss_stage2_loss', losses['axis_stage2'].item(),
                                       self.step + self.resume_step)
                self.writer.add_scalar('axis_loss_stage3_loss', losses['axis_stage3'].item(),
                                       self.step + self.resume_step)
        if imgs is not None:

            gt_mesh_on_source_img = draw_mesh_on_warp(np.array(imgs["source_img"].permute(1, 2, 0).cpu().detach()),
                                                      imgs["gt_mesh_grid"])
            mesh_on_source_img = draw_mesh_on_warp(np.array(imgs["source_img"].permute(1, 2, 0).cpu().detach()),
                                                   imgs["source_mesh_grid"])
            warped_source_img = F.grid_sample(imgs["source_img"].unsqueeze(0),
                                                  imgs["pred_grid_flow"].unsqueeze(0).permute(0, 2, 3, 1), mode='bilinear',
                                                  align_corners=True)
            self.writer.add_image('gt_mesh_on_source_img', gt_mesh_on_source_img.transpose((2, 0, 1)) / 255,
                                  self.step + self.resume_step)
            self.writer.add_image('pred_mesh_on_source_img', mesh_on_source_img.transpose((2, 0, 1)) / 255,
                                  self.step + self.resume_step)
            self.writer.add_image('warped_source_img', warped_source_img.squeeze().cpu().detach(),
                                  self.step + self.resume_step)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)

            logger.log(f"Saving model (rate: {rate})...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            torch.save(state_dict, '{}/{}'.format(bf.join(get_blob_logdir()), filename))

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # save optimizer state
        # if dist.get_rank() == 0:
        #     with bf.BlobFile(
        #         bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
        #         "wb",
        #     ) as f:
        #         torch.save(self.opt.state_dict(), f)
        #
        # dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

    # Helper: get current device (GPU if available, else CPU)
    def _get_device(self, cuda_idx=0):
        return torch.device("cuda:{}".format(cuda_idx) if torch.cuda.is_available() else "cpu")

    # Helper: load state dict without distributed logic
    def _load_state_dict(self, path):
        return torch.load(path, map_location=self.settings.train.device)


def parse_resume_step_from_filename(filename):
    """Parse filenames like 'path/to/modelNNNNNN.pt' to get step count"""
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    """No distributed storage logic for single-GPU"""
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        if ts.dim() == 0:
            ts = ts.unsqueeze(0)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            # quartile = int(4 * sub_t / diffusion.num_timesteps)
            # logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

            # only report loss at different sample time step here
            logger.logkv_mean(f"{key}_t{sub_t}", sub_loss)
