import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import datasets
from utils_data.image_transforms import ArrayToTensor

from train_settings.ccd.evaluation import run_evaluation_docunet
from train_settings.ccd.improved_diffusion import logger
from train_settings.ccd.improved_diffusion.script_util import create_model_and_diffusion
                                                              
from train_settings.models.geotr.geotr_core import GeoTr_Seg_Inf, \
    reload_segmodel, reload_segmodel_medium, Seg

from datasets.doc_dataset.doc_benchmark import Doc_dewarping_Data1

from train_settings.models.geotr.unet_model import UNet
import config.settings
import numpy as np


class WrappedDiffusionModel(torch.nn.Module):
    def __init__(self, model, t, model_kwargs):
        super().__init__()
        self.model = model
        self.t = t
        self.model_kwargs = model_kwargs

    def forward(self, x):
        return self.model(x, self.t, **self.model_kwargs)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def run(settings):
    print('************')
    print('evaluate on {}.'.format(settings.eval.dataset_name))
    print('************')
    logger.configure(dir="sampling/{}".format(settings.eval.dataset_name))
    logger.log(f"Corruption Disabled. Evaluating on Original {settings.eval.eval_dataset_path}")
    logger.log("Loading model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        settings=settings,
        device=settings.eval.device)
    setattr(diffusion, "settings", settings)

    # pretrained_dewarp_model = GeoTr(num_attn_layers=6, num_token=(288//8)**2)
    # seg document image mask
    pretrained_dewarp_model = GeoTr_Seg_Inf()
    reload_segmodel_medium(pretrained_dewarp_model.msk, settings.resume.seg_model_path)
    pretrained_dewarp_model.to(settings.eval.device)
    pretrained_dewarp_model.eval()

    if settings.train.use_line_mask:
        pretrained_line_seg_model = UNet(n_channels=3, n_classes=1)
        line_model_ckpt = torch.load(settings.resume.line_seg_model_path, map_location='cpu')['model']
        pretrained_line_seg_model.load_state_dict(line_model_ckpt, strict=True)
        pretrained_line_seg_model.to(settings.eval.device)
        pretrained_line_seg_model.eval()

        pretrained_seg_model = Seg()
        seg_model_ckpt = torch.load(settings.resume.new_seg_model_path, map_location='cpu')
        pretrained_seg_model.load_state_dict(seg_model_ckpt, strict=True)
        pretrained_seg_model.to(settings.eval.device)
        pretrained_seg_model.eval()

    model.cpu().load_state_dict(torch.load(settings.resume.model_path, map_location="cpu"), strict=False)
    logger.log(f"Model loaded with {settings.resume.model_path}")
    model.to(settings.eval.device)
    print(get_parameter_number(model))
    model.eval()

    logger.log("Creating data loader...")
    logger.info('\n:============== Logging Configs ==============')
    print(settings.eval.eval_dataset_path)
    for key, value in settings.resume.__dict__.items():
        if key in ['model_path', 'timestep_respacing']:
            logger.info(f"\t{key}:\t{value}")
    for key, value in settings.eval.__dict__.items():
        if key in ['eval_dataset_path']:
            logger.info(f"\t{key}:\t{value}")
    logger.info(':===============================================\n')

    # print(settings.env.eval_dataset)
    if settings.eval.dataset_name == "docunet" or settings.eval.dataset_name == "dir300" or settings.eval.dataset_name == "uvdoc" or settings.eval.dataset_name == "docreal":
        # 1. Define training and validation datasets
        input_transform = transforms.Compose([ArrayToTensor(get_float=True)])  # only put channel first
        test_set = datasets.Doc_benchmark(
            settings.eval.eval_dataset_path,
            input_transform,
        )

        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                                 drop_last=False, num_workers=8)
        logger.info(f"Starting sampling")
        norm_target_mesh = get_tps_norm_target_mesh(settings.model.tps_resolution,
                                                    settings.train.image_size, settings.eval.device)
        norm_grid_target_mesh = get_grid_norm_target_mesh(settings.model.grid_resolution,
                                                          settings.train.image_size, settings.eval.device)
        norm_target_mesh.append(norm_grid_target_mesh)
        run_evaluation_docunet(
            settings, logger, test_loader, diffusion, model, pretrained_dewarp_model, pretrained_line_seg_model,
            pretrained_seg_model,
            norm_target_mesh=norm_target_mesh)
        # have different target mesh if train on different dataset (uvdoc or doc3d)
    elif settings.eval.dataset_name == "doc_val":
        val_set = Doc_dewarping_Data1(root_path=settings.eval.eval_dataset_path, transforms=None, resolution=288,
                                      model_setting="doctr")
        val_loader = DataLoader(val_set, batch_size=1, num_workers=4,
                                drop_last=False, pin_memory=True, shuffle=False)
        # prec1 = validate(val_loader, pretrained_dewarp_model)

    logger.log("sampling complete")


def get_norm_target_mesh(tps_resolution, img_resolution, device):
    # grid_bm: [b,2,512,512], backward warping flow (-1,1)
    # TPS grid resolution (default same in x and y)
    # target mesh: rigid mesh
    # source mesh: predicted mesh
    # tps motion is (target mesh - source mesh)
    H, W = img_resolution
    norm_target_mesh = []
    base = coords_grid_tensor((H, W)).repeat(1, 1, 1, 1).to(device)
    for resolution in tps_resolution:
        indices_row = np.round(0 + (H - 1) * np.linspace(0, 1, resolution)).astype(np.int32)
        indices_col = np.round(0 + (W - 1) * np.linspace(0, 1, resolution)).astype(np.int32)
        rows = indices_row[:, np.newaxis]
        cols = indices_col[np.newaxis, :]
        target_mesh = base[:, :, rows, cols]
        target_mesh = target_mesh[0].unsqueeze(0).clone().permute(0, 2, 3, 1)
        mesh_w = target_mesh[..., 0] * 2. / W - 1
        mesh_h = target_mesh[..., 1] * 2. / H - 1
        norm_target_mesh.append(torch.stack([mesh_w, mesh_h], dim=3))  # [b,16,16,2],[-1,1]
    return norm_target_mesh


def get_tps_norm_target_mesh(tps_resolution, img_resolution, device):
    # grid_bm: [b,2,512,512], backward warping flow (-1,1)
    # TPS grid resolution (default same in x and y)
    # target mesh: rigid mesh
    # source mesh: predicted mesh
    # tps motion is (target mesh - source mesh)
    H, W = img_resolution
    norm_target_mesh = []
    base = coords_grid_tensor((H, W)).repeat(1, 1, 1, 1).to(device)
    for resolution in tps_resolution:
        indices_row = np.round(0 + (H - 1) * np.linspace(0, 1, resolution)).astype(np.int32)
        indices_col = np.round(0 + (W - 1) * np.linspace(0, 1, resolution)).astype(np.int32)
        rows = indices_row[:, np.newaxis]
        cols = indices_col[np.newaxis, :]
        target_mesh = base[:, :, rows, cols]
        target_mesh = target_mesh[0].unsqueeze(0).clone().permute(0, 2, 3, 1)
        mesh_w = target_mesh[..., 0] * 2. / W - 1
        mesh_h = target_mesh[..., 1] * 2. / H - 1
        norm_target_mesh.append(torch.stack([mesh_w, mesh_h], dim=3))  # [b,16,16,2],[-1,1]
    return norm_target_mesh


def get_grid_norm_target_mesh(grid_resolution, img_resolution, device):
    # grid_bm: [b,2,512,512], backward warping flow (-1,1)
    # TPS grid resolution (default same in x and y)
    # target mesh: rigid mesh
    # source mesh: predicted mesh
    # tps motion is (target mesh - source mesh)
    H, W = img_resolution
    base = coords_grid_tensor((H, W)).repeat(1, 1, 1, 1).to(device)

    indices_row = np.round(0 + (H - 1) * np.linspace(0, 1, grid_resolution[0])).astype(np.int32)
    indices_col = np.round(0 + (W - 1) * np.linspace(0, 1, grid_resolution[1])).astype(np.int32)
    rows = indices_row[:, np.newaxis]
    cols = indices_col[np.newaxis, :]
    target_mesh = base[:, :, rows, cols]
    target_mesh = target_mesh[0].unsqueeze(0).clone().permute(0, 2, 3, 1)
    mesh_w = target_mesh[..., 0] * 2. / W - 1
    mesh_h = target_mesh[..., 1] * 2. / H - 1
    norm_target_mesh = torch.stack([mesh_w, mesh_h], dim=3)  # [-1,1]
    return norm_target_mesh


def coords_grid_tensor(perturbed_img_shape):
    im_x, im_y = np.mgrid[0:perturbed_img_shape[0] - 1:complex(perturbed_img_shape[0]),
                 0:perturbed_img_shape[1] - 1:complex(perturbed_img_shape[1])]
    coords = np.stack((im_y, im_x), axis=2)
    coords = torch.from_numpy(coords).float().permute(2, 0, 1)
    return coords.unsqueeze(0)


if __name__ == "__main__":
    run(config.settings)
