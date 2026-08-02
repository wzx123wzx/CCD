import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_settings.ccd.evaluation import run_sample_lr_dewarping
from train_settings.ccd.improved_diffusion import logger
from train_settings.ccd.improved_diffusion.script_util import create_model_and_diffusion
from train_settings.ccd.improved_diffusion.gaussian_diffusion import grid2flow
from train_settings.models.geotr.geotr_core import GeoTr_Seg_Inf, reload_segmodel_medium, Seg
from train_settings.models.geotr.unet_model import UNet
import config.settings
import numpy as np


# ---------- mesh helpers (same as val.py) ----------

def coords_grid_tensor(shape):
    im_x, im_y = np.mgrid[0:shape[0] - 1:complex(shape[0]),
                           0:shape[1] - 1:complex(shape[1])]
    coords = np.stack((im_y, im_x), axis=2)
    return torch.from_numpy(coords).float().permute(2, 0, 1).unsqueeze(0)


def get_tps_norm_target_mesh(tps_resolution, img_resolution, device):
    H, W = img_resolution
    base = coords_grid_tensor((H, W)).to(device)
    norm_target_mesh = []
    for resolution in tps_resolution:
        rows = np.round((H - 1) * np.linspace(0, 1, resolution)).astype(np.int32)[:, None]
        cols = np.round((W - 1) * np.linspace(0, 1, resolution)).astype(np.int32)[None, :]
        mesh = base[:, :, rows, cols][0].unsqueeze(0).clone().permute(0, 2, 3, 1)
        norm_target_mesh.append(torch.stack([mesh[..., 0] * 2. / W - 1,
                                             mesh[..., 1] * 2. / H - 1], dim=3))
    return norm_target_mesh


def get_grid_norm_target_mesh(grid_resolution, img_resolution, device):
    H, W = img_resolution
    base = coords_grid_tensor((H, W)).to(device)
    rows = np.round((H - 1) * np.linspace(0, 1, grid_resolution[0])).astype(np.int32)[:, None]
    cols = np.round((W - 1) * np.linspace(0, 1, grid_resolution[1])).astype(np.int32)[None, :]
    mesh = base[:, :, rows, cols][0].unsqueeze(0).clone().permute(0, 2, 3, 1)
    return torch.stack([mesh[..., 0] * 2. / W - 1,
                        mesh[..., 1] * 2. / H - 1], dim=3)


# ---------- main ----------

def load_models(settings):
    model, diffusion = create_model_and_diffusion(
        settings=settings, device=settings.eval.device)
    setattr(diffusion, "settings", settings)

    pretrained_dewarp_model = GeoTr_Seg_Inf()
    reload_segmodel_medium(pretrained_dewarp_model.msk, settings.resume.seg_model_path)
    pretrained_dewarp_model.to(settings.eval.device).eval()

    pretrained_line_seg_model = UNet(n_channels=3, n_classes=1)
    line_ckpt = torch.load(settings.resume.line_seg_model_path, map_location='cpu')['model']
    pretrained_line_seg_model.load_state_dict(line_ckpt, strict=True)
    pretrained_line_seg_model.to(settings.eval.device).eval()

    pretrained_seg_model = Seg()
    seg_ckpt = torch.load(settings.resume.new_seg_model_path, map_location='cpu')
    pretrained_seg_model.load_state_dict(seg_ckpt, strict=True)
    pretrained_seg_model.to(settings.eval.device).eval()

    model.load_state_dict(torch.load(settings.resume.model_path, map_location='cpu'), strict=False)
    model.to(settings.eval.device).eval()

    return model, diffusion, pretrained_dewarp_model, pretrained_line_seg_model, pretrained_seg_model


def preprocess(image_path, device):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_ori = img_bgr[:, :, ::-1].copy()                          # BGR→RGB, uint8 [H,W,3]
    H_ori, W_ori = img_ori.shape[:2]

    img_512 = cv2.resize(img_ori, (512, 512))
    source = torch.from_numpy(img_512).float().permute(2, 0, 1) / 255.  # [3,512,512] [0,1]
    source = source.unsqueeze(0).to(device)                              # [1,3,512,512]
    return source, img_ori, H_ori, W_ori


def infer(image_path, output_path, settings):
    device = settings.eval.device

    print("Loading models...")
    model, diffusion, pretrained_dewarp_model, pretrained_line_seg_model, pretrained_seg_model = \
        load_models(settings)

    print(f"Processing: {image_path}")
    source, img_ori, H_ori, W_ori = preprocess(image_path, device)

    # --- foreground segmentation + background removal ---
    with torch.inference_mode():
        _, mask_x = pretrained_dewarp_model(source)

    if settings.model.exclude_background:
        source = source * mask_x

    # --- textline segmentation ---
    with torch.no_grad():
        mskx, *_ = pretrained_seg_model(source)
        _, textline_mask = pretrained_line_seg_model(mskx)
        textline_mask = F.interpolate(textline_mask, size=512, mode='bilinear', align_corners=False)

    # --- build normalised target meshes ---
    norm_target_mesh = get_tps_norm_target_mesh(
        settings.model.tps_resolution, settings.train.image_size, device)
    norm_target_mesh.append(
        get_grid_norm_target_mesh(settings.model.grid_resolution, settings.train.image_size, device))

    # --- diffusion sampling ---
    sample = run_sample_lr_dewarping(
        settings, logger, diffusion, model,
        source, mask_x, textline_mask,
        seg_feat=None, textline_feat=None,
        norm_target_mesh=norm_target_mesh,
    )

    # --- convert grid motion to flow, warp original image ---
    sample_list = [
        sample[:, :, :settings.model.tps_resolution[0] ** 2].reshape(
            1, 2, settings.model.tps_resolution[0], settings.model.tps_resolution[0]),
        sample[:, :, settings.model.tps_resolution[0] ** 2:
               settings.model.tps_resolution[0] ** 2 + settings.model.tps_resolution[1] ** 2].reshape(
            1, 2, settings.model.tps_resolution[1], settings.model.tps_resolution[1]),
        sample[:, :, -settings.model.grid_resolution[0] * settings.model.grid_resolution[1]:].reshape(
            1, 2, settings.model.grid_resolution[0], settings.model.grid_resolution[1]),
    ]

    pred_flow, _ = grid2flow(sample_list[-1], [H_ori, W_ori], norm_target=norm_target_mesh[-1])

    img_ori_t = torch.from_numpy(img_ori).float().permute(2, 0, 1).unsqueeze(0) / 255.  # [1,3,H,W]
    dewarped = F.grid_sample(img_ori_t.to(device), pred_flow.permute(0, 2, 3, 1),
                             mode='bilinear', align_corners=True)

    # --- save ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_np = (dewarped[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_path, out_np[:, :, ::-1])  # RGB→BGR for cv2
    print(f"Saved dewarped image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image document dewarping inference.")

    parser.add_argument(
        "--image", type=str, required=True, 
        help="Path to the input distorted document image (e.g. input/doc.jpg)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for the output dewarped image (default: <input_name>_dewarped.png)"
    )
    parser.add_argument(
        "--model-path", type=str, default='checkpoints/best_model.pt',
        help="Override the model checkpoint path (default: settings.resume.model_path)"
    )
    parser.add_argument(
        "--device", type=str, default='cuda:0',
        help="Compute device, e.g. 'cuda:0' or 'cpu' (default: settings.eval.device)"
    )

    args = parser.parse_args()

    settings = config.settings
    if args.model_path:
        settings.resume.model_path = args.model_path
    if args.device:
        import torch
        settings.eval.device = torch.device(args.device)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(os.path.dirname(args.image), f"{base}_dewarped.png")

    logger.configure(dir="sampling/infer")
    infer(args.image, args.output, settings)
