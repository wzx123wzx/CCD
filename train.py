import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from termcolor import colored
from torch.utils.data import ConcatDataset

from datasets.load_pre_made_dataset import Doc3dGrid_Dataset, UVDoc_Dataset
from datasets.batch_processing import GLUNetBatchPreprocessing
from utils_data.image_transforms import ArrayToTensor
from utils_data.loaders import Loader
from train_settings.models.geotr.geotr_core import Seg
from train_settings.models.geotr.unet_model import UNet

from train_settings.ccd.improved_diffusion import logger
from train_settings.ccd.improved_diffusion.resample import create_named_schedule_sampler
from train_settings.ccd.improved_diffusion.script_util import create_model_and_diffusion
    
from train_settings.ccd.improved_diffusion.train_util import TrainLoop
import config.settings

np.set_printoptions(sys.maxsize)


def run(settings):
    os.environ['PYTORCH_DOWNLOAD_MIRROR'] = 'https://mirrors.tuna.tsinghua.edu.cn/pytorch'

    torch.cuda.set_device(settings.train.device)
    logger.configure(dir="checkpoints/{}_{}".format(settings.train.train_mode, settings.train.dataset_name))

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(settings=config.settings, device=config.settings.train.device)

    # model.train()
    # print(model)
    if settings.resume.resume_checkpoint:
        model.load_state_dict(torch.load(settings.resume.resume_checkpoint, map_location="cpu"), strict=False)
        print('load checkpoint from: {}'.format(settings.resume.resume_checkpoint))

    print(f"Setting device to {settings.train.device}")
    model = model.to(settings.train.device)

    schedule_sampler = create_named_schedule_sampler(settings.train.schedule_sampler, diffusion)

    pretrained_line_seg_model = UNet(n_channels=3, n_classes=1)
    pretrained_seg_model = Seg()

    # freeze the foreground and text-line segmentation network

    line_model_ckpt = torch.load(settings.resume.line_seg_model_path, map_location="cpu")['model']
    pretrained_line_seg_model.load_state_dict(line_model_ckpt, strict=True)
    pretrained_line_seg_model.to(settings.train.device)
    pretrained_line_seg_model.eval()
    seg_model_ckpt = torch.load(settings.resume.new_seg_model_path, map_location="cpu")
    pretrained_seg_model.load_state_dict(seg_model_ckpt, strict=True)
    pretrained_seg_model.to(settings.train.device)
    pretrained_seg_model.eval()
    logger.log("creating data loader...")

    # 1. Define training and validation datasets
    # datasets, pre-processing of the images is done within the network function !
    if settings.train.dataset_name == 'uvdoc':
        img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        train_dataset, _ = UVDoc_Dataset(settings=settings,
                                                    root=settings.train.train_uvdoc_dataset_path,
                                                    source_image_transform=img_transforms,
                                                    target_image_transform=None,
                                                    flow_transform=flow_transform,
                                                    split=1,
                                                    get_mapping=False)
    elif settings.train.dataset_name == 'doc3d_grid':
        img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        train_dataset, _ = Doc3dGrid_Dataset(settings=settings,
                                                        root=settings.train.train_doc3d_dataset_path,
                                                        source_image_transform=img_transforms,
                                                        target_image_transform=None,
                                                        flow_transform=flow_transform,
                                                        split=1,
                                                        get_mapping=False)
    elif settings.train.dataset_name == 'mix':
        img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        train_doc3d_dataset, _ = Doc3dGrid_Dataset(settings=settings,
                                                              root=settings.train.train_doc3d_dataset_path,
                                                              source_image_transform=img_transforms,
                                                              target_image_transform=None,
                                                              flow_transform=flow_transform,
                                                              split=1,
                                                              get_mapping=False)
        train_uvdoc_dataset, _ = UVDoc_Dataset(settings=settings,
                                                          root=settings.train.train_uvdoc_dataset_path,
                                                          source_image_transform=img_transforms,
                                                          target_image_transform=None,
                                                          flow_transform=flow_transform,
                                                          split=1,
                                                          get_mapping=False)
        train_dataset = ConcatDataset([train_doc3d_dataset, train_uvdoc_dataset])

    train_loader = Loader('train', train_dataset, batch_size=settings.train.batch_size, shuffle=True,
                          drop_last=False, training=True, num_workers=settings.train.n_threads)

    # Setting dataset name into diffusion because of the semantic setting.
    setattr(diffusion, 'dataset', settings.train.dataset_name)

    # but better results are obtained with using simple bilinear interpolation instead of deconvolutions.
    print(colored('==> ', 'blue') + 'model created.')

    logger.log("training...")
    batch_preprocessing = GLUNetBatchPreprocessing(settings, apply_mask=False, apply_mask_zero_borders=False,
                                                   sparse_ground_truth=False)

    # 4. Define loss module    
    TrainLoop(
        model=model,
        pretrained_dewarp_model=pretrained_seg_model,
        pretrained_line_seg_model=pretrained_line_seg_model,
        diffusion=diffusion,
        settings=settings,
        schedule_sampler=schedule_sampler,
        batch_preprocessing=batch_preprocessing,
        data=train_loader).run_loop_dewarping()


if __name__ == "__main__":
    run(config.settings)
