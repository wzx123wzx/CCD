import os

import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils_flow.visualization_utils import visualize_dewarping

from .improved_diffusion.gaussian_diffusion import GaussianDiffusion
import torch
from torchvision.utils import save_image as tv_save_image
from .improved_diffusion.gaussian_diffusion import grid2flow
import config.settings


def prepare_eval_data(settings, batch_preprocessing, SIZE, data):
    # data['doc_image']: [b,3,512,512], [0,1]
    # data['doc_image_ori']: [b,3,H,W], [0,1]
    # data['doc_path']: sample path

    if "doc_image_ori" in data:
        image_ori = data["doc_image_ori"]
    else:
        image_ori = None

    _, _, H_ori, W_ori = image_ori.shape

    # data = batch_preprocessing(data)

    source = data['doc_image'].to(settings.eval.device)  # [1, 3, 512, 512]  torch.float32

    data['doc_image_256'] = F.interpolate(input=source.float(), size=256, mode='bilinear')
    source_256 = data['doc_image_256'].to(settings.eval.device)

    if 'correspondence_mask' in data:
        mask = data['correspondence_mask']  # torch.bool [1, 914, 1380]
    else:
        mask = torch.ones((1, 512, 512), dtype=torch.bool).to(settings.eval.device)  # None

    # output dict of data
    # 'source_image': [B, 3, 512, 512], [0,1]
    # 'source_image_256': [B, 3, 256, 256], [0,1], inter version of source_image
    # 'doc_mask': [B, 1, 512, 512], [0,1]
    # 'flow_map': relative gt flow (for supervision), without normalization (gt flow - base)
    # 'flow_map_inter': Intermediate flow (for auxiliary supervision), same as flow, here is base (t=0)-base=0
    # other variables
    # (H_ori, W_ori): (512,512)
    # source: source_image [B, 3, 512, 512]
    # target: None
    # batch_ori: flow_map, [B, 2, 512, 512]
    # batch_ori_inter: flow_map_inter [B, 2, 512, 512]
    # source_256=source_image_256 [B, 3, 256, 256], [0,1]
    # source_vis=source_image [B, 3, 512, 512], [0,1]
    # mask: all 1 mask [1, 512, 512], bool
    return data, H_ori, W_ori, source, source_256, image_ori, mask


def prepare_train_data(settings, batch_preprocessing, SIZE, data):
    # data['doc_image']: [b,3,512,512], [0,1]
    # data['doc_mask']: [b,1,512,512], [0,1]
    # data['doc_bm']: [b,2,512,512], w/o normalization
    # data['doc_target_image']: [b,3,512,512], [0,1]
    # data['doc_norm_bm']: [b,2,512,512], w normalization

    _, _, H_ori, W_ori = data['doc_image'].shape

    # data = batch_preprocessing(data)

    source = data['doc_image'].to(settings.train.device)  # [1, 3, 914, 1380]  torch.float32
    data['doc_image_256'] = F.interpolate(input=source.float(), size=256, mode='bilinear')

    if 'doc_grid2d' in data:
        grid2d = data['doc_grid2d']
    else:
        grid2d = None

    if 'doc_grid3d' in data:
        grid3d = data['doc_grid3d']
    else:
        grid3d = None

    if 'doc_grid_bm' in data:
        grid_bm = data['doc_grid_bm'].to(settings.train.device)  # [b,2,512,512], w normalization
    else:
        grid_bm = None

    if 'correspondence_mask' in data:
        mask = data['correspondence_mask']  # torch.bool [1, 914, 1380]
    else:
        mask = torch.ones((1, 512, 512), dtype=torch.bool).to(settings.train.device)  # None

    # output dict of data
    # 'source_image': [B, 3, 512, 512], [0,1]
    # 'source_image_256': [B, 3, 256, 256], [0,1], inter version of source_image
    # 'doc_mask': [B, 1, 512, 512], [0,1]
    # 'flow_map': relative gt flow (for supervision), without normalization (gt flow - base)
    # 'flow_map_inter': Intermediate flow (for auxiliary supervision), same as flow, here is base (t=0)-base=0
    # other variables
    # (H_ori, W_ori): (512,512)
    # source: source_image [B, 3, 512, 512]
    # target: None
    # batch_ori: flow_map, [B, 2, 512, 512]
    # batch_ori_inter: flow_map_inter [B, 2, 512, 512]
    # source_256=source_image_256 [B, 3, 256, 256], [0,1]
    # source_vis=source_image [B, 3, 512, 512], [0,1]
    # mask: all 1 mask [1, 512, 512], bool
    return data, H_ori, W_ori, source, grid_bm, grid2d, grid3d, mask,


def run_sample_lr_dewarping(
        settings, logger, diffusion, model, source,
        doc_mask, textline_mask,
        seg_feat=None, textline_feat=None, norm_target_mesh=None,
):
    model_kwsettings = {'source': source,
                        'seg_feat': seg_feat, 'textline_feat': textline_feat,
                        'mask': doc_mask, 'textline_mask': textline_mask}
    # [1, 81, 64, 64] [1, 2, 64, 64] [1, 64, 64, 64]
    grid_h, grid_w = settings.model.grid_resolution

    logger.info(f"\nStarting sampling")

    sample, _ = diffusion.ddim_sample_loop(
        model,
        (1, 2, grid_h, grid_w),  # 1,2,8,8
        noise=None,
        clip_denoised=settings.model.clip_denoised,  # false
        model_kwargs=model_kwsettings,
        eta=0.0,
        progress=True,
        denoised_fn=None,
        logger=logger,
        n_batch=settings.model.n_batch,
        norm_target_mesh=norm_target_mesh,
    )

    sample = torch.clamp(sample, min=-1, max=1)  # tps motion, [-1,1]
    return sample


def run_evaluation_docunet(
        settings, logger, val_loader, diffusion: GaussianDiffusion, model,
        pretrained_dewarp_model, pretrained_line_seg_model=None, pretrained_seg_model=None,
        norm_target_mesh=None
):
    os.makedirs(f'sampling/{settings.eval.dataset_name}/{settings.eval.experiment_name}', exist_ok=True)
    # batch_preprocessing = DocBatchPreprocessing(
    #             settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False
    #         )
    batch_preprocessing = None
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    SIZE = None
    train_t = []
    for i, data in pbar:
        embedding_cond_size = settings.model.embedding_cond_size
        data_name = data['doc_name'][0]
        # ref test
        source_288 = F.interpolate(data['doc_image'], size=288, mode='bilinear', align_corners=True).to(
            settings.eval.device)

        with torch.inference_mode():
            # ref_bm, mask_x = pretrained_dewarp_model(source_288)  # [1,2,288,288] 0~288  0~1
            ref_bm, mask_x = pretrained_dewarp_model(data['doc_image'].to(settings.eval.device))  # [1,2,512,512] 0~512 0~1

            ref_flow = ref_bm / 511.0  # [-1, 1]  # [1,2,288,288]
            # cv2.imwrite('val_doc_image_1.png', 255 * data['doc_image'].cpu().squeeze().permute(1,2,0).numpy())
            # cv2.imwrite('val_doc_mask.png', 255 * mask_x.cpu().squeeze().numpy())
        if settings.model.exclude_background:
            mask_512 = mask_x.clone()
            # mask_ori = F.interpolate(mask_512, size=(data['doc_image_ori'].shape[2], data['doc_image_ori'].shape[3]),
            #                          mode='bilinear',
            #                          align_corners=True).to(settings.eval.device)
            mask_288 = F.interpolate(mask_512, size=288,
                                     mode='bilinear',
                                     align_corners=True).to(settings.eval.device)
            data['doc_image'] = data['doc_image'] * mask_512.cpu()
            # data['doc_image_ori'] = data['doc_image_ori'] * mask_ori.cpu()
            source_288 = source_288 * mask_288
            # cv2.imwrite('val_doc_image.png', 255 * data['doc_image'].cpu().squeeze().permute(1, 2, 0).numpy())
            # cv2.imwrite('val_doc_image_ori.png', 255 * data['doc_image_ori'].cpu().squeeze().permute(1, 2, 0).numpy())
            # cv2.imwrite('source_288.png', 255 * source_288.cpu().squeeze().permute(1, 2, 0).numpy())

        (
            data,
            H_ori,  # 1458
            W_ori,  # 1141
            source,  # [1, 3, 512, 512] 0-1
            source_256,  # [1, 3, 256, 256] 0-1
            image_ori,  # [1, 3, 1458, 1141] [0,255]
            mask,  # [1, 512, 512], bool
        ) = prepare_eval_data(settings, batch_preprocessing, SIZE, data)

        with torch.no_grad():
            if settings.train.use_gt_mask == False:
                mskx, d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d = pretrained_seg_model(source)
                hx6 = F.interpolate(hx6, size=embedding_cond_size, mode='bilinear', align_corners=False)
                hx5d = F.interpolate(hx5d, size=embedding_cond_size, mode='bilinear', align_corners=False)
                hx4d = F.interpolate(hx4d, size=embedding_cond_size, mode='bilinear', align_corners=False)
                hx3d = F.interpolate(hx3d, size=embedding_cond_size, mode='bilinear', align_corners=False)
                hx2d = F.interpolate(hx2d, size=embedding_cond_size, mode='bilinear', align_corners=False)
                hx1d = F.interpolate(hx1d, size=embedding_cond_size, mode='bilinear', align_corners=False)

                seg_feat = torch.cat((hx6, hx5d, hx4d, hx3d, hx2d, hx1d), dim=1)  # [1, 384, 64, 64]

                if settings.train.use_line_mask:
                    textline_feat, textline_mask = pretrained_line_seg_model(mskx)  # [1, 64, 256, 256]
                    textline_feat = F.interpolate(textline_feat, size=embedding_cond_size, mode='bilinear',
                                                  align_corners=False)  # [3, 64, 64, 64]
                    # textline_mask = F.interpolate(textline_mask, size=512,
                    #                               mode='bilinear',
                    #                               align_corners=False)
                    # cv2.imwrite('textline_mask.png', 255 * textline_mask.squeeze().cpu().detach().numpy())
            else:
                seg_feat = None
                textline_feat = None

        # logger.info(f"Starting sampling.")

        import time
        begin_train = time.time()
        sample = run_sample_lr_dewarping(
            settings,
            logger,
            diffusion,
            model,
            source,  # [B, 3, 512, 512] 0~1
            mask_x,
            textline_mask,
            seg_feat,
            textline_feat,
            norm_target_mesh,
        )  # sample: [1, 2, 16, 16], tps motion

        train_t.append(time.time() - begin_train)  
        sample = [sample[:, :, :settings.model.tps_resolution[0] ** 2].reshape([1, 2, settings.model.tps_resolution[0], settings.model.tps_resolution[0]]),
             sample[:, :, settings.model.tps_resolution[0] ** 2:settings.model.tps_resolution[0] ** 2 + settings.model.tps_resolution[1] ** 2].reshape(
                 [1, 2, settings.model.tps_resolution[1], settings.model.tps_resolution[1]]),
             sample[:, :, -settings.model.grid_resolution[0] * settings.model.grid_resolution[1]:].reshape(
                 [1, 2, settings.model.grid_resolution[0], settings.model.grid_resolution[1]])]
        pred_grid_flow, source_mesh = grid2flow(sample[-1], [H_ori, W_ori],
                                                norm_target=norm_target_mesh[-1])  # tps flow [1,2,H_ori, W_ori], [-1,1]

        if settings.train.visualize:
            visualize_dewarping(settings, pred_grid_flow, data, i, image_ori, data_name, ref_flow)

    print(len(train_t))
    print("Elapsed time:{:.2f} avg_second ".format(sum(train_t) / len(train_t)))


def coords_grid_tensor(perturbed_img_shape):
    im_x, im_y = np.mgrid[0:perturbed_img_shape[0] - 1:complex(perturbed_img_shape[0]),
                 0:perturbed_img_shape[1] - 1:complex(perturbed_img_shape[1])]
    coords = np.stack((im_y, im_x), axis=2)  
    coords = torch.from_numpy(coords).float().permute(2, 0, 1).to(config.settings.eval.device)  # (2, 512, 512)
    return coords.unsqueeze(0)  # [2, 512, 512]


def validate(local_rank, args, val_loader, model, criterion):
    for i, sample in enumerate(val_loader):
        input1, label = sample  # [2, 3, 288, 288],[2, 2, 288, 288]
        input1 = input1.to(local_rank, non_blocking=True)
        label = label.to(local_rank, non_blocking=True)
        # label = (label/288.0-0.5)*2

        with torch.no_grad():
            output = model(input1)  # [3b, 2, 288, 288]
            # loss = F.l1_loss(output, label) 
            # test point
            # bm_test=(output/288.0-0.5)*2
            bm_test = (output / 992.0 - 0.5) * 2
            label = (label / 992.0 - 0.5) * 2
            # bm_test = output
            bm_test = F.interpolate(bm_test, size=(1000, 1000), mode='bilinear', align_corners=True)
            label = F.interpolate(label, size=(1000, 1000), mode='bilinear', align_corners=True)
            input1 = F.interpolate(input1, size=(1000, 1000), mode='bilinear', align_corners=True)
            regis_image1 = F.grid_sample(input=input1, grid=bm_test.permute(0, 2, 3, 1), align_corners=True)
            regis_image2 = F.grid_sample(input=input1, grid=label.permute(0, 2, 3, 1), align_corners=True)

            # regis_image2 = F.grid_sample(input=a_sample[None], grid=bm_test[None].permute(0,2,3,1), align_corners=True)
            tv_save_image(input1[0], "backup/test/ori.png")
            tv_save_image(regis_image1[0], "backup/test/aaa.png")
            tv_save_image(regis_image2[0], "backup/test/gt.png")

            # warped_src = warp(source_vis.to(sample.device).float(), sample) # [1, 3, 1629, 981]
            # warped_src = warped_src[0].permute(1, 2, 0).detach().cpu().numpy() # (1873, 1353, 3)
            # warped_src = Image.fromarray((warped_src).astype(np.uint8))
            # warped_src.save(f"vis_hp/{config.settings.env.eval_dataset_name}/{config.settings.name}/dewarped_pred/warped_{data_path[0].split('/')[-1]}")

    return None
