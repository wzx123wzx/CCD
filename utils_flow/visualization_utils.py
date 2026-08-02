import os
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image
from torchvision.utils import save_image

from datasets.utils import flow_viz
from datasets.utils.warping import register_model2
import config.settings as settings
reg_model_bilin = register_model2((512,512), 'bilinear')

def visualize(sample, category, rate, name_dataset, i, batch_vis, source_vis, target_vis, mask):
    os.makedirs(f'vis_{category}/{rate}_1_{name_dataset+2}/pred_flow', exist_ok=True) # pred flow
    os.makedirs(f'vis_{category}/{rate}_1_{name_dataset+2}/gt_flow', exist_ok=True) # gt flow
    os.makedirs(f'vis_{category}/{rate}_1_{name_dataset+2}/src_samples', exist_ok=True) # original source
    os.makedirs(f'vis_{category}/{rate}_1_{name_dataset+2}/trg_samples', exist_ok=True) # original target
    os.makedirs(f'vis_{category}/{rate}_1_{name_dataset+2}/dewarped_pred', exist_ok=True) # pred dewarped
    os.makedirs(f'vis_{category}/{rate}_1_{name_dataset+2}/dewarped_gt', exist_ok=True) # gt dewarped
    os.makedirs(f'vis_{category}/{rate}_1_{name_dataset+2}/mask', exist_ok=True) # mask of matched regions in the target
    
    for j in range(len(sample)):
        flow_vis = sample[j].detach().permute(1,2,0).float().cpu().numpy()
        flow_vis = flow_viz.flow_to_image(flow_vis)
        plt.imsave(f'vis_{category}/{rate}_1_{name_dataset+2}/pred_flow/flow_{i}_{j}.png', flow_vis / 255.0)

        flow_gt_vis = batch_vis[j].detach().permute(1,2,0).float().cpu().numpy()
        flow_gt_vis = flow_viz.flow_to_image(flow_gt_vis)
        plt.imsave(f'vis_{category}/{rate}_1_{name_dataset+2}/gt_flow/gt_{i}_{j}.png', flow_gt_vis / 255.0)

        src = source_vis[j].permute(1, 2, 0).cpu().numpy()
        src = Image.fromarray((src).astype(np.uint8))
        src.save(f'vis_{category}/{rate}_1_{name_dataset+2}/src_samples/src_{i}_{j}.png')

        trg = target_vis[j].permute(1, 2, 0).cpu().numpy()
        trg = Image.fromarray((trg).astype(np.uint8))
        trg.save(f'vis_{category}/{rate}_1_{name_dataset+2}/trg_samples/trg_{i}_{j}.png')
        
        warped_src = warp(source_vis[j:j+1].to(sample.device).float(), sample)
        warped_src_masked = warped_src * mask[j:j+1].float()
        warped_src = warped_src[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_src_masked = warped_src_masked[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_src = Image.fromarray((warped_src).astype(np.uint8))
        warped_src_masked = Image.fromarray((warped_src_masked).astype(np.uint8))
        warped_src.save(f'vis_{category}/{rate}_1_{name_dataset+2}/dewarped_pred/warped_{i}_{j}.png')
        warped_src_masked.save(f'vis_{category}/{rate}_1_{name_dataset+2}/dewarped_pred/warped_masked_{i}_{j}.png')
        
        warped_gt = warp(source_vis[j:j+1].to(batch_vis.device).float(), batch_vis)
        warped_gt_masked = warped_gt * mask[j:j+1].float()
        warped_gt = warped_gt[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_gt_masked = warped_gt_masked[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_gt = Image.fromarray((warped_gt).astype(np.uint8))
        warped_gt_masked = Image.fromarray((warped_gt_masked).astype(np.uint8))
        warped_gt.save(f'vis_{category}/{rate}_1_{name_dataset+2}/dewarped_gt/warped_{i}_{j}.png')
        warped_gt_masked.save(f'vis_{category}/{rate}_1_{name_dataset+2}/dewarped_gt/warped_masked_{i}_{j}.png')

        mask_vis = th.stack((mask[j], mask[j], mask[j]))
        save_image(mask_vis.float(), f'vis_{category}/{rate}_1_{name_dataset+2}/mask/mask_{i}_{j}.png')
    
    



def visualize_dewarping(settings, pred_flow, data, i, source_vis, data_name, ref_flow=None):
    os.makedirs(f'sampling/{settings.eval.dataset_name}/{settings.eval.experiment_name}/pred_flow', exist_ok=True) # pred flow
    os.makedirs(f'sampling/{settings.eval.dataset_name}/{settings.eval.experiment_name}/pred_dewarped', exist_ok=True) # pred dewarped

    flow_vis = flow_viz.vis_flow(pred_flow[0].detach().float().cpu())
    cv2.imwrite(f"sampling/{settings.eval.dataset_name}/{settings.eval.experiment_name}/pred_flow/{data_name}.png", flow_vis)

    warped_image = F.grid_sample(source_vis.to(settings.eval.device),
                               pred_flow.permute(0, 2, 3, 1),
                               mode=settings.model.interpolation_mode, align_corners=True)
    warped_image = warped_image[0].permute(1, 2, 0).detach().cpu().numpy() * 255.

    cv2.imwrite(f"sampling/{settings.eval.dataset_name}/{settings.eval.experiment_name}/pred_dewarped/{data_name}.png", cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))

    if ref_flow is not None:
        os.makedirs(f'vis_hp/{settings.eval.dataset_name}/{settings.eval.experiment_name}/pred_flow_ref', exist_ok=True) # pred flow
        os.makedirs(f'vis_hp/{settings.eval.dataset_name}/{settings.eval.experiment_name}/dewarped_pred_ref', exist_ok=True) # pred dewarped

        # warped_src_ref = warp(source_vis.to(ref_flow.device).float(), ref_flow) # [1, 3, 1629, 981]
        # warped_src_ref = reg_model_bilin([source_vis.to(ref_flow.device).float(), ref_flow])
        # warped_src_ref = warped_src_ref[0].permute(1, 2, 0).detach().cpu().numpy()#*255. # (1873, 1353, 3)
        # warped_src_ref = Image.fromarray((warped_src_ref).astype(np.uint8))
        # warped_src_ref.save(f"vis_hp/{settings.eval.dataset_name}/{settings.eval.experiment_name}/dewarped_pred_ref/warped_{data_path[0].split('/')[-1]}")
        
