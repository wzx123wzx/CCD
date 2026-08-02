import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import cv2
import h5py
import os
device = torch.device("cuda:0")


def make_doc3d_dataset_list(dir):
    # edit for my dataset folder structure
    sub_dataset_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    dataset_list = []
    for idx in range(1,22):
        for sample_name in os.listdir('{}/img/{}'.format(dir, idx)):
            sample_name = sample_name[:-4]
            img = '{}/img/{}/{}.png'.format(dir, idx, sample_name)  # source image
            flow_map = '{}/bm/{}/{}.npy'.format(dir, idx, sample_name)   # flow_map
            # abd = '{}/alb/{}.png'.format(dir, sample_name)   # recon
            grid2d = '{}/grid2D/{}/{}.mat'.format(dir, idx, sample_name)
            abd = '{}/recon/{}/{}.png'.format(dir, idx, '{}chess48{}'.format(sample_name[:-4], sample_name[-4:]))
            dataset_list.append([img, grid2d, flow_map, abd])

    return dataset_list



def get_tps_motion_from_grid2d(grid2d, tps_resolution, img_resolution):
    # grid2d: [b,2,45,31], gt source mesh
    # TPS grid resolution (default same in x and y)
    # target mesh: rigid mesh
    # source mesh: predicted mesh
    # tps motion is (target mesh - source mesh)
    H, W = img_resolution
    B, _, grid_h, grid_w = grid2d.shape

    grid_indices_row = np.round(0 + (grid_h - 1) * np.linspace(0, 1, tps_resolution[0])).astype(np.int32)
    grid_indices_col = np.round(0 + (grid_w - 1) * np.linspace(0, 1, tps_resolution[1])).astype(np.int32)
    grid_rows = grid_indices_row[:, np.newaxis]
    grid_cols = grid_indices_col[np.newaxis, :]
    source_mesh = grid2d[:, :, grid_rows, grid_cols]

    rows = np.linspace(0, H - 1, num=grid_h, dtype=np.float32)
    cols = np.linspace(0, W - 1, num=grid_w, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(cols, rows, indexing='xy')
    base = np.stack([row_grid, col_grid], axis=0)
    base = torch.from_numpy(base[np.newaxis, ...]).repeat(B, 1, 1, 1).to(grid2d.device)

    target_mesh = base[:, :, grid_rows, grid_cols]

    tps_motion = source_mesh - target_mesh
    tps_motion[:, 0, :, :] = tps_motion[:, 0, :, :] / (W - 1)
    tps_motion[:, 1, :, :] = tps_motion[:, 1, :, :] / (H - 1)

    norm_target_mesh = target_mesh[0].unsqueeze(0).clone()
    norm_target_mesh = norm_target_mesh.permute(0,2,3,1)
    mesh_w = norm_target_mesh[..., 0] * 2. / W - 1
    mesh_h = norm_target_mesh[..., 1] * 2. / H - 1
    norm_target_mesh = torch.stack([mesh_w, mesh_h], dim=3)  # [b,16,16,2],[-1,1]
    return source_mesh.permute(0,2,3,1).to(dtype=torch.float32), norm_target_mesh.to(dtype=torch.float32), tps_motion.to(dtype=torch.float32)



def grid2flow(grid_motion, output_size, norm_target):
    if grid_motion.shape[0] != norm_target.shape[0]:
        target = norm_target.repeat(grid_motion.shape[0], 1, 1, 1).clone()
    else:
        target = norm_target.clone()
    # tps_motion: tps motion w normalization, [b,2,16,16]
    grid_motion_1 = grid_motion.clone()
    B, _, mesh_h, mesh_w = grid_motion_1.shape
    H, W = output_size
    grid_motion_1[:, 0, :, :] = grid_motion_1[:, 0, :, :] * (W - 1)
    grid_motion_1[:, 1, :, :] = grid_motion_1[:, 1, :, :] * (H - 1)

    target[..., 0] = (target[..., 0] + 1) / 2 * (W - 1)
    target[..., 1] = (target[..., 1] + 1) / 2 * (H - 1)
    target = target.permute(0, 3, 1, 2)
    source = target + grid_motion_1
    source = source.permute(0, 2, 3, 1)
    grid_flow = source.clone()
    grid_flow[:, :, :, 0] = grid_flow[:, :, :, 0] / (W - 1)
    grid_flow[:, :, :, 1] = grid_flow[:, :, :, 1] / (H - 1)
    grid_flow = (grid_flow * 2.0) - 1.0
    grid_flow = F.interpolate(
        grid_flow.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True
    )

    return grid_flow, source


def load_grid2d_mat_for_doc3d(grid2d_path):
    grid2d = h5py.File(grid2d_path)
    grid2d = grid2d['grid2D'][:].transpose((2, 1, 0))
    return grid2d

root = '/root/autodl-tmp/dataset/Doc3d'

dataset_list = make_doc3d_dataset_list(root)
B = 10
for idx in range(1,22):
    if not os.path.isdir('{}/target_img/{}'.format(root, idx)):
        os.mkdir('{}/target_img/{}'.format(root, idx))
        os.mkdir('{}/grid_bm/{}'.format(root, idx))

# print(len(dataset_list))
for i in range(0, len(dataset_list), B):
    img_list = []
    grid2d_list = []
    img_path_list = []
    for j in range(B):
        if (i+j) < len(dataset_list):
            inputs_paths, grid2d_path, flow_path, abd_path = dataset_list[i+j]
            img = cv2.imread(inputs_paths, 1)[:, :, ::-1].astype(np.uint8)
            grid2d = load_grid2d_mat_for_doc3d(grid2d_path)  # [h,w,(column,row)]
            img_list.append(img)
            grid2d_list.append(grid2d)
            img_path_list.append(inputs_paths)

    img_tensor = torch.from_numpy(np.array(img_list)).to(dtype=torch.float32).to(device)
    grid2d_tensor = torch.from_numpy(np.array(grid2d_list)).to(device)
    gt_mesh, norm_target_mesh, gt_grid_motion = get_tps_motion_from_grid2d(grid2d_tensor.permute(0, 3, 1, 2), [45, 31],
                                                                          [448, 448])
    grid_flow, source_mesh = grid2flow(gt_grid_motion, [448, 448], norm_target=norm_target_mesh)
    warped_img = F.grid_sample(img_tensor.permute(0, 3, 1, 2), grid_flow.permute(0, 2, 3, 1), mode='bilinear',
                               align_corners=True)
    # cv2.imwrite('warped_img.jpg', warped_img[5].permute(1,2,0).cpu().numpy())
    for j in range(B):
        if i + j < len(dataset_list):
            dir_idx = (img_path_list[j].split('/')[-2])
            img_name = (img_path_list[j].split('/')[-1]).split('.')[0]
            cv2.imwrite('{}/target_img/{}/{}.png'.format(root, dir_idx, img_name), warped_img[j].permute(1, 2, 0).cpu().numpy().take([2,1,0], axis=-1))
            np.save('{}/grid_bm/{}/{}.npy'.format(root, dir_idx, img_name), grid_flow[j].cpu().numpy())
    print('process {} images.'.format(i+B))
print('done')

