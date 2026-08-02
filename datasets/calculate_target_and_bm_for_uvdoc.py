import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import cv2
import h5py

# device = torch.device("cuda:1")
device = torch.device("cuda:0")
# device = torch.device("cpu")

def make_uvdoc_dataset_list(dir):
    # edit for my dataset folder structure
    dataset_list = []
    for sample in os.listdir('{}/metadata_sample'.format(dir)):
        with open('{}/metadata_sample/{}'.format(dir, sample), 'r', encoding='utf-8') as file:
            data = json.load(file)

            texture_path = '{}/textures_ori/{}'.format(dir, data['texture_name'])
            background_path = '{}/background/{}'.format(dir, data['background_name'])

            geom_name = '{}/metadata_geom/{}'.format(dir, data['geom_name'])
            img_path = '{}/img/{}.png'.format(dir, data['sample_id'])
            warped_texture_path = '{}/warped_textures/{}.png'.format(dir, data['sample_id'])
            grid2d_path = '{}/grid2d/{}.mat'.format(dir, data['geom_name'])
            grid3d_path = '{}/grid3d/{}.mat'.format(dir, data['geom_name'])
            mask_path = '{}/seg/{}.mat'.format(dir, data['geom_name'])
            uv_path = '{}/uvmap/{}.mat'.format(dir, data['geom_name'])
            wc_path = '{}/wc/{}.exr'.format(dir, data['geom_name'])

            dataset_list.append(
                [img_path, grid2d_path, grid3d_path, mask_path])

    return dataset_list


def get_norm_target_mesh(tps_resolution, img_resolution, dataset_name='doc3d', ):
    H, W = img_resolution
    grid_h = 512
    grid_w = 512
    if dataset_name == 'doc3d':
        grid_h = 512
        grid_w = 512
    elif dataset_name == 'uvdoc':
        grid_h = 89
        grid_w = 61

    B = 1
    grid_indices_row = np.round(0 + (grid_h - 1) * np.linspace(0, 1, tps_resolution[0])).astype(np.int32)
    grid_indices_col = np.round(0 + (grid_w - 1) * np.linspace(0, 1, tps_resolution[1])).astype(np.int32)
    grid_rows = grid_indices_row[:, np.newaxis]
    grid_cols = grid_indices_col[np.newaxis, :]

    rows = np.linspace(0, H - 1, num=grid_h, dtype=np.float32)
    cols = np.linspace(0, W - 1, num=grid_w, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(cols, rows, indexing='xy')
    base = np.stack([row_grid, col_grid], axis=0)
    base = torch.from_numpy(base[np.newaxis, ...]).repeat(B, 1, 1, 1)

    target_mesh = base[:, :, grid_rows, grid_cols]

    norm_target_mesh = target_mesh.clone()
    norm_target_mesh = norm_target_mesh.permute(0, 2, 3, 1)
    mesh_w = norm_target_mesh[..., 0] * 2. / W - 1
    mesh_h = norm_target_mesh[..., 1] * 2. / H - 1
    norm_target_mesh = torch.stack([mesh_w, mesh_h], dim=3)  # [b,16,16,2],[-1,1]
    return norm_target_mesh.to(dtype=torch.float32).to(device)


def load_grid2d_mat(grid2d_path):
    grid2d = h5py.File(grid2d_path)
    grid2d = grid2d['grid2d'][:].transpose((2, 1, 0))
    grid2d[:, :, 0] = grid2d[:, :, 0]
    grid2d[:, :, 1] = grid2d[:, :, 1]
    return grid2d


def get_tps_motion_from_grid2d(grid2d, tps_resolution, img_resolution):
    # grid2d: [b,2,89,61], gt source mesh
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

    # feat_rows = np.linspace(0, settings.model.update_feat_size[0] - 1, num=grid_h, dtype=np.float32)
    # feat_cols = np.linspace(0, settings.model.update_feat_size[1] - 1, num=grid_w, dtype=np.float32)
    # feat_row_grid, feat_col_grid = np.meshgrid(feat_cols, feat_rows, indexing='xy')
    # feat_base = np.stack([feat_row_grid, feat_col_grid], axis=0)
    # feat_base = torch.from_numpy(feat_base[np.newaxis, ...]).repeat(B, 1, 1, 1).to(grid2d.device)

    target_mesh = base[:, :, grid_rows, grid_cols]
    # feat_target_mesh = feat_base[:, :, grid_rows, grid_cols]

    tps_motion = source_mesh - target_mesh
    tps_motion[:, 0, :, :] = tps_motion[:, 0, :, :] / (W - 1)
    tps_motion[:, 1, :, :] = tps_motion[:, 1, :, :] / (H - 1)

    norm_target_mesh = target_mesh[0].unsqueeze(0).clone()
    norm_target_mesh = norm_target_mesh.permute(0, 2, 3, 1)
    mesh_w = norm_target_mesh[..., 0] * 2. / W - 1
    mesh_h = norm_target_mesh[..., 1] * 2. / H - 1
    norm_target_mesh = torch.stack([mesh_w, mesh_h], dim=3)  # [b,16,16,2],[-1,1]
    return source_mesh.permute(0, 2, 3, 1).to(dtype=torch.float32), norm_target_mesh.to(
        dtype=torch.float32), tps_motion.to(dtype=torch.float32)


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

root = '/root/autodl-tmp/dataset/UVDoc'
dataset_list = make_uvdoc_dataset_list(root)
# B = 50
B = 10

for i in range(0, len(dataset_list), B):
    img_list = []
    grid2d_list = []
    img_path_list = []
    for j in range(B):
        img_path, grid2d_path, grid3d_path, mask_path = dataset_list[i + j]
        img = cv2.imread(img_path, 1)[:, :, ::-1].astype(np.uint8)
        grid2d = load_grid2d_mat(grid2d_path)  # [h,w,(column,row)]
        img_list.append(img)
        grid2d_list.append(grid2d)
        img_path_list.append(img_path)
        # print(i+j)

    img_tensor = torch.from_numpy(np.array(img_list)).to(dtype=torch.float32).to(device)
    grid2d_tensor = torch.from_numpy(np.array(grid2d_list)).to(device)
    gt_mesh, norm_target_mesh, gt_grid_motion = get_tps_motion_from_grid2d(grid2d_tensor.permute(0, 3, 1, 2), [89, 61],
                                                                          [712, 488])
    grid_flow, source_mesh = grid2flow(gt_grid_motion, [712, 488], norm_target=norm_target_mesh)
    warped_img = F.grid_sample(img_tensor.permute(0, 3, 1, 2), grid_flow.permute(0, 2, 3, 1), mode='bilinear',
                               align_corners=True)
    # cv2.imwrite('warped_img.jpg', warped_img[3].permute(1,2,0).cpu().numpy())
    for j in range(B):
        img_name = (img_path_list[j].split('/')[-1]).split('.')[0]
        np.save('{}/grid_bm/{}.npy'.format(root, img_name), grid_flow[j].cpu().numpy())
        cv2.imwrite('{}/target_img/{}.png'.format(root, img_name),
                    warped_img[j].permute(1, 2, 0).cpu().numpy().take([2, 1, 0], axis=-1))
    print('process {} images.'.format(i + B))
print('done')

# find -name "*.png" | wc -l
