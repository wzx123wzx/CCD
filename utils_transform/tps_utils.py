import torch
import numpy as np
import torch.nn.functional as F
import cv2
import config.settings as settings


def tps2flow(tps_motion, output_size, norm_target, solve_inv=False, inter=False):
    if inter:
        # interpolate target points to accelerate flow calculation
        if tps_motion.shape[0] != norm_target.shape[0]:
            target = norm_target.repeat(tps_motion.shape[0], 1, 1, 1).clone()
        else:
            target = norm_target.clone()
        tps_motion_1 = tps_motion.clone()
        B, _, mesh_h, mesh_w = tps_motion_1.shape
        H, W = output_size
        tps_motion_1[:, 0, :, :] = tps_motion_1[:, 0, :, :] * (W - 1)
        tps_motion_1[:, 1, :, :] = tps_motion_1[:, 1, :, :] * (H - 1)

        target[..., 0] = (target[..., 0] + 1) / 2 * (W - 1)
        target[..., 1] = (target[..., 1] + 1) / 2 * (H - 1)
        target = target.permute(0, 3, 1, 2)
        source = target + tps_motion_1  # [b,2,h,w]
        norm_source = source.clone()
        norm_source[:, 0, :, :] = 2 * (norm_source[:, 0, :, :] / (W - 1)) - 1
        norm_source[:, 1, :, :] = 2 * (norm_source[:, 1, :, :] / (H - 1)) - 1
        norm_source = norm_source.permute(0, 3, 1, 2)
        tps_flow = F.interpolate(norm_source, (H, W), mode='bilinear')

        return tps_flow, None, None, source
    if tps_motion.shape[0] != norm_target.shape[0]:
        target = norm_target.repeat(tps_motion.shape[0], 1, 1, 1).clone()
    else:
        target = norm_target.clone()
    # tps_motion: tps motion w normalization, [b,2,16,16]
    tps_motion_1 = tps_motion.clone()
    B, _, mesh_h, mesh_w = tps_motion_1.shape
    H, W = output_size
    tps_motion_1[:, 0, :, :] = tps_motion_1[:, 0, :, :] * (W - 1)
    tps_motion_1[:, 1, :, :] = tps_motion_1[:, 1, :, :] * (H - 1)

    # indices_row = np.round(0 + (H - 1) * np.linspace(0, 1, mesh_h)).astype(np.int32)
    # indices_col = np.round(0 + (W - 1) * np.linspace(0, 1, mesh_w)).astype(np.int32)
    # rows = indices_row[:, np.newaxis]
    # cols = indices_col[np.newaxis, :]
    # base = coords_grid_tensor((H, W)).repeat(B, 1, 1, 1)
    # target = base[:, :, rows, cols]
    # source = target + tps_motion_1

    target[..., 0] = (target[..., 0] + 1) / 2 * (W - 1)
    target[..., 1] = (target[..., 1] + 1) / 2 * (H - 1)
    target = target.permute(0, 3, 1, 2)
    source = target + tps_motion_1
    target = target.permute(0, 2, 3, 1)
    source = source.permute(0, 2, 3, 1)

    def get_norm_mesh(mesh, height, width):
        B, _, _, _ = mesh.shape
        mesh_w = mesh[..., 0] * 2. / float(width) - 1.
        mesh_h = mesh[..., 1] * 2. / float(height) - 1.
        norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # bs*(grid_h+1)*(grid_w+1)*2

        return norm_mesh.reshape([B, -1, 2])  # bs*-1*2

    def _meshgrid(height, width, source):
        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0)).to(
            source.device)
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width])).to(
            source.device)

        x_t_flat = x_t.reshape([1, 1, -1])
        y_t_flat = y_t.reshape([1, 1, -1])

        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:, :, 0], 2).to(source.device)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:, :, 1], 2).to(source.device)  # [bn, pn, 1]

        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g).to(source.device)  # [bn, 1, h*w]

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)  # [bn, 3+pn, h*w]

        return grid

    def _transform(T, source, out_size):
        num_batch, _, _ = source.shape

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, source)  # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        x_s_flat = x_s.reshape([-1])
        y_s_flat = y_s.reshape([-1])

        # input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        # output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])
        # output = output.permute(0, 3, 1, 2)

        flow_x = x_s_flat.reshape([num_batch, -1])
        flow_y = y_s_flat.reshape([num_batch, -1])
        tps_flow = torch.stack([flow_x, flow_y], 1).reshape([num_batch, 2, out_height, out_width])

        return tps_flow

    def _solve_system(source, target):
        num_batch = source.size()[0]
        num_point = source.size()[1]

        np.set_printoptions(precision=8)

        ones = torch.ones(num_batch, num_point, 1).float().to(source.device)

        p = torch.cat([ones, source], 2)  # [bn, pn, 3]

        p_1 = p.reshape([num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1 - p_2), 3)  # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]

        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, pn]

        zeros = torch.zeros(num_batch, 3, 3).float().to(source.device)

        W_0 = torch.cat((p, r), 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0, 2, 1)), 2)  # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1)  # [bn, pn+3, pn+3]

        W_inv = torch.inverse(W.type(torch.float64))

        zeros2 = torch.zeros(num_batch, 3, 2).to(source.device)

        tp = torch.cat((target, zeros2), 1)  # [bn, pn+3, 2]

        T = torch.matmul(W_inv, tp.type(torch.float64))  # [bn, pn+3, 2]
        T = T.permute(0, 2, 1)  # [bn, 2, pn+3]

        return T.type(torch.float32)

    norm_target_mesh = get_norm_mesh(target, output_size[0], output_size[1])
    norm_source_mesh = get_norm_mesh(source, output_size[0], output_size[1])
    T = _solve_system(norm_target_mesh, norm_source_mesh)  # backwarp warping, so swap source and target here!!!
    tps_flow = _transform(T, norm_target_mesh, output_size)
    if solve_inv:
        T_inv = _solve_system(norm_source_mesh, norm_target_mesh)
    else:
        T_inv = None

    return tps_flow, T, T_inv, source


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


def cal_inv_tps(tps_motion, output_size, norm_target):
    if tps_motion.shape[0] != norm_target.shape[0]:
        target = norm_target.repeat(tps_motion.shape[0], 1, 1, 1).clone()
    else:
        target = norm_target.clone()
    # tps_motion: tps motion w normalization, [b,2,16,16]
    tps_motion_1 = tps_motion.clone()
    B, _, mesh_h, mesh_w = tps_motion_1.shape
    H, W = output_size
    tps_motion_1[:, 0, :, :] = tps_motion_1[:, 0, :, :] * (W - 1)
    tps_motion_1[:, 1, :, :] = tps_motion_1[:, 1, :, :] * (H - 1)

    target[..., 0] = (target[..., 0] + 1) / 2 * (W - 1)
    target[..., 1] = (target[..., 1] + 1) / 2 * (H - 1)
    target = target.permute(0, 3, 1, 2)
    source = target + tps_motion_1
    target = target.permute(0, 2, 3, 1)
    source = source.permute(0, 2, 3, 1)

    def get_norm_mesh(mesh, height, width):
        B, _, _, _ = mesh.shape
        mesh_w = mesh[..., 0] * 2. / float(width) - 1.
        mesh_h = mesh[..., 1] * 2. / float(height) - 1.
        norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # bs*(grid_h+1)*(grid_w+1)*2

        return norm_mesh.reshape([B, -1, 2])  # bs*-1*2

    def _solve_system(source, target):
        num_batch = source.size()[0]
        num_point = source.size()[1]

        np.set_printoptions(precision=8)

        ones = torch.ones(num_batch, num_point, 1).float().to(source.device)

        p = torch.cat([ones, source], 2)  # [bn, pn, 3]

        p_1 = p.reshape([num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1 - p_2), 3)  # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]

        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, pn]

        zeros = torch.zeros(num_batch, 3, 3).float().to(source.device)

        W_0 = torch.cat((p, r), 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0, 2, 1)), 2)  # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1)  # [bn, pn+3, pn+3]

        W_inv = torch.inverse(W.type(torch.float64))

        zeros2 = torch.zeros(num_batch, 3, 2).to(source.device)

        tp = torch.cat((target, zeros2), 1)  # [bn, pn+3, 2]

        T = torch.matmul(W_inv, tp.type(torch.float64))  # [bn, pn+3, 2]
        T = T.permute(0, 2, 1)  # [bn, 2, pn+3]

        return T.type(torch.float32)

    norm_target_mesh = get_norm_mesh(target, output_size[0], output_size[1])
    norm_source_mesh = get_norm_mesh(source, output_size[0], output_size[1])

    T_inv = _solve_system(norm_source_mesh, norm_target_mesh)

    return T_inv


def tps2_3dflow(source, target, source_depth, output_size):
    # source: P, [grid_h, grid_w, 3], [-1,1]
    # target: Q, [grid_h, grid_w, 3], [-1,1]
    # output_size: [h,w]
    # Q = [K, P] * [W, M]^T

    # Q: [n, 3]: target, each row: [x_i, y_i, z_i]

    # K: [n, n], O(p_i-p_j), from source, for tps3d, the radius function is r, for tps2d, the radius function is r^2*log(r^2)
    # W: [n, 3], nonlinear transformation parameters

    # P: [n, 4], source, each row: [1, x_i, y_i, z_i]
    # M: [4, 3], linear transformation parameters

    # set constrain: P^T * W = 0
    # M = (P^T * K^-1 * P)^-1 * P^T * K^-1 * Q
    # W = K^-1 * (Q - P * M)

    def _meshgrid(height, width, source, source_depth):
        num_batch = source.size()[0]

        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1, 1.0, width), 0)).to(
            source.device)
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1, 1.0, height), 1), torch.ones([1, width])).to(
            source.device)
        z_t = (2 * source_depth - 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z_t = z_t.repeat(1, 1, width, height)

        x_t_flat = x_t.unsqueeze(0).repeat(num_batch, 1, 1).reshape([num_batch, 1, -1])  # [b,1,h*w]
        y_t_flat = y_t.unsqueeze(0).repeat(num_batch, 1, 1).reshape([num_batch, 1, -1])  # [b,1,h*w]
        z_t_flat = z_t.reshape([num_batch, 1, -1])  # [b,1,h*w]

        px = torch.unsqueeze(source[:, :, 0], 2).to(source.device)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:, :, 1], 2).to(source.device)  # [bn, pn, 1]
        pz = torch.unsqueeze(source[:, :, 2], 2).to(source.device)

        d = torch.square(x_t_flat - px) + torch.square(y_t_flat - py) + torch.square(z_t_flat - pz)
        r = torch.sqrt(d)  # [bn, pn, h*w]

        # d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py) + torch.square(z_t_flat - pz)
        # r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, h*w]

        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        z_t_flat_g = z_t_flat.expand(num_batch, -1, -1)

        ones = torch.ones_like(x_t_flat_g).to(source.device)  # [bn, 1, h*w]

        grid = torch.cat((r, ones, x_t_flat_g, y_t_flat_g, z_t_flat_g), 1)  # [bn, 3+pn, h*w]

        return grid.permute(0, 2, 1)

    def _transform(W, M, source, source_depth, out_size):
        num_batch, _, _ = source.shape

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, source, source_depth)  # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(grid, torch.cat([W, M], dim=1))

        x_s = T_g[:, :, 0]
        y_s = T_g[:, :, 1]
        z_s = T_g[:, :, 2]

        x_s_flat = x_s.reshape([-1])
        y_s_flat = y_s.reshape([-1])
        z_s_flat = z_s.reshape([-1])

        flow_x = x_s_flat.reshape([num_batch, -1])
        flow_y = y_s_flat.reshape([num_batch, -1])
        flow_z = z_s_flat.reshape([num_batch, -1])

        tps_3dflow = torch.stack([flow_x, flow_y, flow_z], 1).reshape([num_batch, 3, out_height, out_width])

        return tps_3dflow

    def _solve_system(P, Q):
        b, n, _ = P.shape

        ones = torch.ones(b, n, 1).float().to(P.device)
        P = torch.cat([ones, P], dim=-1)  # [bn, pn, 4]

        p_1 = P.reshape([b, -1, 1, 4])  # [bn, pn, 1, 3]
        p_2 = P.reshape([b, 1, -1, 4])  # [bn, 1, pn, 3]

        d = torch.sqrt(torch.sum(torch.square(p_1 - p_2), 3))  # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]
        K = d  # [b,n,n]

        # d2 = torch.sum(torch.square(p_1 - p_2), 3)  # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]
        # K = d2 * torch.log(d2 + 1e-6)   # [b,n,n]

        K_inv = torch.inverse(K)
        P_T = P.permute(0, 2, 1)
        M = torch.inverse(P_T @ K_inv @ P) @ P_T @ K_inv @ Q
        W = K_inv @ (Q - P @ M)

        W_inv = W.permute(0, 2, 1)
        bending_energy = W_inv @ K @ W
        bending_energy = bending_energy.diagonal(dim1=1, dim2=2)
        # eigenvalues = torch.linalg.eigvalsh(K, UPLO='L')
        # bending_energy_x = W_inv[:, 0, :].unsqueeze(1) @ K @ W[:, :, 0].unsqueeze(-1)
        # bending_energy_y = W_inv[:, 1, :].unsqueeze(1) @ K @ W[:, :, 1].unsqueeze(-1)
        # bending_energy_z = W_inv[:, 2, :].unsqueeze(1) @ K @ W[:, :, 2].unsqueeze(-1)
        return W, M, bending_energy

    b, grid_h, grid_w, _ = source.shape
    n = grid_h * grid_w
    source = source.reshape([b, -1, 3])  # b*n*3, n=grid_h*grid_W
    target = target.reshape([b, -1, 3])
    W, M, bending_energy = _solve_system(source, target)  # backward warping, so swap source and target here
    tps_3dflow = _transform(W, M, source, source_depth, output_size)

    return tps_3dflow, bending_energy


def cal_mesh_3d(depth_motion, target_depth, source_mesh, norm_target_mesh):
    target_depth_mesh = target_depth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    target_depth_mesh = target_depth_mesh.repeat(1, 1, settings.model.tps_resolution,
                                                 settings.model.tps_resolution)  # [b,1,8,8]
    source_depth_mesh = depth_motion + target_depth_mesh
    source_depth_mesh = source_depth_mesh * 2 - 1  # [0,1] -> [-1,1]
    norm_source_mesh = source_mesh.clone()
    norm_source_mesh[..., 0] = (norm_source_mesh[..., 0] / (settings.train.image_size[1] - 1)) * 2 - 1
    norm_source_mesh[..., 1] = (norm_source_mesh[..., 1] / (settings.train.image_size[0] - 1)) * 2 - 1
    source_mesh_3d = torch.cat([norm_source_mesh, source_depth_mesh.permute(0, 2, 3, 1)], dim=-1)  # [b,8,8,3], [-1,1]
    target_mesh_3d = norm_target_mesh.repeat(source_mesh_3d.shape[0], 1, 1, 1).clone()
    target_mesh_3d = torch.cat([target_mesh_3d, (target_depth_mesh * 2 - 1).permute(0, 2, 3, 1)],
                               dim=-1)  # [b,8,8,3], [-1,1]

    return source_mesh_3d, target_mesh_3d


def draw_mesh_on_warp(warp, f_local):
    grid_h, grid_w, _ = f_local.shape
    warp = np.ascontiguousarray(255 * warp)

    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h):
        for j in range(grid_w):

            num = num + 1
            if j == (grid_w - 1) and i == (grid_h - 1):
                continue
            elif j == (grid_w - 1):
                cv2.line(warp, (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                         (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])), point_color, thickness, lineType)
            elif i == (grid_h - 1):
                cv2.line(warp, (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                         (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])), point_color, thickness, lineType)
            else:
                cv2.line(warp, (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                         (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                         (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])), point_color, thickness, lineType)

    return warp


def get_norm_mesh(mesh):
    # input: b,grid_h,grid_w,2
    # output:[-1,1]
    norm_mesh = mesh.clone()
    norm_mesh[:, :, :, 0] = (norm_mesh[:, :, :, 0] / (settings.train.image_size[1] - 1)) * 2 - 1
    norm_mesh[:, :, :, 1] = (norm_mesh[:, :, :, 1] / (settings.train.image_size[0] - 1)) * 2 - 1
    return norm_mesh

def upsample_tps(tps_motion, grid_h, grid_w, up_h, up_w, output_size, norm_target, up_norm_target):
    if grid_h == up_h and grid_w == up_w:
        return tps_motion
    if tps_motion.shape[0] != norm_target.shape[0]:
        target = norm_target.repeat(tps_motion.shape[0], 1, 1, 1).clone()
        up_norm_target_mesh = up_norm_target.repeat(tps_motion.shape[0], 1, 1, 1).clone().reshape([tps_motion.shape[0], -1, 2])
    else:
        target = norm_target.clone()
        up_norm_target_mesh = up_norm_target.clone().reshape([tps_motion.shape[0], -1, 2])
    # tps_motion: tps motion w normalization, [b,2,16,16]
    tps_motion_1 = tps_motion.clone()
    B, _, mesh_h, mesh_w = tps_motion_1.shape
    H, W = output_size
    tps_motion_1[:, 0, :, :] = tps_motion_1[:, 0, :, :] * (W - 1)
    tps_motion_1[:, 1, :, :] = tps_motion_1[:, 1, :, :] * (H - 1)

    target[..., 0] = (target[..., 0] + 1) / 2 * (W - 1)
    target[..., 1] = (target[..., 1] + 1) / 2 * (H - 1)
    target = target.permute(0, 3, 1, 2)
    source = target + tps_motion_1
    target = target.permute(0, 2, 3, 1)
    source = source.permute(0, 2, 3, 1)

    def get_norm_mesh(mesh, height, width):
        B, _, _, _ = mesh.shape
        mesh_w = mesh[..., 0] * 2. / float(width) - 1.
        mesh_h = mesh[..., 1] * 2. / float(height) - 1.
        norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # bs*(grid_h+1)*(grid_w+1)*2

        return norm_mesh.reshape([B, -1, 2])  # bs*-1*2

    def _meshgrid(up_source, source):
        x_t_flat = up_source[:, :, 0].unsqueeze(1)
        y_t_flat = up_source[:, :, 1].unsqueeze(1)
        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:, :, 0], 2).to(source.device)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:, :, 1], 2).to(source.device)  # [bn, pn, 1]

        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g).to(source.device)  # [bn, 1, h*w]

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)  # [bn, 3+pn, h*w]

        return grid

    def _transform(T, source, up_source, out_size):
        num_batch, _, _ = source.shape

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(up_source, source)  # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        # (-1,1), tps_motion
        # x_s,y_s tps warped grid, (-1,1), grid, (-1,1)
        tps_motion_x = (x_s - grid[:, 1, :]) / 2
        tps_motion_y = (y_s - grid[:, 2, :]) / 2

        tps_motion_new = torch.stack([tps_motion_x, tps_motion_y], 1)
        tps_motion_new = tps_motion_new.reshape([num_batch, 2, out_height, out_width])

        return tps_motion_new

    def _solve_system(source, target):
        num_batch = source.size()[0]
        num_point = source.size()[1]

        np.set_printoptions(precision=8)

        ones = torch.ones(num_batch, num_point, 1).float().to(source.device)

        p = torch.cat([ones, source], 2)  # [bn, pn, 3]

        p_1 = p.reshape([num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1 - p_2), 3)  # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]

        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, pn]

        zeros = torch.zeros(num_batch, 3, 3).float().to(source.device)

        W_0 = torch.cat((p, r), 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0, 2, 1)), 2)  # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1)  # [bn, pn+3, pn+3]

        W_inv = torch.inverse(W.type(torch.float64))

        zeros2 = torch.zeros(num_batch, 3, 2).to(source.device)

        tp = torch.cat((target, zeros2), 1)  # [bn, pn+3, 2]

        T = torch.matmul(W_inv, tp.type(torch.float64))  # [bn, pn+3, 2]
        T = T.permute(0, 2, 1)  # [bn, 2, pn+3]

        return T.type(torch.float32)

    norm_target_mesh = get_norm_mesh(target, output_size[0], output_size[1])
    norm_source_mesh = get_norm_mesh(source, output_size[0], output_size[1])
    T = _solve_system(norm_target_mesh, norm_source_mesh)  # backwarp warping, so swap source and target here!!!
    upsample_tps_motion = _transform(T, norm_target_mesh, up_norm_target_mesh, [up_h, up_w])

    return upsample_tps_motion
