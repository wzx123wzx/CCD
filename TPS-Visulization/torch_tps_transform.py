import torch
import numpy as np
import torch.nn.functional as F

from constant import device


# transforming an image
# from target (control points) to source (control points)
# all the points should be normalized from -1 ~1

def transformer(U, source, target, out_size):
    def _repeat(x, n_repeats):
        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1, 1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size):
        num_batch, num_channels, height, width = im.size()

        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]

        zero = 0
        max_y = height - 1
        max_x = width - 1

        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x).to(device)
        x1 = torch.clamp(x1, zero, max_x).to(device)
        y0 = torch.clamp(y0, zero, max_y).to(device)
        y1 = torch.clamp(y1, zero, max_y).to(device)
        dim2 = torch.from_numpy(np.array(width)).to(device)
        dim1 = torch.from_numpy(np.array(width * height)).to(device)

        base = _repeat(torch.arange(0, num_batch) * dim1.cpu(), (out_height * out_width)).to(device)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _meshgrid(height, width, source):
        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0)).to(device)
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width])).to(device)

        x_t_flat = x_t.reshape([1, 1, -1])
        y_t_flat = y_t.reshape([1, 1, -1])

        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:, :, 0], 2).to(device)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:, :, 1], 2).to(device)  # [bn, pn, 1]

        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g).to(device)  # [bn, 1, h*w]

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)  # [bn, 3+pn, h*w]

        return grid

    def _transform(T, source, input_dim, out_size):
        num_batch, num_channels, height, width = input_dim.size()

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, source)  # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        x_s_flat = x_s.reshape([-1])
        y_s_flat = y_s.reshape([-1])

        # 手写的插值算法，速度慢
        # input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        # output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])
        # output = output.permute(0, 3, 1, 2)

        # 使用pytorch的grid_sample函数实现插值，速度更快，另外还对mask进行了变换
        flow_x = x_s_flat.reshape([num_batch, -1])
        flow_y = y_s_flat.reshape([num_batch, -1])
        grid_flow = torch.stack([flow_x, flow_y], 1).reshape([num_batch, 2, out_height, out_width])
        img_transformed = F.grid_sample(input_dim, grid_flow.permute(0, 2, 3, 1), align_corners=True,
                                        padding_mode='zeros')
        # mask = torch.ones([num_batch, 1, height, width], device=device)
        # mask_transformed = F.grid_sample(mask, grid_flow.permute(0, 2, 3, 1), align_corners=True, padding_mode='zeros')
        # mask_transformed[mask_transformed < 0.99] = 0
        # mask_transformed[mask_transformed > 0] = 1
        # output = torch.cat([img_transformed, mask_transformed], dim=1)

        return img_transformed  # , condition

    def _solve_system(source, target):
        num_batch = source.size()[0]
        num_point = source.size()[1]

        np.set_printoptions(precision=8)

        ones = torch.ones(num_batch, num_point, 1).float().to(device)

        p = torch.cat([ones, source], 2)  # [bn, pn, 3]

        p_1 = p.reshape([num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1 - p_2), 3)  # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]

        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, pn]

        zeros = torch.zeros(num_batch, 3, 3).float().to(device)

        W_0 = torch.cat((p, r), 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0, 2, 1)), 2)  # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1)  # [bn, pn+3, pn+3]

        # W_inv = torch.inverse(W.type(torch.float64))
        W_inv = torch.inverse(W.type(torch.float64).cpu()).to(device)

        zeros2 = torch.zeros(num_batch, 3, 2).to(device)

        tp = torch.cat((target, zeros2), 1)  # [bn, pn+3, 2]

        T = torch.matmul(W_inv, tp.type(torch.float64))  # [bn, pn+3, 2]
        T = T.permute(0, 2, 1)  # [bn, 2, pn+3]

        return T.type(torch.float32)

    T = _solve_system(source, target)

    output = _transform(T, source, U, out_size)

    return output


def H2Mesh(H, rigid_mesh, grid_h, grid_w, device):
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2).to(device)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h + 1) * (grid_w + 1), 1).to(device)

    ori_pt = torch.cat((ori_pt, ones), 2)  # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H, ori_pt.permute(0, 2, 1))  # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:, 0, :] / tar_pt[:, 2, :], 2)
    mesh_y = torch.unsqueeze(tar_pt[:, 1, :] / tar_pt[:, 2, :], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h + 1, grid_w + 1, 2])

    return mesh


# rigid mesh
def get_rigid_mesh(batch_size, height, width, grid_h, grid_w, device):
    ww = torch.matmul(torch.ones([grid_h + 1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w + 1), 0)).to(
        device)
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h + 1), 1),
                      torch.ones([1, grid_w + 1])).to(device)

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)), 2)  # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt


def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[..., 0] * 2. / float(width) - 1.
    mesh_h = mesh[..., 1] * 2. / float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2])  # bs*-1*2
