"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""
import numpy as np
import torch
import config.settings as settings


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs



# axis alignment loss
def axis_alignment_loss(mesh, gt_mesh, T_inv, size):
    # mesh: mesh, [b, grid_h, grid_w, 2]
    # gt_mesh: mesh, [b, grid_h, grid_w, 2]
    # T_inv: [b, 2, num_mesh_grids+3], tps transformation parameters
    def get_norm_mesh(mesh, height, width):
        B, _, _, _ = mesh.shape
        mesh_w = mesh[..., 0] * 2. / float(width) - 1.
        mesh_h = mesh[..., 1] * 2. / float(height) - 1.
        norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # bs*(grid_h+1)*(grid_w+1)*2

        return norm_mesh.reshape([B, -1, 2])  # bs*-1*2

    def _meshgrid(mesh, gt_mesh):
        # new_source: [b, pn_new, 2]
        # source: [b, pn, 2]
        x_t = mesh[:, :, 0]  # [b, pn_new]
        y_t = mesh[:, :, 1]  # [b, pn_new]

        x_t_flat = x_t.unsqueeze(1)  # [b,1,pn_new]
        y_t_flat = y_t.unsqueeze(1)  # [b,1,pn_new]

        px = torch.unsqueeze(gt_mesh[:, :, 0], 2).to(gt_mesh.device)  # [bn, pn, 1]
        py = torch.unsqueeze(gt_mesh[:, :, 1], 2).to(gt_mesh.device)  # [bn, pn, 1]

        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)  # [b, pn, pn_new]
        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, pn_new]
        ones = torch.ones_like(x_t_flat).to(gt_mesh.device)  # [bn, 1, pn_new]

        grid = torch.cat((ones, x_t_flat, y_t_flat, r), 1)  # [bn, 3+pn, pn_new]

        return grid

    b, grid_h, grid_w, _ = gt_mesh.shape
    norm_mesh = get_norm_mesh(mesh, size[0], size[1])
    norm_gt_mesh = get_norm_mesh(gt_mesh, size[0], size[1])
    meshgrid = _meshgrid(norm_gt_mesh, norm_mesh)
    T_meshgrid = torch.matmul(T_inv, meshgrid)
    T_meshgrid = T_meshgrid.reshape(b, 2, grid_h, grid_w)  # [b,2,grid_h,grid_w]
    # T_meshgrid_x = (T_meshgrid[:, 0, :, :] + 1) / 2 * settings.train.image_size[1]   # [b, grid_h, grid_w]
    # T_meshgrid_y = (T_meshgrid[:, 1, :, :] + 1) / 2 * settings.train.image_size[0]   # [b, grid_h, grid_w]
    T_meshgrid_x = T_meshgrid[:, 0, :, :]  # [b, grid_h, grid_w]
    T_meshgrid_y = T_meshgrid[:, 1, :, :]
    var_x = torch.var(T_meshgrid_x, dim=1)  # column, [b, grid_w]
    var_y = torch.var(T_meshgrid_y, dim=2)  # row, [b, grid_w]
    T_meshgrid_x = (T_meshgrid_x + 1) / 2 * (settings.train.image_size[1] - 1)
    T_meshgrid_y = (T_meshgrid_y + 1) / 2 * (settings.train.image_size[0] - 1)
    return torch.mean(var_x) + torch.mean(var_y), torch.cat([T_meshgrid_x.unsqueeze(1), T_meshgrid_y.unsqueeze(1)], 1)

