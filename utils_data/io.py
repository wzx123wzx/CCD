import numpy as np
import h5py
import cv2


import torch.nn.functional as F
import random
import config.settings as settings



def tight_crop_mix_doc3dgrid(img, mask, grid2d, grid3d, grid_bm):
    # img [512,512,3] unit8
    # mask [512,512] unit8

    size = mask.shape
    [y, x] = (mask[:, :, 0]).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    # Preserve the background by operating on the original image without cropping.
    # new_img = img.copy()
    # new_mask = mask.copy()

    # Add a random in-bounds offset without exceeding image boundaries.
    offset = 25
    cx1 = random.randint(5, offset)
    cx2 = random.randint(5, offset)
    cy1 = random.randint(5, offset)
    cy2 = random.randint(5, offset)

    # Adjust the crop bounds while preserving the full image background.
    final_minx = max(0, minx - cx1)
    final_maxx = min(size[1], maxx + cx2)
    final_miny = max(0, miny - cy1)
    final_maxy = min(size[0], maxy + cy2)

    # Crop the document region while retaining the background dimensions.
    cropped_img = img[final_miny:final_maxy, final_minx:final_maxx, :]
    cropped_mask = mask[final_miny:final_maxy, final_minx:final_maxx, :]

    # Update the backward-mapping coordinates.
    t = final_miny
    b = size[0] - final_maxy
    l = final_minx
    r = size[1] - final_maxx

    grid2d[:, :, 1] = grid2d[:, :, 1] - t
    grid2d[:, :, 0] = grid2d[:, :, 0] - l
    grid2d = 511 * grid2d / np.array([511.0 - l - r, 511.0 - t - b])

    grid_bm = (grid_bm + 1) / 2 * 512
    grid_bm[:, :, 1] = grid_bm[:, :, 1] - t
    grid_bm[:, :, 0] = grid_bm[:, :, 0] - l
    grid_bm = 511 * grid_bm / np.array([511.0 - l - r, 511.0 - t - b])  # 0~1
    grid_bm = (grid_bm / 512) * 2 - 1
    return cropped_img, cropped_mask / 255., grid2d, grid3d, grid_bm

def tight_crop_mix_uvdoc(img, mask, grid2d, grid3d, grid_bm):
    # img [512,512,3] unit8
    # mask [512,512] unit8

    size = mask.shape
    [y, x] = (mask[:, :, 0]).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    # Preserve the background by operating on the original image without cropping.
    # new_img = img.copy()
    # new_mask = mask.copy()

    # Add a random in-bounds offset without exceeding image boundaries.
    offset = 25
    cx1 = random.randint(5, offset)
    cx2 = random.randint(5, offset)
    cy1 = random.randint(5, offset)
    cy2 = random.randint(5, offset)

    # Adjust the crop bounds while preserving the full image background.
    final_minx = max(0, minx - cx1)
    final_maxx = min(size[1], maxx + cx2)
    final_miny = max(0, miny - cy1)
    final_maxy = min(size[0], maxy + cy2)

    # Crop the document region while retaining the background dimensions.
    cropped_img = img[final_miny:final_maxy, final_minx:final_maxx, :]
    cropped_mask = mask[final_miny:final_maxy, final_minx:final_maxx, :]

    # Update the backward-mapping coordinates.
    t = final_miny
    b = size[0] - final_maxy
    l = final_minx
    r = size[1] - final_maxx

    grid2d[:, :, 1] = grid2d[:, :, 1] - t
    grid2d[:, :, 0] = grid2d[:, :, 0] - l
    grid2d = 511 * grid2d / np.array([511.0 - l - r, 511.0 - t - b])

    grid_bm = (grid_bm + 1) / 2 * 512
    grid_bm[:, :, 1] = grid_bm[:, :, 1] - t
    grid_bm[:, :, 0] = grid_bm[:, :, 0] - l
    grid_bm = 511 * grid_bm / np.array([511.0 - l - r, 511.0 - t - b])  # 0~1
    grid_bm = (grid_bm / 512) * 2 - 1

    return cropped_img, cropped_mask / 255., grid2d, grid3d, grid_bm



def augmentation_mix_doc3dgrid(img, mask, grid2d, grid3d, grid_bm, bg=None):
    """
    Data augmentation function for document images and flow fields.

    Args:
        img: Source document image, shape [512, 512, 3], dtype uint8
        mask: Document region mask (255=document area, 0=background), shape [512, 512, 1], dtype uint8

    Returns:
        img: Augmented source image, shape [512, 512, 3], dtype uint8
        mask: Resized document mask, shape [512, 512, 1], dtype uint8
    """
    # tight crop
    img, mask, grid2d, grid3d, grid_bm = tight_crop_mix_doc3dgrid(img, mask, grid2d, grid3d, grid_bm)

    # replace bg
    if bg is not None:
        [fh, fw, _] = img.shape
        chance = random.random()
        # chance = 0.25
        if chance > 0.3:
            bg = cv2.resize(bg, (200, 200))
            bg = np.tile(bg, (3, 3, 1))  # (600, 600, 3)
            bg = bg[: fh, : fw, :]
            msk = mask
        elif chance < 0.3 and chance > 0.2:
            c = np.array([random.random(), random.random(), random.random()])
            bg = np.ones((fh, fw, 3)) * c
            msk = mask
            # cv2.imwrite("vis_hp/debug_vis/tex2.png", bg)
        else:
            bg = np.zeros((fh, fw, 3))
            msk = np.ones((fh, fw, 3))
        img = bg * (1 - msk) + img * msk

    mask = cv2.resize(mask, (512, 512))
    img = cv2.resize(img, (512, 512))
    return img, mask, grid2d, grid3d, grid_bm

def augmentation_mix_uvdoc(img, mask, grid2d, grid3d, grid_bm, bg=None):
    """
    Data augmentation function for document images and flow fields.

    Args:
        img: Source document image, shape [512, 512, 3], dtype uint8
        mask: Document region mask (255=document area, 0=background), shape [512, 512, 1], dtype uint8

    Returns:
        img: Augmented source image, shape [512, 512, 3], dtype uint8
        mask: Resized document mask, shape [512, 512, 1], dtype uint8
    """
    # tight crop
    img, mask, grid2d, grid3d, grid_bm = tight_crop_mix_uvdoc(img, mask, grid2d, grid3d, grid_bm)

    # replace bg
    if bg is not None:
        [fh, fw, _] = img.shape
        chance = random.random()
        # chance = 0.25
        if chance > 0.3:
            bg = cv2.resize(bg, (200, 200))
            bg = np.tile(bg, (3, 3, 1))  # (600, 600, 3)
            bg = bg[: fh, : fw, :]
            msk = mask
        elif chance < 0.3 and chance > 0.2:
            c = np.array([random.random(), random.random(), random.random()])
            bg = np.ones((fh, fw, 3)) * c
            msk = mask
            # cv2.imwrite("vis_hp/debug_vis/tex2.png", bg)
        else:
            bg = np.zeros((fh, fw, 3))
            msk = np.ones((fh, fw, 3))
        img = bg * (1 - msk) + img * msk

    mask = cv2.resize(mask, (512, 512))
    img = cv2.resize(img, (512, 512))
    return img, mask, grid2d, grid3d, grid_bm


def load_mask_mat(mask_path):
    mask = h5py.File(mask_path)
    mask = 255 * mask['seg'][:].transpose((1, 0))
    # mask = cv2.resize(mask, (512,512))

    return mask


def load_grid2d_mat(grid2d_path):
    grid2d = h5py.File(grid2d_path)
    grid2d = grid2d['grid2d'][:].transpose((2, 1, 0))
    return grid2d

def load_grid2d_mat_for_doc3d(grid2d_path):
    grid2d = h5py.File(grid2d_path)
    grid2d = grid2d['grid2D'][:].transpose((2, 1, 0))
    return grid2d


def load_grid3d_mat(grid3d_path):
    grid3d = h5py.File(grid3d_path)
    grid3d = grid3d['grid3d'][:].transpose((2, 1, 0))
    xmx, xmn, ymx, ymn, zmx, zmn = settings.train.uvdoc_grid3d_normalization
    grid3d[:, :, 0] = (grid3d[:, :, 0] - xmn) / (xmx - xmn)
    grid3d[:, :, 1] = (grid3d[:, :, 1] - ymn) / (ymx - ymn)
    grid3d[:, :, 2] = (grid3d[:, :, 2] - zmn) / (zmx - zmn)
    grid3d[:, :, 2] = 1 - grid3d[:, :, 2]
    # grid3d = 2 * grid3d - 1
    return grid3d   # [depth, column, row], [0,1]

def load_grid3d_mat_for_doc3d(grid3d_path):
    grid3d = h5py.File(grid3d_path)
    grid3d = grid3d['grid3D'][:].transpose((2, 1, 0))
    xmx, xmn, ymx, ymn, zmx, zmn = settings.train.doc3d_grid3d_normalization
    grid3d[:, :, 0] = (grid3d[:, :, 0] - zmn) / (zmx - zmn)
    grid3d[:, :, 1] = (grid3d[:, :, 1] - ymn) / (ymx - ymn)
    grid3d[:, :, 2] = (grid3d[:, :, 2] - xmn) / (xmx - xmn)
    grid3d[:, :, 2] = 1 - grid3d[:, :, 2]
    # grid3d = 2 * grid3d - 1
    return grid3d   # [depth, column, row], [0,1]

def load_grid_bm_mat_for_doc3d(grid_bm_path):
    grid_bm = np.load(grid_bm_path)
    bm0 = cv2.resize(grid_bm[0, :, :], (512, 512))
    bm1 = cv2.resize(grid_bm[1, :, :], (512, 512))
    grid_bm = np.stack([bm0, bm1], axis=-1)
    return grid_bm   # backward warping flow [0,1], [2,512,512]

def load_grid_bm_mat_for_uvdoc(grid_bm_path):
    grid_bm = np.load(grid_bm_path)
    bm0 = cv2.resize(grid_bm[0, :, :], (512, 512))
    bm1 = cv2.resize(grid_bm[1, :, :], (512, 512))
    grid_bm = np.stack([bm0, bm1], axis=-1)
    return grid_bm   # backward warping flow [0,1], [2,512,512]


