import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
from torch.utils.data import Dataset

import glob
import cv2

from datasets.utils.general_utils import pil_loader, prepare_image, prepare_bm_docregis
import config.settings as settings




class Doc_benchmark(Dataset):
    def __init__(
            self, data_root, input_transform) -> None:
        self.data_root = data_root
        self.input_transform = input_transform
        self.init_img_parms()

    def init_img_parms(self):
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.data_paths = sorted([
            f for f in os.listdir(self.data_root)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])

    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        return im

    def __len__(self):
        return len(self.data_paths)

    def cv2_image_load(self, sample_path, new_size=(512, 512)):
        img_ori = cv2.imread(sample_path)[:, :, ::-1].astype(np.uint8)
        img = cv2.resize(img_ori, new_size)
        return img, img_ori

    def find_jpg_in_folder(self, folder_path):
        jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        return jpg_files

    def __getitem__(self, idx):  # 0,1,2,3(random sampled idx)

        sample_path = os.path.join(self.data_root, self.data_paths[idx])
        img, img_ori = self.cv2_image_load(sample_path, new_size=(settings.train.image_size[0], settings.train.image_size[1]))
        img = self.input_transform(img) / 255.
        img_ori = self.input_transform(img_ori) / 255.

        data_dict = {
            "doc_image": img,
            "doc_image_ori": img_ori,
            "doc_path": sample_path,
            "doc_name":self.data_paths[idx].split('.')[0]
        }

        return data_dict


class Doc_dewarping_Data1(Dataset):
    def __init__(self, root_path, transforms=None, resolution=288, model_setting="dewarpnet"):
        self.work_path = root_path  # './triple_data'
        if transforms is not None:
            self.transforms = transforms
        self.shuffle = True
        self.label_paths = []
        self.init_img_parms()
        self.loader = pil_loader
        self.resolution = resolution
        self.dataset_type = "docaligner"
        self.model_setting = model_setting

    def init_img_parms(self):
        self.labels = os.listdir(self.work_path)  # ['00001', '00002',..., '00006']
        for label in self.labels:
            label_path = os.path.join(self.work_path, label)  # './inv3d/data/train/00001'
            self.label_paths.append(label_path)  # ['./inv3d/data/train/00001',...]

    def __getitem__(self, index):
        sample_path = self.label_paths[index]
        if self.model_setting == "doctr":
            img, t, b, l, r, H, W = prepare_image(
                os.path.join(sample_path, 'warped_document.png'), os.path.join(sample_path, "warped_UV.npz"),
                # os.path.join(sample_path, 'warped_BM.npy'),
                color_jitter=True,
                # spatial_trans = self.spatial_trans, 
                resolution=self.resolution,
            )  # (3, 288, 288),0-1
            bm = prepare_bm_docregis(os.path.join(sample_path, 'warped_BM.npz'),
                                     self.resolution, t, b, l, r, H, W)  # (2, 288, 288), unnormalized [0, 288], x before y, row-major order
            # bm = (bm/288.0-0.5)*2
        return img, bm  # (3, 288, 288)，(2, 288, 288)

    def __len__(self):
        return len(self.label_paths)
