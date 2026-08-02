from datasets.listdataset import Aug_Mix_Doc3dGrid_ListDataset, Aug_Mix_UVDoc_ListDataset
    
from datasets.util import split2list
import json
import os

def make_mix_doc3dgrid_dataset_list(dir, split=None):
    # edit for my dataset folder structure
    sub_dataset_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                        '18', '19', '20', '21']
    dataset_list = []
    # for idx in range(1, 22):
    for idx in range(1, 22):
        for sample_name in os.listdir('{}/img/{}'.format(dir, idx)):
            sample_name = sample_name[:-4]
            img = '{}/img/{}/{}.png'.format(dir, idx, sample_name)  # source image
            target = '{}/target_img/{}/{}.png'.format(dir, idx, sample_name)  # target image
            recon = '{}/recon/{}/{}.png'.format(dir, idx, '{}chess48{}'.format(sample_name[:-4], sample_name[-4:]))
            grid2d = '{}/grid2D/{}/{}.mat'.format(dir, idx, sample_name)
            grid3d = '{}/grid3D/{}/{}.mat'.format(dir, idx, sample_name)
            grid_bm = '{}/grid_bm/{}/{}.npy'.format(dir, idx, sample_name)
            dataset_list.append([img, target, recon, grid2d, grid3d, grid_bm])

    return split2list(dataset_list, split, default_split=0.97)


def make_mix_uvdoc_dataset_list(dir, split=None):
    # edit for my dataset folder structure
    dataset_list = []
    for sample in os.listdir('{}/metadata_sample'.format(dir)):
        with open('{}/metadata_sample/{}'.format(dir, sample), 'r', encoding='utf-8') as file:
            data = json.load(file)
            img_path = '{}/img/{}.png'.format(dir, data['sample_id'])
            target_path = '{}/target_img/{}.png'.format(dir, data['sample_id'])
            grid2d_path = '{}/grid2d/{}.mat'.format(dir, data['geom_name'])
            grid3d_path = '{}/grid3d/{}.mat'.format(dir, data['geom_name'])
            mask_path = '{}/seg/{}.mat'.format(dir, data['geom_name'])
            grid_bm_path = '{}/grid_bm/{}.npy'.format(dir, data['sample_id'])
            dataset_list.append(
                [img_path, target_path, mask_path, grid2d_path, grid3d_path, grid_bm_path])
    return split2list(dataset_list, split, default_split=0.97)


def Doc3dGrid_Dataset(settings, root, source_image_transform=None, target_image_transform=None,
                                 flow_transform=None,
                                 co_transform=None, split=None, get_mapping=False, compute_mask_zero_borders=False,
                                 add_discontinuity=False, min_nbr_perturbations=5, max_nbr_perturbations=6,
                                 parameters_v2=None):
    # get original dataset path
    # train_list, test_list = make_doc3d_dataset_list(root, split) # get list [[sample1],[sample2],...]
    train_list, test_list = make_mix_doc3dgrid_dataset_list(root, split)
    train_dataset = Aug_Mix_Doc3dGrid_ListDataset(root, train_list,
                                                         source_image_transform=source_image_transform,
                                                         target_image_transform=target_image_transform,
                                                         flow_transform=flow_transform, co_transform=co_transform,
                                                         get_mapping=get_mapping,
                                                         compute_mask_zero_borders=compute_mask_zero_borders,
                                                         exclude_background=settings.model.exclude_background,
                                                         use_img_loss=settings.train.use_img_l1_loss or settings.train.use_img_l2_loss)
    test_dataset = Aug_Mix_Doc3dGrid_ListDataset(root, test_list, source_image_transform=source_image_transform,
                                                        target_image_transform=target_image_transform,
                                                        flow_transform=flow_transform, co_transform=co_transform,
                                                        get_mapping=get_mapping,
                                                        compute_mask_zero_borders=compute_mask_zero_borders,
                                                        exclude_background=settings.model.exclude_background,
                                                        use_img_loss=settings.train.use_img_l1_loss or settings.train.use_img_l2_loss)

    return train_dataset, test_dataset


def UVDoc_Dataset(settings, root, source_image_transform=None, target_image_transform=None,
                             flow_transform=None,
                             co_transform=None, split=None, get_mapping=False, compute_mask_zero_borders=False,
                             add_discontinuity=False, min_nbr_perturbations=5, max_nbr_perturbations=6,
                             parameters_v2=None):
    # get original dataset path
    train_list, test_list = make_mix_uvdoc_dataset_list(root, split)
    train_dataset = Aug_Mix_UVDoc_ListDataset(root, train_list, source_image_transform=source_image_transform,
                                                     target_image_transform=target_image_transform,
                                                     flow_transform=flow_transform, co_transform=co_transform,
                                                     get_mapping=get_mapping,
                                                     compute_mask_zero_borders=compute_mask_zero_borders,
                                                     use_img_loss=settings.train.use_img_l1_loss or settings.train.use_img_l2_loss)
    test_dataset = Aug_Mix_UVDoc_ListDataset(root, test_list, source_image_transform=source_image_transform,
                                                    target_image_transform=target_image_transform,
                                                    flow_transform=flow_transform, co_transform=co_transform,
                                                    get_mapping=get_mapping,
                                                    compute_mask_zero_borders=compute_mask_zero_borders,
                                                    use_img_loss=settings.train.use_img_l1_loss or settings.train.use_img_l2_loss)

    return train_dataset, test_dataset

