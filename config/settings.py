import torch


class train:
    image_size = [512, 512]
    device = torch.device("cuda:0")
    dataset_name = 'mix'
    if dataset_name == 'mix':
        train_doc3d_dataset_path = '/root/dataset/Doc3d'
        train_uvdoc_dataset_path = '/root/dataset/UVDoc'
        bg_aug_doc3d_dataset = True
        uvdoc_grid3d_normalization = (0.11433014, -0.12551452, 0.12401487, -0.12401487, 0.1952378, -0.1952378)
        doc3d_grid3d_normalization = (1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497)
    elif dataset_name == 'uvdoc':
        train_uvdoc_dataset_path = '/root/dataset/UVDoc'
        uvdoc_grid3d_normalization = (0.11433014, -0.12551452, 0.12401487, -0.12401487, 0.1952378, -0.1952378)
    elif dataset_name == 'doc3d_grid':
        bg_aug_doc3d_dataset = True
        train_doc3d_dataset_path = '/root/dataset/Doc3d'
        doc3d_grid3d_normalization = (1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497)

    train_mode = 'Cascade'
    use_gt_mask = False
    use_line_mask = True
    use_init_flow = False

    lr = 1e-4
    batch_size = 24
    n_threads = 12
    weight_decay = 0.0
    lr_anneal_steps = 0
    schedule_sampler = 'uniform'  # 'uniform' 'multi' 'fixed'
    microbatch = -1
    ema_rate = 0.9999
    use_fp16 = False
    fp16_scale_growth = 0.001
    visualize = True

    # Loss flags
    use_mesh_motion_l2_loss = True
    use_axis_loss = True
    use_img_l1_loss = False
    use_img_l2_loss = False

    # Loss weights
    w_mesh_motion_l2 = [600, 1200, 1800]


class eval:
    device = torch.device("cuda:0")
    dataset_name = "docunet"
    if dataset_name == 'docunet':
        eval_dataset_path = '/root/dataset/DocUNet/crop'
    elif dataset_name == 'uvdoc':
        eval_dataset_path = '/root/dataset/UVDoc/benchmark/img'
    elif dataset_name == 'dir300':
        eval_dataset_path = '/root/dataset/DIR300/dist'
    experiment_name = "test"
    save_tps_control_points = False  # Export first-stage TPS control points for interactive visualization.


class model:
    grid_resolution = [45, 31]
    tps_resolution = [8, 12]
    embedding_cond_size = 64
    update_feat_size = [64, 64]
    radius = 4
    learn_sigma = False
    sigma_small = False
    diffusion_steps = 3
    noise_schedule = 'cosine'
    use_kl = False
    predict_xstart = True
    rescale_timesteps = True
    rescale_learned_sigmas = True
    clip_denoised = False
    n_batch = 1
    interpolation_mode = 'bilinear'
    in_channels = 2
    exclude_background = True
    separate_cross_attn = 'seq'  # seq, para, one, concat
    use_decoder = False


class log:
    log_interval = 50
    save_interval = 2000
    tb_interval = 50
    resume_step = 0


class resume:
    resume_checkpoint = None
    seg_model_path = 'checkpoints/seg_model_512.pth'
    line_seg_model_path = 'checkpoints/line_model2.pth'
    new_seg_model_path = 'checkpoints/seg_model_512.pth'
    model_path = "checkpoints/best_model.pt"
    timestep_respacing = ''
