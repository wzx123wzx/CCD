import torch

batch_size = 1
grid_h = 7
grid_w = 7
img_h = 384
img_w = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
