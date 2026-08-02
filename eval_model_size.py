import torch
# from .train_settings.ccd.improved_diffusion import dist_util
from train_settings.ccd.improved_diffusion.script_util import args_to_dict, create_model_and_diffusion
import config.settings

def print_model_breakdown(model):
    print(f"\n{'Module Name':<25} | {'Parameters':<15} | {'Size (MB)':<10}")
    print("-" * 55)

    total_params = 0
    total_size = 0

    # 1. Iterate over top-level submodules
    for name, module in model.named_children():
        # Count total parameters in this submodule
        num_params = sum(p.numel() for p in module.parameters())

        # Compute parameter memory in bytes (accounts for float32/float16 precision)
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())

        # Compute buffer memory in bytes (e.g. BatchNorm running_mean)
        buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())

        size_mb = (param_size + buffer_size) / 1024 ** 2

        print(f"{name:<25} | {num_params:<15} | {size_mb:<10.2f}")

    # 2. Aggregate the full model (including top-level parameters not under named_children)
    all_param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    all_buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (all_param_size + all_buffer_size) / 1024 ** 2
    total_param_count = sum(p.numel() for p in model.parameters())

    print("-" * 55)
    print(f"{'Total Model':<25} | {total_param_count:<15} | {total_size_mb:<10.2f}")
    print("-" * 55 + "\n")

# --- Usage ---
# Instantiate model first, then call:
# print_model_breakdown(model)

# state_dict = dist_util.load_state_dict('checkpoints/model019000.pt', map_location='cpu')
model, diffusion = create_model_and_diffusion(settings=config.settings, device=config.settings.train.device)
print_model_breakdown(model)
print('done')