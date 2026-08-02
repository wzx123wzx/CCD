# Cascaded Coordinate Diffusion: Extending the Generative Paradigm for Document Dewarping

Document image geometric correction via a cascade diffusion-based geometric transformation field predictor.

## Overview

![Framework Overview](overview.png)

CCD predicts document dewarping transformations using a diffusion model with a DiT (Diffusion Transformer) backbone. The model generates a cascade of geometric transformation fields at three resolutions (two TPS control-point grids and one dense backward-mapping grid), conditioned on the input document image, a foreground segmentation mask, and a text-line segmentation mask.

## Environment

```bash
conda create -n CCD python=3.10.19
conda activate CCD

pip install torch==2.8.0+cu128 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

OCR metrics additionally require Tesseract OCR, `pytesseract`, `python-Levenshtein`, and `jiwer`. Visual benchmark metrics additionally require MATLAB and the MATLAB Engine for Python.

---

## Pretrained Weights

Download the pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1TtBAgxAxq4ftiTOIEkm9EHhzBz8eqghg?usp=sharing), then place all downloaded weight files in the `checkpoints/` directory.

---

## Data Preparation

### Doc3D

```
dataset/Doc3d/
├── img/          # warped document images  (sub-folders 1-21)
├── target_img/   # flat GT images
├── grid2D/       # 2D control-point meshes (.mat)
├── grid3D/       # 3D control-point meshes (.mat)
├── grid_bm/      # normalised backward mappings (.npy)
└── recon/        # checkerboard reconstruction images
```

### UVDoc

```
dataset/UVDoc/
├── img/               # warped images (.png)
├── target_img/        # flat GT images
├── grid2d/            # 2D meshes (.mat)
├── grid3d/            # 3D meshes (.mat)
├── grid_bm/           # backward mappings (.npy)
├── seg/               # foreground masks (.mat)
├── metadata_sample/   # per-sample JSON metadata
└── benchmark/         # UVDoc benchmark split
    └── img/           # 50 benchmark images
```

### DocUNet Benchmark

```
dataset/DocUNet/
├── crop/     # distorted input images (512×512)
├── crop512/  # distorted input images (512×512, alternative format)
└── scan/     # flat GT scans
```

### DIR300 Benchmark

```
dataset/DIR300/
├── dist/   # distorted input images
└── gt/     # ground-truth flat images
```

---

## Training

Update dataset paths in `config/settings.py` (`class train`), then:

```bash
# Train on Doc3D + UVDoc (mixed)
python train.py
```

Training logs and checkpoints are saved to `checkpoints/Cascade_<dataset_name>/`.

---

## Evaluation

Set `config/settings.py` `eval.dataset_name` and run the corresponding script:

```bash
# DocUNet benchmark
python val.py          # set eval.dataset_name = "docunet"

# UVDoc benchmark
python val_uvdoc.py

# DIR300 benchmark
python val_dir300.py
```

Dewarped images are saved to `sampling/<dataset_name>/test/pred_dewarped/`.

### Computing Metrics

After generating predictions, compute MS-SSIM / LD / AD / CER / ED:

```bash
# UVDoc
python evaluation/UVDocBenchmark_eval.py \
    --uvdoc-path /path/to/dataset/UVDoc/benchmark \
    --pred-path sampling/uvdoc/test/pred_dewarped \
    --save-path sampling/uvdoc/test

# DocUNet
python evaluation/docUnet_eval.py \
    --docunet-path /path/to/dataset/DocUNet/scan \
    --pred-path sampling/docunet/test/pred_dewarped \
    --inputs-path /path/to/dataset/DocUNet/crop \
    --save-path sampling/docunet/test

# DIR300
python evaluation/dir300_eval.py \
    --dir300-path /path/to/dataset/DIR300/gt \
    --pred-path sampling/dir300/test/pred_dewarped \
    --inputs-path /path/to/dataset/DIR300/dist \
    --save-path sampling/dir300/test
```

> Visual metrics (MS-SSIM, LD, AD) require MATLAB Engine for Python.

---

## Single-Image Inference

```bash
python infer.py --image /path/to/distorted_doc.jpg

# Specify output path and model
python infer.py \
    --image /path/to/distorted_doc.jpg \
    --output /path/to/dewarped.png \
    --model-path checkpoints/best_model.pt \
    --device cuda:0
```

The output is saved at the original input resolution.

---

## TPS-based Interactive Refinement

To facilitate practical deployment, CCD provides an intuitive human-in-the-loop interface for interactive refinement. Leveraging the TPS parameterization, the interface is initialized with the model-predicted source control points, providing an accurate starting point for manual adjustment. Users can refine local geometry by directly dragging sparse control points.

The interface supports two adjustment modes:

- **Forward Mode:** The estimated source mesh on the distorted image is fixed. Users directly adjust target control points overlaid on the preliminary warped result to locally stretch or compress the image and eliminate residual distortions.
- **Backward Mode:** The target mesh in the flattened coordinate space is fixed as a rigid, uniform grid. Users directly adjust source control points on the distorted input image to align them with the physical document geometry.

### Usage

1. Enable control-point export in `config/settings.py`:

   ```python
   class eval:
       save_tps_control_points = True
   ```

2. Run `infer.py` as above. In addition to the dewarped image, it saves `<output_name>_control_points.json` next to the output image. During benchmark evaluation, JSON files are saved to `sampling/<dataset_name>/<experiment_name>/pred_control_points/`.

3. Install the GUI dependency and launch the tool:

   ```bash
   pip install PyQt5
   cd TPS-Visulization
   python ForwardBackwardWarp_paperdemo.py
   ```

4. In the GUI, load the original distorted image and click **Load** to select the exported JSON file. Choose **Forward** or **Backward** mode according to the desired adjustment paradigm, drag the relevant control points to refine the dewarping result, then click **Save Warped Image**.

Control-point export is disabled by default (`eval.save_tps_control_points = False`).

The demonstrations below use [the input image](TPS-Visulization/32_4.png) and its [exported TPS control points](TPS-Visulization/32_4.json).

<div align="center">
  <video src="https://github.com/user-attachments/assets/ffea88c9-6071-4150-b953-b8e92090df62" width="70%" poster=""> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/21cc6b33-1d17-4c7b-9709-88ebae85b033" width="70%" poster=""> </video>
</div>

<table>
  <tr>
    <td align="center"><b>Forward Warp</b></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/ffea88c9-6071-4150-b953-b8e92090df62" width="70%" poster=""> </video></td>
  </tr>
  <tr>
    <td align="center"><b>Backward Warp</b></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/21cc6b33-1d17-4c7b-9709-88ebae85b033" width="70%" poster=""> </video></td>
  </tr>
</table>




---

## Acknowledgements

We sincerely thank the following projects, since our code is largely based on or inspired by
[improved-diffusion](https://github.com/openai/improved-diffusion),
[DiT](https://github.com/facebookresearch/DiT), and
[DvD](https://github.com/hanquansanren/DvD),
[UVDoc](https://github.com/tanguymagne/UVDoc),
[DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet), and
[DocScanner](https://github.com/fh2019ustc/DocScanner).

