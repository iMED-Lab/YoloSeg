<div align="center">
  <h1>🔎 YoloSeg: You Only Label Once for Medical Image Segmentation</h1>
</div>

---

## 👀 Overview

YoloSeg is a two-stage framework for medical image segmentation using only **one labeled image**.

1. **Foundation model-drvien pseudo-label generation**  
   Generateing multi-view pseudo labels and divergence masks from a single labeled image using SAM2.

2. **Robust pseudo-label learning for segmentation model**  
   Training a segmentation network with dual-component loss and cross-patch data augmentation.

We validated YoloSeg on 10 diverse public datasets, achieving an average Dice only **3.08%** lower than the fully supervised baseline.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/iMED-Lab/YoloSeg.git
cd YoloSeg
```

### 2. Create environments

#### Create a conda environment

```bash
conda create -n yoloseg python=3.10 -y
conda activate yoloseg
```

#### Install PyTorch

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

#### Install SAM2 and other dependencies

```bash
pip install -e ./code_pl
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

Please organize the dataset as follows:

```text
DatasetName/
├── file_list/
│   ├── train_all_frames.txt
│   ├── train_label_frames.txt
│   └── test_all_frames.txt
├── Train/
│   ├── JPEGImages/
│   │   ├── XXX001.png
│   │   ├── XXX002.png
│   │   └── ...
│   └── Annotations/
│       ├── XXX001.png
│       ├── XXX002.png
│       └── ...
└── Test/
    ├── JPEGImages/
    │   ├── XXX101.png
    │   ├── XXX102.png
    │   └── ...
    └── Annotations/
        ├── XXX101.png
        ├── XXX102.png
        └── ...
```

Notes:

* `train_all_frames.txt`: all training image filenames
* `train_label_frames.txt`: the selected labeled image filename for the one-shot setting
* `test_all_frames.txt`: all test image filenames
* Filenames in `file_list/*.txt` should be plain filenames, for example: `XXX001.png`
* Input images should be 3-channel `.png`
* Ground-truth labels should be single-channel `.png`
* Pseudo labels generated in Stage 1 are also single-channel `.png`

We provide a dataset structure example in: `YoloSeg/data/ISIC2016`

---

## 🤖 Model Preparation

Please download SAM2 checkpoints from the official repository: [SAM2 Official Repository](https://github.com/facebookresearch/sam2)

Then place the downloaded checkpoint files under: `YoloSeg/code_pl/checkpoints/`

For example:

```text
YoloSeg/
└── code_pl/
    └── checkpoints/
        ├── sam2.1_hiera_tiny.pt
        ├── sam2.1_hiera_small.pt
        ├── sam2.1_hiera_base_plus.pt
        └── sam2.1_hiera_large.pt
```

We recommend using `sam2.1_hiera_small.pt` by default.

---

## ⚡ Run YoloSeg

#### 1. Stage 1: Multi-view Pseudo-label Generation

Run Stage 1 to generate:

* `pl_original`
* `pl_rotate`
* `pl_flip`
* `divergence_mask`

```bash
python code_pl/multi_view_inference.py \
  --data-root /path/to/DatasetName \
  --checkpoint code_pl/checkpoints/sam2.1_hiera_small.pt \
  --cfg code_pl/configs/sam2.1/sam2.1_hiera_s.yaml
```

After Stage 1, the `Train/` directory will be automatically updated as:

```text
Train/
├── JPEGImages/
├── Annotations/
├── pl_original/
├── pl_rotate/
├── pl_flip/
└── divergence_mask/
```

#### 2. Stage 2: Segmentation Training

Train the segmentation model with the generated pseudo labels:

```bash
python code_seg/train.py \
  --data-root /path/to/DatasetName \
  --exp-name yoloseg_unet \
  --num-classes 2 \
  --image-size 256 \
  --batch-size 4 \
  --epochs 100
```

#### 3. Testing

Run testing with the trained model:

```bash
python code_seg/test.py \
  --data-root /path/to/DatasetName \
  --checkpoint checkpoints/yoloseg_unet/best.pth \
  --output-dir outputs/yoloseg_unet_test \
  --num-classes 2 \
  --image-size 256
```

---

## 🙏 Acknowledgements

We would like to thank the authors of the following open-source projects:

- [SAM2](https://github.com/facebookresearch/sam2)
- [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

Their excellent work has greatly inspired and supported this project.

---

## 📜 Citation

If you find YoloSeg useful, please cite:

```bibtex
@article{yoloseg2026,
  title   = {YoloSeg: You Only Label Once for Medical Image Segmentation},
  author  = {Zhang, Mingen and Gu, Yuanyuan and Wang, Meng and Mou, Lei and Zhang, Jingfeng and Zhao, Yitian},
  journal = {},
  year    = {2026}
}
```

---

## 🧠 Questions

If you have any questions, feel free to contact: **zhangmingen@nimte.ac.cn**
