# DINOv3 Patch Similarity Viewer

## Demo Single Image

![Interactive Patch Similarity Demo](assets/Test_Interactive_video.gif)

## Demo 2 Images

![Multiple Interactive Patch Similarity Demo](assets/Multiple_Interactive_test_video.gif)

> **Note:** This README and repository are for educational purposes. The creation of this repo was inspired by the DINOv3 paper to help visualize and understand the output of the model.

## Purpose

This repository provides interactive tools to visualize and explore patch-wise similarity in images using the DINOv3 vision transformer model. It is designed for researchers, students, and practitioners interested in understanding how self-supervised vision transformers perceive and relate different regions of an image.

## About DINOv3

- **Paper:** [DINOv3: Self-supervised Vision Transformers with Enormous Teacher Models](https://arxiv.org/abs/2508.10104)
- **Meta Research Page:** [Meta DINOv3 Publication](https://ai.meta.com/dinov3/)
- **Official GitHub:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

**Note:**  
The DINOv3 model weights require access approval.  
You can request access via the [Meta Research page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) or by selecting the desired model on [Hugging Face model collection](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009).

## Features

- **Interactive Visualization:** Click on image patches or use arrow keys to explore patch similarity heatmaps.
- **Single or Two-Image Mode:** If one image is specified, shows self-similarity. If two images are specified, shows both self-similarity and cross-image similarity overlays interactively.
- **Image Preprocessing:** Loads and pads images without resizing, preserving the original aspect ratio.
- **Cosine Similarity Calculation:** Computes and visualizes cosine similarity between image patches.
- **Robust Fallback:** If an image URL fails to load, a default image is used.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Model Selection

You can choose from several DINOv3 models available on Hugging Face (click to view each model card):

- [facebook/dinov3-vit7b16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)
- [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
- [facebook/dinov3-convnext-small-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)
- [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- [facebook/dinov3-convnext-base-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m)
- [facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
- [facebook/dinov3-convnext-tiny-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m)
- [facebook/dinov3-vitl16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m)
- [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- [facebook/dinov3-convnext-large-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m)
- [facebook/dinov3-vit7b16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m)

## Usage

### Python Script

Run the interactive viewer with the default COCO image:

```bash
python DINOv3CosSimilarity.py
```

#### Single Image Mode

Specify your own image (local path or URL):

```bash
python DINOv3CosSimilarity.py --image path/to/your/image.jpg
python DINOv3CosSimilarity.py --image https://yourdomain.com/image.png
```

#### Two Image Mode

Specify two images (local paths or URLs):

```bash
python DINOv3CosSimilarity.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
python DINOv3CosSimilarity.py --image1 https://yourdomain.com/image1.png --image2 https://yourdomain.com/image2.png
```

#### Model Selection

Specify the model with `--model` (default is vits16):

```bash
python DINOv3CosSimilarity.py --model facebook/dinov3-vitb16-pretrain-lvd1689m
```

#### Other Options

- `--show_grid` : Draw patch grid
- `--annotate_indices` : Write patch indices on cells
- `--overlay_alpha <float>` : Set heatmap alpha (default 0.55)
- `--patch_size <int>` : Override patch size (default: model's patch size)

#### Controls

- Mouse click to select a patch
- Arrow keys to move selection
- '1', '2', or 't' to switch active image (in two-image mode)
- 'q' to quit

### Jupyter Notebook

1. Open `PatchCosSimilarity.ipynb` in Jupyter Notebook.
2. Run the cells to load an image and visualize patch similarities.
3. Set `url1` for single-image mode, or both `url1` and `url2` for two-image mode.
4. If an image fails to load, a default image will be used automatically.
5. Set the `model_id` variable to any of the models listed above (see commented lines at the top of the notebook).

**Notebook Controls:**  
- Mouse click to select a patch  
- Arrow keys to move selection  
- '1', '2', or 't' to switch active image (in two-image mode)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This project utilizes the DINOv3 model from Hugging Face's Transformers library, along with PyTorch, Matplotlib, and Pillow
