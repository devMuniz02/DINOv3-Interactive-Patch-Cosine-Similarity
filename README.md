[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# DINOv3 Patch Similarity Viewer

> **Note:** This README and repository are for educational purposes. The creation of this repo was inspired by the DINOv3 paper to help visualize and understand the output of the model.

This repository provides interactive tools to visualize and explore patch-wise similarity in images using the DINOv3 vision transformer model. It is designed for researchers, students, and practitioners interested in understanding how self-supervised vision transformers perceive and relate different regions of an image.

## Overview

Interactive tool to visualize patch-wise similarity in images using DINOv3 models. Select any Hugging Face DINOv3 model, load an image, and explore cosine similarity heatmaps for educational and research purposes.

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `.gitattributes` | Top-level file included in the repository. |
| `.gitignore` | Top-level file included in the repository. |
| `app.py` | Top-level file included in the repository. |
| `camera_similarity.py` | Top-level file included in the repository. |
| `DINOv3CosSimilarity.py` | Top-level file included in the repository. |
| `LICENSE` | Repository license information. |
| `PatchCosSimilarity.ipynb` | Notebook used for experiments, analysis, or interactive demos. |
| `README.md` | Primary project documentation. |
| `requirements.txt` | Python dependency specification for local setup. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/DINOv3-Interactive-Patch-Cosine-Similarity.git
   cd DINOv3-Interactive-Patch-Cosine-Similarity
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Run the main application:
```bash
python app.py
```

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

## Usage

### Gradio app

Run the Gradio app:

```bash
python app.py
```

After runnig the app, go to [http://localhost:7860/](http://localhost:7860/) to see the app running.

Then: 
- Choose Dataset and model name
- For Single image similarity:
    - Choose only one file or URL
- For 2 image similarity:
    - Choose images from file and/or URL
- Click button "Initialize / Update "
- Select the desired patch from the image
- Watch the results

**Note:** 

*Overlay alpha* is the intensity of the overlay of patches on top of image

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

## Demo Single Image

![Interactive Patch Similarity Demo](assets/Test_Interactive_video.gif)

## Demo 2 Images

![Multiple Interactive Patch Similarity Demo](assets/Multiple_Interactive_test_video.gif)

### Real-time Camera Similarity

Experience DINOv3 patch similarity in real-time with your camera:

```bash
python camera_similarity.py
```

This interactive camera application displays a 3x2 grid showing:
- **Original**: Live camera feed
- **Original + Cosine Overlay**: Camera feed with similarity heatmap overlay
- **Cosine Overlay Only**: Pure similarity heatmap visualization
- **Last Selection**: Shows the frame where you made your last patch selection with grid lines
- **PCA Visualization**: 3-component PCA visualization of patch features
- **Empty**: Reserved space for future features

**Camera Controls:**
- Click on any patch in the camera feed to select it as reference
- The similarity heatmap will update in real-time showing patches similar to your selection
- Press ESC to quit

**Demo Video:**
![Camera Similarity Demo](assets/DINOv3CameraTest.mp4)

**Requirements:**
- Camera/webcam connected to your system
- OpenCV for camera capture and display

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

## Purpose

This repository provides interactive tools to visualize and explore patch-wise similarity in images using the DINOv3 vision transformer model. It is designed for researchers, students, and practitioners interested in understanding how self-supervised vision transformers perceive and relate different regions of an image.
