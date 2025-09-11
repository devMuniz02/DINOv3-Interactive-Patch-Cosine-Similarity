# app.py
# Gradio UI for interactive DINOv3 patch similarity (single or dual image)
# - No AutoImageProcessor, no resize (only pad to multiple of patch size)
# - Single image: click to show self-similarity; selected cell outlined in RED
# - Two images: click on one side -> self overlay on source, cross overlay on target; best match on target outlined in YELLOW
# - Red selection rectangle is hidden on the non-active image
# - Patch size inferred from model (no override). Patch indices are not annotated.
# - Dataset selector (LVD-1689M / SAT-493M); model dropdown shows only the short name between "dinov3-" and "-pretrain".
# - Sample URL dropdowns switch between LVD (COCO/Picsum) and SAT (satellite imagery) and auto-fill / clear uploads.

import os
import io
import math
import urllib.request
from functools import lru_cache
from typing import Optional, Tuple, Dict, List

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from transformers import AutoModel
from matplotlib import colormaps as cm

token = os.environ.get("HF_TOKEN")

# ---------- Provided model IDs (ground truth list) ----------
MODEL_ID_LIST = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-sat493m",
    "facebook/dinov3-vit7b16-pretrain-sat493m",
]

DATASET_LABELS = {
    "LVD-1689M": "lvd1689m",
    "SAT-493M": "sat493m",
}

def build_model_maps(model_ids: List[str]):
    """
    Returns:
      valid_map[(dataset_key, short_name)] -> full_model_id
      options_by_dataset[dataset_key] -> [short_name,...]  (display order preserved)
    """
    valid_map: Dict[Tuple[str, str], str] = {}
    options_by_dataset: Dict[str, List[str]] = {"lvd1689m": [], "sat493m": []}

    for mid in model_ids:
        # Expect pattern: "facebook/dinov3-<short>-pretrain-<dataset>"
        try:
            prefix = "facebook/dinov3-"
            start = mid.index(prefix) + len(prefix)
            pre_idx = mid.index("-pretrain", start)
            short = mid[start:pre_idx]
            dataset = mid.split("-pretrain-")[-1].strip()
        except Exception:
            # Skip anything that doesn't match the expected pattern
            continue

        key = (dataset, short)
        valid_map[key] = mid
        if dataset in options_by_dataset and short not in options_by_dataset[dataset]:
            options_by_dataset[dataset].append(short)

    return valid_map, options_by_dataset

VALID_MODEL_MAP, MODEL_OPTIONS_BY_DATASET = build_model_maps(MODEL_ID_LIST)

# ---------- Defaults / knobs ----------
DEFAULT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_DATASET_LABEL = "LVD-1689M"  # initial radio
DEFAULT_OVERLAY_ALPHA = 0.55
DEFAULT_SHOW_GRID = True

# ---------- Normalization presets ----------
NORMALIZE_STATS = {
    "lvd1689m": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
    "sat493m": {
        "mean": [0.430, 0.411, 0.296],
        "std":  [0.213, 0.156, 0.143],
    },
}

# ---------- Sample image URLs (dependent on dataset) ----------
SAMPLE_URL_CHOICES: Dict[str, List[Tuple[str, str]]] = {
    # LVD: current ones
    "lvd1689m": [
        ("– choose a sample –", ""),
        ("COCO: 2 Cats on sofa (039769)", "http://images.cocodataset.org/val2017/000000039769.jpg"),
        ("COCO: Person skiing (000785)", "http://images.cocodataset.org/val2017/000000000785.jpg"),
        ("COCO: People running (000872)", "http://images.cocodataset.org/val2017/000000000872.jpg"),
        ("Picsum: Mountain (ID=1000)", "https://picsum.photos/id/1000/800/600"),
        ("Picsum: Kayak (ID=1011)", "https://picsum.photos/id/1011/800/600"),
        ("Picsum: Man and dog (ID=1012)", "https://picsum.photos/id/1012/800/600"),
    ],
    # SAT: satellite imagery examples
    "sat493m": [
        ("– choose a satellite sample –", ""),
        ("Los Angeles — Downtown", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-13162953.111392,4035684.000887,-13162647.363277,4035989.748999&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Chicago — The Loop", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-9755772.575579,5142721.481539,-9755466.827467,5143027.229656&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("San Francisco — FiDi", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-13625779.317660,4549493.705020,-13625473.569543,4549799.453132&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Seattle — Downtown", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-13618135.614829,6041468.060117,-13617829.866717,6041773.808232&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Houston — Downtown", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-10616682.825155,3472648.850537,-10616377.077043,3472954.598652&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Boston — Downtown", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-7910429.838718,5214954.473271,-7910124.090606,5215260.221383&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Miami — Brickell", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-8927271.625996,2970992.633903,-8926965.877884,2971298.382015&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Washington, DC — White House area", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-8575814.169943,4706877.546259,-8575508.421826,4707183.294371&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Philadelphia — City Hall", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-8367523.267865,4858910.795516,-8367217.519750,4859216.543633&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Mexico — Monterrey Macroplaza", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-11167335.176921,2957692.590981,-11167029.428809,2957998.339093&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Mexico — Guadalajara Centro", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-11504728.219772,2353228.571302,-11504422.471660,2353534.319414&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Mexico — CDMX Zócalo", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-11035634.177186,2205781.543740,-11035328.429074,2206087.291852&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Texas — Dallas Downtown", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-10775518.969934,3865535.175922,-10775213.221817,3865840.924038&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Texas — Austin Capitol", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-10880543.446795,3538766.880005,-10880237.698683,3539072.628117&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
        ("Texas — San Antonio River Walk", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-10964394.866824,3429614.803614,-10964089.118712,3429920.551726&bboxSR=102100&imageSR=102100&size=1024,1024&format=jpg&f=image"),
    ],
}

def _sample_labels_for(dataset_label: str):
    key = DATASET_LABELS.get(dataset_label, "lvd1689m")
    return [label for label, _ in SAMPLE_URL_CHOICES.get(key, [])]

def _apply_sample(dataset_label: str, sample_label: str):
    """Fill textbox with chosen sample URL and clear any uploaded image."""
    key = DATASET_LABELS.get(dataset_label, "lvd1689m")
    sample_map = dict(SAMPLE_URL_CHOICES.get(key, []))
    url = sample_map.get(sample_label, "")
    return gr.update(value=url), None  # (textbox update, clear upload)

# ---------- Utility ----------
def load_image_from_any(src: Optional[Image.Image], url: Optional[str]) -> Optional[Image.Image]:
    # Prefer URL if present
    if url and str(url).strip().lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    return None

def pad_to_multiple(pil_img: Image.Image, multiple: int = 16) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    W, H = pil_img.size
    H_pad = int(math.ceil(H / multiple) * multiple)
    W_pad = int(math.ceil(W / multiple) * multiple)
    if (H_pad, W_pad) == (H, W):
        return pil_img, (0, 0, 0, 0)
    canvas = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))
    canvas.paste(pil_img, (0, 0))
    return canvas, (0, 0, W_pad - W, H_pad - H)

def preprocess_no_resize(pil_img: Image.Image, multiple: int = 16, dataset_key: str = "lvd1689m"):
    img_padded, pad_box = pad_to_multiple(pil_img, multiple=multiple)

    # Pick stats based on dataset (default to LVD if unknown)
    stats = NORMALIZE_STATS.get(dataset_key, NORMALIZE_STATS["lvd1689m"])
    mean, std = stats["mean"], stats["std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    pixel_tensor = transform(img_padded).unsqueeze(0)  # (1,3,H,W)
    disp_np = np.array(img_padded, dtype=np.uint8)
    return {"pixel_values": pixel_tensor}, disp_np, pad_box


def upsample_nearest(arr: np.ndarray, H: int, W: int, ps: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr.repeat(ps, 0).repeat(ps, 1)
    elif arr.ndim == 3:
        rows, cols, ch = arr.shape
        arr2 = arr.repeat(ps, 0).repeat(ps, 1)
        return arr2.reshape(rows * ps, cols * ps, ch)
    raise ValueError("upsample_nearest expects (rows,cols) or (rows,cols,channels)")

def blend_overlay(base_uint8: np.ndarray, overlay_rgb_float: np.ndarray, alpha: float) -> np.ndarray:
    base = base_uint8.astype(np.float32)
    over = (overlay_rgb_float * 255.0).astype(np.float32)
    out = (1.0 - alpha) * base + alpha * over
    return np.clip(out, 0, 255).astype(np.uint8)

def draw_grid(img: Image.Image, rows: int, cols: int, ps: int):
    d = ImageDraw.Draw(img)
    W, H = img.size
    for r in range(1, rows):
        y = r * ps
        d.line([(0, y), (W, y)], fill=(255, 255, 255), width=1)
    for c in range(1, cols):
        x = c * ps
        d.line([(x, 0), (x, H)], fill=(255, 255, 255), width=1)

def rc_to_idx(r: int, c: int, cols: int) -> int:
    return int(r) * cols + int(c)

def idx_to_rc(i: int, cols: int) -> Tuple[int, int]:
    return int(i) // cols, int(i) % cols

# ---------- Model cache ----------
@lru_cache(maxsize=3)
def load_model_cached(full_model_id: str, device_str: str):
    device = torch.device(device_str)
    model = AutoModel.from_pretrained(full_model_id).to(device)
    model.eval()
    return model

def infer_patch_size(model, default: int = 16) -> int:
    if hasattr(model, "config") and hasattr(model.config, "patch_size"):
        ps = model.config.patch_size
        if isinstance(ps, (tuple, list)): return int(ps[0])
        return int(ps)
    if hasattr(model, "patch_size"):
        ps = model.patch_size
        if isinstance(ps, (tuple, list)): return int(ps[0])
        return int(ps)
    return default

# ---------- Per-image state ----------
class PatchImageState:
    def __init__(self, pil_img: Image.Image, model, device_str: str, ps: int, dataset_key: str):
        self.pil = pil_img
        self.ps = ps
        self.dataset_key = dataset_key
        inputs, disp_np, _ = preprocess_no_resize(pil_img, multiple=ps, dataset_key=dataset_key)
        self.disp = disp_np
        pv = inputs["pixel_values"].to(device_str)  # (1,3,H,W)
        _, _, H, W = pv.shape
        self.H, self.W = int(H), int(W)
        self.rows, self.cols = self.H // ps, self.W // ps

        with torch.no_grad():
            out = model(pixel_values=pv)
        hs = out.last_hidden_state.squeeze(0).detach().cpu().numpy()  # (T,D)

        T, D = hs.shape
        n_patches = self.rows * self.cols
        n_special = T - n_patches  # class + maybe registers
        if n_special < 1:
            raise RuntimeError(
                f"Token mismatch: T={T}, rows*cols={n_patches}, HxW={self.H}x{self.W}, ps={ps}"
            )
        self.D = D
        patches = hs[n_special:, :].reshape(self.rows, self.cols, D)
        self.X = patches.reshape(-1, D)
        self.Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-8)

# ---------- Rendering / compute ----------
def render_with_cosmap(
    st: PatchImageState,
    cos_map: Optional[np.ndarray],
    overlay_alpha: float,
    show_grid_flag: bool,
    select_idx: Optional[int] = None,
    best_idx: Optional[int] = None,
) -> Image.Image:
    H, W, ps = st.H, st.W, st.ps
    rows, cols = st.rows, st.cols

    if cos_map is None:
        disp = np.full((rows, cols), 0.5, dtype=np.float32)
    else:
        vmin, vmax = 0.0, 1.0
        rng = vmax - vmin if vmax > vmin else 1e-8
        disp = (cos_map - vmin) / rng

    cmap = cm.get_cmap("magma")
    rgba = cmap(disp)
    rgb = rgba[..., :3]

    if select_idx is not None:
        rs, cs = idx_to_rc(select_idx, cols)
        rgb[rs, cs, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    over_rgb_up = upsample_nearest(rgb, H, W, ps)
    blended = blend_overlay(st.disp, over_rgb_up, float(overlay_alpha))
    pil = Image.fromarray(blended)

    draw = ImageDraw.Draw(pil)
    if show_grid_flag:
        draw_grid(pil, rows, cols, ps)

    if select_idx is not None:
        r, c = idx_to_rc(select_idx, cols)
        x0, y0 = c * ps, r * ps
        x1, y1 = x0 + ps - 1, y0 + ps - 1
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=2)

    if best_idx is not None:
        r, c = idx_to_rc(best_idx, cols)
        x0, y0 = c * ps, r * ps
        x1, y1 = x0 + ps - 1, y0 + ps - 1
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 255, 0), width=2)

    return pil

def compute_self_and_cross(
    src: PatchImageState,
    tgt: Optional[PatchImageState],
    q_idx: int,
):
    q = src.X[q_idx]
    qn = q / (np.linalg.norm(q) + 1e-8)

    cos_self = src.Xn @ qn
    cos_map_self = cos_self.reshape(src.rows, src.cols)
    self_stats = (float(cos_map_self.min()), float(cos_map_self.max()))

    cross_result = None
    cos_map_cross = None
    if tgt is not None:
        cos_cross = tgt.Xn @ qn
        cos_map_cross = cos_cross.reshape(tgt.rows, tgt.cols)
        cross_min, cross_max = float(cos_map_cross.min()), float(cos_map_cross.max())
        best_idx = int(np.argmax(cos_cross))
        cross_result = (cross_min, cross_max, best_idx)

    return cos_map_self, cos_map_cross, self_stats, cross_result

# ---------- Gradio helpers for model & samples ----------
def dataset_label_to_key(label: str) -> str:
    return DATASET_LABELS.get(label, "lvd1689m")

def update_model_dropdown(dataset_label: str):
    key = dataset_label_to_key(dataset_label)
    opts = MODEL_OPTIONS_BY_DATASET.get(key, [])
    default_val = opts[0] if opts else None
    return gr.update(choices=opts, value=default_val)

def update_model_and_samples(dataset_label: str):
    # Update model dropdown
    model_update = update_model_dropdown(dataset_label)
    # Update both sample dropdowns to dataset-specific options
    labels = _sample_labels_for(dataset_label)
    sample_update = gr.update(choices=labels, value=(labels[0] if labels else None))
    return model_update, sample_update, sample_update

def resolve_full_model_id(dataset_label: str, short_name: str) -> Optional[str]:
    key = (dataset_label_to_key(dataset_label), short_name)
    return VALID_MODEL_MAP.get(key)

# ---------- Gradio callbacks ----------
def init_states(
    left_img_in: Optional[Image.Image],
    left_url: str,
    right_img_in: Optional[Image.Image],
    right_url: str,
    dataset_label: str,
    short_model: str,
    show_grid_flag: bool,
    overlay_alpha: float,
):
    # Resolve images
    left_img = load_image_from_any(left_img_in, left_url)
    right_img = load_image_from_any(right_img_in, right_url)
    if left_img is None and right_img is None:
        left_img = load_image_from_any(None, DEFAULT_URL)

    # Resolve model
    full_model_id = resolve_full_model_id(dataset_label, short_model)
    if not full_model_id:
        return (gr.update(), gr.update(), None, None, 0, -1, -1, 16,
                f"❌ Model not available: {dataset_label} / {short_model}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_cached(full_model_id, device_str)
    ps = infer_patch_size(model, 16)
    
    # Get dataset_key ("lvd1689m" or "sat493m") from the radio label
    dataset_key = dataset_label_to_key(dataset_label)

    left_state = PatchImageState(left_img, model, device_str, ps, dataset_key) if left_img is not None else None
    right_state = PatchImageState(right_img, model, device_str, ps, dataset_key) if right_img is not None else None

    active_side = 0 if left_state is not None else 1

    status = f"✔ Loaded: {full_model_id}  |  ps={ps}"
    out_left, out_right = None, None

    if left_state is not None and right_state is not None:
        q_idx = (left_state.rows // 2) * left_state.cols + (left_state.cols // 2)
        cos_self, cos_cross, (smin, smax), cross_info = compute_self_and_cross(left_state, right_state, q_idx)
        best_idx = cross_info[2] if cross_info else None
        out_left = render_with_cosmap(left_state, cos_self, overlay_alpha, show_grid_flag,
                                      select_idx=q_idx, best_idx=None)
        out_right = render_with_cosmap(right_state, cos_cross, overlay_alpha, show_grid_flag,
                                       select_idx=None, best_idx=best_idx)
        status += (f"  |  LEFT {left_state.rows}x{left_state.cols} self∈[{smin:.3f},{smax:.3f}]  "
                   f"|  RIGHT cross best={best_idx}")
        left_idx, right_idx = q_idx, (right_state.rows // 2) * right_state.cols + (right_state.cols // 2)
    elif left_state is not None:
        q_idx = (left_state.rows // 2) * left_state.cols + (left_state.cols // 2)
        cos_self, _, (smin, smax), _ = compute_self_and_cross(left_state, None, q_idx)
        out_left = render_with_cosmap(left_state, cos_self, overlay_alpha, show_grid_flag,
                                      select_idx=q_idx, best_idx=None)
        status += f"  |  Single LEFT {left_state.rows}x{left_state.cols} self∈[{smin:.3f},{smax:.3f}]"
        left_idx, right_idx = q_idx, -1
    else:
        q_idx = (right_state.rows // 2) * right_state.cols + (right_state.cols // 2)
        cos_self, _, (smin, smax), _ = compute_self_and_cross(right_state, None, q_idx)
        out_right = render_with_cosmap(right_state, cos_self, overlay_alpha, show_grid_flag,
                                       select_idx=q_idx, best_idx=None)
        status += f"  |  Single RIGHT {right_state.rows}x{right_state.cols} self∈[{smin:.3f},{smax:.3f}]"
        left_idx, right_idx = -1, q_idx

    return (
        out_left, out_right,
        left_state, right_state,
        active_side,
        left_idx, right_idx,
        ps,
        status
    )

def _coords_to_idx(x: int, y: int, st: PatchImageState) -> int:
    r = int(np.clip(y // st.ps, 0, st.rows - 1))
    c = int(np.clip(x // st.ps, 0, st.cols - 1))
    return rc_to_idx(r, c, st.cols)

def on_select_left(
    evt: gr.SelectData,
    left_state: Optional[PatchImageState],
    right_state: Optional[PatchImageState],
    show_grid_flag: bool,
    overlay_alpha: float,
    ps: int,
):
    if left_state is None:
        return gr.update(), gr.update(), 0, -1, -1, "Upload/Load a LEFT image first."

    x, y = evt.index
    q_idx = _coords_to_idx(x, y, left_state)

    if right_state is not None:
        cos_self, cos_cross, (smin, smax), cross_info = compute_self_and_cross(left_state, right_state, q_idx)
        best_idx = cross_info[2]
        out_left = render_with_cosmap(left_state, cos_self, overlay_alpha, show_grid_flag,
                                      select_idx=q_idx, best_idx=None)
        out_right = render_with_cosmap(right_state, cos_cross, overlay_alpha, show_grid_flag,
                                       select_idx=None, best_idx=best_idx)
        status = (f"LEFT {left_state.rows}x{left_state.cols} self∈[{smin:.3f},{smax:.3f}]  |  "
                  f"RIGHT cross best idx={best_idx}")
        return out_left, out_right, 0, q_idx, -1, status
    else:
        cos_self, _, (smin, smax), _ = compute_self_and_cross(left_state, None, q_idx)
        out_left = render_with_cosmap(left_state, cos_self, overlay_alpha, show_grid_flag,
                                      select_idx=q_idx, best_idx=None)
        status = f"Single LEFT • idx={q_idx} • self∈[{smin:.3f},{smax:.3f}]"
        return out_left, gr.update(), 0, q_idx, -1, status

def on_select_right(
    evt: gr.SelectData,
    left_state: Optional[PatchImageState],
    right_state: Optional[PatchImageState],
    show_grid_flag: bool,
    overlay_alpha: float,
    ps: int,
):
    if right_state is None:
        return gr.update(), gr.update(), 1, -1, -1, "Upload/Load a RIGHT image first."

    x, y = evt.index
    q_idx = _coords_to_idx(x, y, right_state)

    if left_state is not None:
        cos_self, cos_cross, (smin, smax), cross_info = compute_self_and_cross(right_state, left_state, q_idx)
        best_idx = cross_info[2]
        out_right = render_with_cosmap(right_state, cos_self, overlay_alpha, show_grid_flag,
                                       select_idx=q_idx, best_idx=None)
        out_left = render_with_cosmap(left_state, cos_cross, overlay_alpha, show_grid_flag,
                                      select_idx=None, best_idx=best_idx)
        status = (f"RIGHT {right_state.rows}x{right_state.cols} self∈[{smin:.3f},{smax:.3f}]  |  "
                  f"LEFT cross best idx={best_idx}")
        return out_left, out_right, 1, -1, q_idx, status
    else:
        cos_self, _, (smin, smax), _ = compute_self_and_cross(right_state, None, q_idx)
        out_right = render_with_cosmap(right_state, cos_self, overlay_alpha, show_grid_flag,
                                       select_idx=q_idx, best_idx=None)
        status = f"Single RIGHT • idx={q_idx} • self∈[{smin:.3f},{smax:.3f}]"
        return gr.update(), out_right, 1, -1, q_idx, status

def rebuild_with_settings(
    left_state: Optional[PatchImageState],
    right_state: Optional[PatchImageState],
    active_side: int,
    left_idx: int,
    right_idx: int,
    show_grid_flag: bool,
    overlay_alpha: float,
    ps: int,
):
    if left_state is None and right_state is None:
        return gr.update(), gr.update(), "Load an image first."

    if left_state is not None and right_state is not None:
        if active_side == 0:
            q_idx = left_idx if left_idx >= 0 else (left_state.rows//2)*left_state.cols + (left_state.cols//2)
            cos_self, cos_cross, _, cross_info = compute_self_and_cross(left_state, right_state, q_idx)
            best_idx = cross_info[2]
            out_left = render_with_cosmap(left_state, cos_self, overlay_alpha, show_grid_flag,
                                          select_idx=q_idx, best_idx=None)
            out_right = render_with_cosmap(right_state, cos_cross, overlay_alpha, show_grid_flag,
                                           select_idx=None, best_idx=best_idx)
        else:
            q_idx = right_idx if right_idx >= 0 else (right_state.rows//2)*right_state.cols + (right_state.cols//2)
            cos_self, cos_cross, _, cross_info = compute_self_and_cross(right_state, left_state, q_idx)
            best_idx = cross_info[2]
            out_right = render_with_cosmap(right_state, cos_self, overlay_alpha, show_grid_flag,
                                           select_idx=q_idx, best_idx=None)
            out_left = render_with_cosmap(left_state, cos_cross, overlay_alpha, show_grid_flag,
                                          select_idx=None, best_idx=best_idx)
        return out_left, out_right, "Updated overlays."
    elif left_state is not None:
        q_idx = left_idx if left_idx >= 0 else (left_state.rows//2)*left_state.cols + (left_state.cols//2)
        cos_self, _, _, _ = compute_self_and_cross(left_state, None, q_idx)
        out_left = render_with_cosmap(left_state, cos_self, overlay_alpha, show_grid_flag,
                                      select_idx=q_idx, best_idx=None)
        return out_left, gr.update(), "Updated overlays."
    else:
        q_idx = right_idx if right_idx >= 0 else (right_state.rows//2)*right_state.cols + (right_state.cols//2)
        cos_self, _, _, _ = compute_self_and_cross(right_state, None, q_idx)
        out_right = render_with_cosmap(right_state, cos_self, overlay_alpha, show_grid_flag,
                                       select_idx=q_idx, best_idx=None)
        return gr.update(), out_right, "Updated overlays."

# ---------- Gradio UI ----------
with gr.Blocks(title="DINOv3 Patch Similarity (Self & Cross)") as demo:
    gr.Markdown(
        """
        # DINOv3 Patch Similarity (Self & Cross)
        1) Pick **Dataset** (LVD-1689M / SAT-493M).  
        2) Pick **Model**.  
        3) Upload one or two images (or paste URLs) and press **Initialize / Update**.  
        - Click on a patch to update overlays.  
        - In two-image mode, the non-active image hides the red selection and shows **yellow** best match.
        """
    )

    with gr.Row():
        dataset_radio = gr.Radio(
            label="Dataset",
            choices=list(DATASET_LABELS.keys()),
            value=DEFAULT_DATASET_LABEL,
            interactive=True
        )
        initial_key = DATASET_LABELS[DEFAULT_DATASET_LABEL]
        initial_models = MODEL_OPTIONS_BY_DATASET.get(initial_key, [])
        model_dropdown = gr.Dropdown(
            label="Model name",
            choices=initial_models,
            value=(initial_models[0] if initial_models else None),
            interactive=True
        )

    # initial sample labels based on default dataset
    initial_sample_labels = [label for label, _ in SAMPLE_URL_CHOICES.get(initial_key, [])]

    with gr.Row():
        with gr.Column():
            left_input = gr.Image(label="Left Image (upload)", type="pil",
                                  sources=["upload", "clipboard", "webcam"], interactive=True)
            left_url = gr.Textbox(label="Left Image URL (optional)", placeholder="https://...")
            left_sample = gr.Dropdown(label="Use a sample URL",
                                      choices=initial_sample_labels,
                                      value=(initial_sample_labels[0] if initial_sample_labels else None),
                                      interactive=True)
        with gr.Column():
            right_input = gr.Image(label="Right Image (upload)", type="pil",
                                   sources=["upload", "clipboard", "webcam"], interactive=True)
            right_url = gr.Textbox(label="Right Image URL (optional)", placeholder="https://...")
            right_sample = gr.Dropdown(label="Use a sample URL",
                                       choices=initial_sample_labels,
                                       value=(initial_sample_labels[0] if initial_sample_labels else None),
                                       interactive=True)

    with gr.Accordion("Overlay Settings", open=True):
        show_grid = gr.Checkbox(label="Show patch grid", value=DEFAULT_SHOW_GRID)
        overlay_alpha = gr.Slider(label="Overlay alpha", minimum=0.0, maximum=1.0,
                                  value=DEFAULT_OVERLAY_ALPHA, step=0.01)

    init_btn = gr.Button("Initialize / Update", variant="primary")

    with gr.Row():
        left_view = gr.Image(label="LEFT (click to select patch)", interactive=True)
        right_view = gr.Image(label="RIGHT (click to select patch)", interactive=True)

    status = gr.Markdown("")

    # Hidden states
    left_state = gr.State(None)
    right_state = gr.State(None)
    active_side = gr.State(0)
    left_idx = gr.State(-1)
    right_idx = gr.State(-1)
    ps_state = gr.State(16)

    # Update model dropdown and sample lists when dataset changes
    dataset_radio.change(
        fn=update_model_and_samples,
        inputs=[dataset_radio],
        outputs=[model_dropdown, left_sample, right_sample]
    )

    # When a sample is chosen, set URL and clear any uploaded image (prefer URL)
    left_sample.change(
        fn=_apply_sample,
        inputs=[dataset_radio, left_sample],
        outputs=[left_url, left_input]
    )
    right_sample.change(
        fn=_apply_sample,
        inputs=[dataset_radio, right_sample],
        outputs=[right_url, right_input]
    )

    # Initialize / reload model + overlays
    init_btn.click(
        fn=init_states,
        inputs=[left_input, left_url, right_input, right_url, dataset_radio, model_dropdown, show_grid, overlay_alpha],
        outputs=[left_view, right_view, left_state, right_state, active_side, left_idx, right_idx, ps_state, status],
        show_progress=True
    )

    # Click handlers
    left_view.select(
        fn=on_select_left,
        inputs=[left_state, right_state, show_grid, overlay_alpha, ps_state],
        outputs=[left_view, right_view, active_side, left_idx, right_idx, status]
    )
    right_view.select(
        fn=on_select_right,
        inputs=[left_state, right_state, show_grid, overlay_alpha, ps_state],
        outputs=[left_view, right_view, active_side, left_idx, right_idx, status]
    )

    # Live re-render on setting changes
    show_grid.change(
        fn=rebuild_with_settings,
        inputs=[left_state, right_state, active_side, left_idx, right_idx, show_grid, overlay_alpha, ps_state],
        outputs=[left_view, right_view, status]
    )
    overlay_alpha.change(
        fn=rebuild_with_settings,
        inputs=[left_state, right_state, active_side, left_idx, right_idx, show_grid, overlay_alpha, ps_state],
        outputs=[left_view, right_view, status]
    )

if __name__ == "__main__":
    demo.queue().launch()
