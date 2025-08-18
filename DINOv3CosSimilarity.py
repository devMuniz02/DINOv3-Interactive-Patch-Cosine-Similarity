# filepath: DINOv3-PatchSimilarity/DINOv3CosSimilarity.py
# dino_patch_explorer.py
# Interactive DINOv3 patch similarity viewer (NO AutoImageProcessor, NO resize)
# - Uses: url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         image = load_image(url)
# - Click a patch or use arrow keys to select it
# - Cosine similarity heatmap updates live
# - Selected patch cell is colored pure RED (cos=1.0)
# - Image is NOT resized; only right/bottom padding to multiples of 16

# ---- User config ----
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
show_grid = True
annotate_indices = False
overlay_alpha = 0.55
patch_size_override = None  # set to 16 to force; None = read from model if available
# ----------------------

import sys, math, io, urllib.request, argparse, os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
try:
    matplotlib.use("TkAgg")  # open a native window when possible
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoModel


DEFAULT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
SHOW_GRID = True
ANNOTATE_INDICES = False
OVERLAY_ALPHA = 0.55
PATCH_SIZE_OVERRIDE = None


# ---------- Image I/O ----------
def load_image(path_or_url):
    """Load image from local path or URL."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img


# ---------- Preprocessing (custom, no resize) ----------
def pad_to_multiple(pil_img, multiple=16):
    """Pad PIL image on right/bottom so (H,W) are multiples of `multiple`."""
    W, H = pil_img.size
    H_pad = int(math.ceil(H / multiple) * multiple)
    W_pad = int(math.ceil(W / multiple) * multiple)
    if (H_pad, W_pad) == (H, W):
        return pil_img, (0, 0, 0, 0)
    canvas = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))
    canvas.paste(pil_img, (0, 0))
    return canvas, (0, 0, W_pad - W, H_pad - H)


def preprocess_image_no_resize(pil_img):
    """Pad (right/bottom) -> ToTensor -> Normalize (ImageNet stats)."""
    img_padded, pad_box = pad_to_multiple(pil_img, multiple=16)
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1], CxHxW
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    pixel_tensor = transform(img_padded).unsqueeze(0)  # (1,3,H,W)
    disp_np = np.array(img_padded, dtype=np.uint8)     # (H,W,3) for display
    return {"pixel_values": pixel_tensor}, disp_np, pad_box


# ---------- Utilities ----------
def upsample_nearest(arr, H, W, ps):
    """Nearest upsample for 2D or 3D arrays with last-dim channels."""
    if arr.ndim == 2:
        return arr.repeat(ps, 0).repeat(ps, 1)
    elif arr.ndim == 3:
        C = arr.shape[-1]
        return arr.repeat(ps, 0).repeat(ps, 1).reshape(H, W, C)
    raise ValueError("Unsupported ndim for upsample")


def draw_grid(ax, rows, cols, ps):
    for r in range(1, rows):
        ax.axhline(r * ps - 0.5, lw=0.8, alpha=0.6, color="white", zorder=3)
    for c in range(1, cols):
        ax.axvline(c * ps - 0.5, lw=0.8, alpha=0.6, color="white", zorder=3)


def draw_indices(ax, rows, cols, ps):
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            ax.text(c * ps + ps / 2, r * ps + ps / 2, str(idx),
                    ha="center", va="center", fontsize=7,
                    color="white", alpha=0.95, zorder=4)


def main():
    # ---- Available DINOv3 models on Hugging Face ----
    # facebook/dinov3-vit7b16-pretrain-lvd1689m
    # facebook/dinov3-vits16-pretrain-lvd1689m
    # facebook/dinov3-convnext-small-pretrain-lvd1689m
    # facebook/dinov3-vitb16-pretrain-lvd1689m
    # facebook/dinov3-convnext-base-pretrain-lvd1689m
    # facebook/dinov3-vits16plus-pretrain-lvd1689m
    # facebook/dinov3-convnext-tiny-pretrain-lvd1689m
    # facebook/dinov3-vitl16-pretrain-sat493m
    # facebook/dinov3-vitl16-pretrain-lvd1689m
    # facebook/dinov3-vith16plus-pretrain-lvd1689m
    # facebook/dinov3-convnext-large-pretrain-lvd1689m
    # facebook/dinov3-vit7b16-pretrain-sat493m

    DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

    parser = argparse.ArgumentParser(description="DINOv3 Patch Similarity Viewer")
    parser.add_argument("--image", type=str, default=DEFAULT_URL,
                        help="Path or URL to image (default: COCO sample image)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID,
                        help="DINOv3 model name from Hugging Face (default: vits16). See script for options.")
    args = parser.parse_args()

    # Load URL image exactly as requested
    image = load_image(args.image)  # <-- your requested call
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}; Image size: {image.size[1]}x{image.size[0]} (HxW displayed after padding)")
    print(f"[info] Using model: {args.model}")

    # Model
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    # Preprocess (no resize, right/bottom pad only)
    inputs, img_disp, _ = preprocess_image_no_resize(image)
    pixel_values = inputs["pixel_values"].to(device)  # (1,3,H,W)
    _, _, H, W = pixel_values.shape

    # Patch size
    ps = PATCH_SIZE_OVERRIDE
    if ps is None:
        ps = getattr(getattr(model, "config", object()), "patch_size", 16)
    rows, cols = H // ps, W // ps

    # Forward
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    hs = out.last_hidden_state.squeeze(0).detach().cpu().numpy()  # (T, D)

    T, D = hs.shape
    n_patches = rows * cols
    n_special = T - n_patches  # class + possible register tokens
    if n_special < 1:
        print("[error] Token shape mismatch. Check image size/padding and model.")
        print(f"T={T}, rows*cols={n_patches}, HxW={H}x{W}, ps={ps}")
        sys.exit(1)

    # Patch embeddings -> (rows, cols, D)
    patch_embs = hs[n_special:, :].reshape(rows, cols, D)

    # Precompute normalized matrix for fast cosine
    X = patch_embs.reshape(-1, D)                      # (N, D)
    X_norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / X_norms                                   # (N, D)

    # Figure
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img_disp, zorder=0)
    ax.set_axis_off()

    # Initialize overlay as RGBA via colormap
    init_scalar = 0.5 * np.ones((rows, cols), dtype=np.float32)
    cmap = plt.get_cmap("magma")
    rgba_init = cmap(init_scalar)                      # (rows, cols, 4) in [0,1]
    overlay_img = upsample_nearest(rgba_init, H, W, ps)
    overlay = ax.imshow(overlay_img, alpha=OVERLAY_ALPHA, zorder=1)

    # Grid / indices above overlay
    if SHOW_GRID:
        draw_grid(ax, rows, cols, ps)
    if ANNOTATE_INDICES:
        draw_indices(ax, rows, cols, ps)

    # Selection rectangle on top
    sel_rect = Rectangle((0, 0), ps, ps, fill=False, lw=2.0, ec="red", zorder=5)
    ax.add_patch(sel_rect)

    def rc_to_idx(r, c): return int(r) * cols + int(c)
    def idx_to_rc(i): return (i // cols, i % cols)

    current_idx = rc_to_idx(rows // 2, cols // 2)

    def update(idx):
        nonlocal current_idx
        idx = int(np.clip(idx, 0, rows * cols - 1))
        current_idx = idx
        r, c = idx_to_rc(idx)

        v = X[idx]
        vn = v / (np.linalg.norm(v) + 1e-8)

        cos = Xn @ vn                                # (N,) in [-1,1]
        cos_map = cos.reshape(rows, cols)

        # Normalize to [0,1] for coloring
        disp = (cos_map - cos_map.min()) / (cos_map.ptp() + 1e-8)

        # Build RGBA via colormap, then force selected cell to pure red
        rgba = cmap(disp)                            # (rows, cols, 4)
        rgba[r, c, 0:3] = np.array([1.0, 0.0, 0.0]) # RED cell
        rgba[r, c, 3]   = 1.0                       # full alpha for that cell

        # Upsample to pixel grid and update overlay
        rgba_up = upsample_nearest(rgba, H, W, ps)
        overlay.set_data(rgba_up)

        # Update selection rectangle
        sel_rect.set_xy((c * ps, r * ps))

        ax.set_title(
            f"Patch idx={idx} (r={r}, c={c}) • cos∈[{cos_map.min():.3f},{cos_map.max():.3f}]",
            fontsize=11
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        r = int(np.clip(y // ps, 0, rows - 1))
        c = int(np.clip(x // ps, 0, cols - 1))
        update(rc_to_idx(r, c))

    def on_key(event):
        r, c = idx_to_rc(current_idx)
        if event.key == "left":
            c = max(0, c - 1)
        elif event.key == "right":
            c = min(cols - 1, c + 1)
        elif event.key == "up":
            r = max(0, r - 1)
        elif event.key == "down":
            r = min(rows - 1, r + 1)
        elif event.key == "q":
            plt.close(fig); return
        else:
            return
        update(rc_to_idx(r, c))

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update(current_idx)
    print("[info] Controls: mouse click to select • arrow keys to move • 'q' to quit")
    plt.show()


if __name__ == "__main__":
    main()