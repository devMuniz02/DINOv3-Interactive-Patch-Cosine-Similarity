# filepath: DINOv3-PatchSimilarity/DINOv3CosSimilarity2Images.py
# one_or_two_image_patch_explorer.py
# Interactive DINOv3 patch similarity viewer (NO AutoImageProcessor, NO resize)
# - Single-image mode (0 or 1 image given): original behavior
# - Two-image mode (2 images given): when you click or move on one image,
#   shows BOTH overlays (self on source, cross on target).
#   NEW: Red selection rectangle is hidden on the non-active image.

import sys, math, io, urllib.request, argparse, os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoModel

# ---------- Defaults / knobs ----------
DEFAULT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

SHOW_GRID = True
ANNOTATE_INDICES = False
OVERLAY_ALPHA = 0.55
PATCH_SIZE_OVERRIDE = None   # set 16 to force; None = read from model if available

# ---------- Image I/O ----------
def load_image(path_or_url):
    if str(path_or_url).startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")

# ---------- Preprocessing (custom, no resize) ----------
def pad_to_multiple(pil_img, multiple=16):
    W, H = pil_img.size
    H_pad = int(math.ceil(H / multiple) * multiple)
    W_pad = int(math.ceil(W / multiple) * multiple)
    if (H_pad, W_pad) == (H, W):
        return pil_img, (0, 0, 0, 0)
    canvas = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))
    canvas.paste(pil_img, (0, 0))
    return canvas, (0, 0, W_pad - W, H_pad - H)

def preprocess_image_no_resize(pil_img, multiple=16):
    img_padded, pad_box = pad_to_multiple(pil_img, multiple=multiple)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    pixel_tensor = transform(img_padded).unsqueeze(0)  # (1,3,H,W)
    disp_np = np.array(img_padded, dtype=np.uint8)     # (H,W,3) for display
    return {"pixel_values": pixel_tensor}, disp_np, pad_box

# ---------- Utilities ----------
def upsample_nearest(arr, H, W, ps):
    if arr.ndim == 2:
        return arr.repeat(ps, 0).repeat(ps, 1)
    elif arr.ndim == 3:
        return arr.repeat(ps, 0).repeat(ps, 1)
    raise ValueError("upsample_nearest expects (rows,cols) or (rows,cols,channels)")

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

def rc_to_idx(r, c, cols): return int(r) * cols + int(c)
def idx_to_rc(i, cols):    return (int(i) // cols, int(i) % cols)

# ---------- Per-image embeddings ----------
class PatchImageState:
    def __init__(self, pil_img, model, device, ps):
        self.pil = pil_img
        self.ps = ps
        inputs, disp_np, _ = preprocess_image_no_resize(pil_img, multiple=ps)
        self.disp = disp_np
        self.pixel_values = inputs["pixel_values"].to(device)  # (1,3,H,W)
        _, _, self.H, self.W = self.pixel_values.shape
        self.rows, self.cols = self.H // ps, self.W // ps

        with torch.no_grad():
            out = model(pixel_values=self.pixel_values)
        hs = out.last_hidden_state.squeeze(0).detach().cpu().numpy()  # (T,D)

        T, D = hs.shape
        n_patches = self.rows * self.cols
        n_special = T - n_patches  # class + possible register tokens
        if n_special < 1:
            raise RuntimeError(
                f"[error] Token shape mismatch. T={T}, rows*cols={n_patches}, HxW={self.H}x{self.W}, ps={ps}"
            )

        self.D = D
        self.patch_embs = hs[n_special:, :].reshape(self.rows, self.cols, D)  # (rows,cols,D)
        self.X = self.patch_embs.reshape(-1, D)  # (N,D)
        self.Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-8)  # normalized

        # UI bits (set later by the viewers)
        self.ax = None
        self.overlay_im = None
        self.sel_rect = None
        self.best_rect = None

# ---------- Single-image mode ----------
def run_single_image(img_path, model, device, ps, show_grid, annotate_indices, overlay_alpha):
    img = load_image(img_path)
    st = PatchImageState(img, model, device, ps)

    fig, ax = plt.subplots(figsize=(9, 9))
    st.ax = ax
    ax.imshow(st.disp, zorder=0)
    ax.set_axis_off()
    if show_grid:
        draw_grid(ax, st.rows, st.cols, st.ps)
    if annotate_indices:
        draw_indices(ax, st.rows, st.cols, st.ps)

    # neutral overlay to start
    init_scalar = 0.5 * np.ones((st.rows, st.cols), dtype=np.float32)
    rgba = plt.get_cmap("magma")(init_scalar)
    rgba_up = upsample_nearest(rgba, st.H, st.W, st.ps)
    st.overlay_im = ax.imshow(rgba_up, alpha=overlay_alpha, zorder=1)

    st.sel_rect = Rectangle((0, 0), st.ps, st.ps, fill=False, lw=2.0, ec="red", zorder=5)
    ax.add_patch(st.sel_rect)

    current_idx = (st.rows // 2) * st.cols + st.cols // 2
    cmap = plt.get_cmap("magma")

    def update(idx):
        nonlocal current_idx
        current_idx = int(np.clip(idx, 0, st.rows * st.cols - 1))
        r, c = idx_to_rc(current_idx, st.cols)

        q = st.X[current_idx]
        qn = q / (np.linalg.norm(q) + 1e-8)
        cos = st.Xn @ qn
        cos_map = cos.reshape(st.rows, st.cols)

        disp = (cos_map - cos_map.min()) / (cos_map.ptp() + 1e-8)
        rgba = cmap(disp)
        # Force selected cell to pure RED (and full alpha in the RGBA array)
        rgba[r, c, 0:3] = np.array([1.0, 0.0, 0.0])
        rgba[r, c, 3]   = 1.0

        st.overlay_im.set_data(upsample_nearest(rgba, st.H, st.W, st.ps))
        st.overlay_im.set_alpha(overlay_alpha)  # global alpha

        st.sel_rect.set_xy((c * st.ps, r * st.ps))
        ax.set_title(
            f"Single-image • idx={current_idx} (r={r}, c={c}) • cos∈[{cos_map.min():.3f},{cos_map.max():.3f}]",
            fontsize=11
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        r = int(np.clip(event.ydata // st.ps, 0, st.rows - 1))
        c = int(np.clip(event.xdata // st.ps, 0, st.cols - 1))
        update(rc_to_idx(r, c, st.cols))

    def on_key(event):
        nonlocal current_idx
        r, c = idx_to_rc(current_idx, st.cols)
        if event.key == "left":
            c = max(0, c - 1)
        elif event.key == "right":
            c = min(st.cols - 1, c + 1)
        elif event.key == "up":
            r = max(0, r - 1)
        elif event.key == "down":
            r = min(st.rows - 1, r + 1)
        elif event.key == "q":
            plt.close(fig); return
        else:
            return
        update(rc_to_idx(r, c, st.cols))

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update(current_idx)
    print("[single-image] Controls: click to select • arrows to move • 'q' to quit")
    plt.tight_layout()
    plt.show()

# ---------- Two-image mode (shows BOTH overlays; hides red rect on non-active) ----------
def run_two_images(img1_path, img2_path, model, device, ps, show_grid, annotate_indices, overlay_alpha):
    img1, img2 = load_image(img1_path), load_image(img2_path)
    S = [PatchImageState(img1, model, device, ps),
         PatchImageState(img2, model, device, ps)]
    if S[0].D != S[1].D:
        raise RuntimeError("Embedding dims differ — use the same model for both images.")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6))
    axs = [axL, axR]
    for i, (ax, st) in enumerate(zip(axs, S)):
        st.ax = ax
        ax.imshow(st.disp, zorder=0)
        ax.set_axis_off()
        if show_grid:
            draw_grid(ax, st.rows, st.cols, st.ps)
        if annotate_indices:
            draw_indices(ax, st.rows, st.cols, st.ps)
        # start overlays (hidden until first render)
        init_scalar = 0.5 * np.ones((st.rows, st.cols), dtype=np.float32)
        rgba = plt.get_cmap("magma")(init_scalar)
        rgba_up = upsample_nearest(rgba, st.H, st.W, st.ps)
        st.overlay_im = ax.imshow(rgba_up, alpha=0.0, zorder=1)

        st.sel_rect  = Rectangle((0, 0), st.ps, st.ps, fill=False, lw=2.0, ec="red",    zorder=5)
        st.best_rect = Rectangle((0, 0), st.ps, st.ps, fill=False, lw=2.0, ec="yellow", zorder=6)
        ax.add_patch(st.sel_rect)
        ax.add_patch(st.best_rect)
        st.best_rect.set_visible(False)

    active_side = 0  # 0=left, 1=right
    current_idx = [ (S[0].rows//2)*S[0].cols + S[0].cols//2,
                    (S[1].rows//2)*S[1].cols + S[1].cols//2 ]
    cmap = plt.get_cmap("magma")

    def set_titles(src_i=None, self_stats=None, cross_stats=None):
        axs[0].set_title(f"LEFT  • {S[0].rows}x{S[0].cols} patches • {'ACTIVE' if active_side==0 else ''}", fontsize=10)
        axs[1].set_title(f"RIGHT • {S[1].rows}x{S[1].cols} patches • {'ACTIVE' if active_side==1 else ''}", fontsize=10)
        if src_i is not None and self_stats is not None and cross_stats is not None:
            src_name = "LEFT" if src_i == 0 else "RIGHT"
            tgt_name = "RIGHT" if src_i == 0 else "LEFT"
            fig.suptitle(
                f"Source: {src_name}  |  Self cos∈[{self_stats[0]:.3f},{self_stats[1]:.3f}]  •  "
                f"{tgt_name} cos∈[{cross_stats[0]:.3f},{cross_stats[1]:.3f}]  |  "
                f"Controls: click=select • arrows=move • '1'/'2'/'t'=switch side • 'q'=quit",
                fontsize=11
            )
        else:
            fig.suptitle(
                "Controls: click=select • arrows=move • '1'/'2'/'t'=switch side • 'q'=quit",
                fontsize=11
            )

    def clamp_idx(i, st):
        return int(np.clip(i, 0, st.rows*st.cols - 1))

    def update_selection_rects():
        # position rects
        for i, st in enumerate(S):
            r, c = idx_to_rc(current_idx[i], st.cols)
            st.sel_rect.set_xy((c * st.ps, r * st.ps))
        # visibility: only active side shows red rect
        for i, st in enumerate(S):
            st.sel_rect.set_visible(i == active_side)

    def compute_and_show_both_from_src(src_i):
        """Show self-similarity on src and cross-similarity on the other image."""
        src = S[src_i]
        tgt_i = 1 - src_i
        tgt = S[tgt_i]

        q_idx = clamp_idx(current_idx[src_i], src)
        q = src.X[q_idx]
        qn = q / (np.linalg.norm(q) + 1e-8)

        # --- Self on src ---
        cos_self = src.Xn @ qn
        cos_map_self = cos_self.reshape(src.rows, src.cols)
        disp_self = (cos_map_self - cos_map_self.min()) / (cos_map_self.ptp() + 1e-8)
        rgba_self = cmap(disp_self)
        r0, c0 = idx_to_rc(q_idx, src.cols)
        rgba_self[r0, c0, 0:3] = np.array([1.0, 0.0, 0.0])
        rgba_self[r0, c0, 3]   = 1.0
        src.overlay_im.set_data(upsample_nearest(rgba_self, src.H, src.W, src.ps))
        src.overlay_im.set_alpha(overlay_alpha)

        # --- Cross on tgt ---
        cos_cross = tgt.Xn @ qn
        cos_map_cross = cos_cross.reshape(tgt.rows, tgt.cols)
        disp_cross = (cos_map_cross - cos_map_cross.min()) / (cos_map_cross.ptp() + 1e-8)
        rgba_cross = cmap(disp_cross)
        tgt.overlay_im.set_data(upsample_nearest(rgba_cross, tgt.H, tgt.W, tgt.ps))
        tgt.overlay_im.set_alpha(overlay_alpha)

        # highlight best match on target
        best = int(np.argmax(cos_cross))
        br, bc = idx_to_rc(best, tgt.cols)
        tgt.best_rect.set_xy((bc * tgt.ps, br * tgt.ps))
        tgt.best_rect.set_visible(True)

        # Hide best on source (self best is the selected cell)
        S[1 - tgt_i].best_rect.set_visible(False)

        set_titles(src_i, (cos_map_self.min(), cos_map_self.max()),
                          (cos_map_cross.min(), cos_map_cross.max()))
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal active_side
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        side = 0 if event.inaxes is axs[0] else (1 if event.inaxes is axs[1] else None)
        if side is None: return
        st = S[side]
        r = int(np.clip(event.ydata // st.ps, 0, st.rows - 1))
        c = int(np.clip(event.xdata // st.ps, 0, st.cols - 1))
        current_idx[side] = rc_to_idx(r, c, st.cols)
        active_side = side
        update_selection_rects()        # <-- updates visibility
        compute_and_show_both_from_src(active_side)

    def on_key(event):
        nonlocal active_side
        if event.key == "q":
            plt.close(fig); return
        if event.key in ("t", "T"):
            active_side = 1 - active_side
            update_selection_rects()    # <-- updates visibility
            compute_and_show_both_from_src(active_side); return
        if event.key == "1":
            active_side = 0
            update_selection_rects()
            compute_and_show_both_from_src(active_side); return
        if event.key == "2":
            active_side = 1
            update_selection_rects()
            compute_and_show_both_from_src(active_side); return

        st = S[active_side]
        r, c = idx_to_rc(current_idx[active_side], st.cols)
        if event.key == "left":
            c = max(0, c - 1)
        elif event.key == "right":
            c = min(st.cols - 1, c + 1)
        elif event.key == "up":
            r = max(0, r - 1)
        elif event.key == "down":
            r = min(st.rows - 1, r + 1)
        else:
            return
        current_idx[active_side] = rc_to_idx(r, c, st.cols)
        update_selection_rects()        # keep visibility rule consistent
        compute_and_show_both_from_src(active_side)

    update_selection_rects()            # initialize positions + visibility
    set_titles()
    compute_and_show_both_from_src(active_side)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("[two-image BOTH] Controls:")
    print("  • Click on LEFT/RIGHT to select query patch (shows self + cross overlays)")
    print("  • Arrow keys move selection on ACTIVE side")
    print("  • '1'/'2'/'t' to switch side • 'q' to quit")
    plt.tight_layout()
    plt.show()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="DINOv3 Patch Similarity Viewer (1 or 2 images; two-image shows BOTH overlays)")
    # Accept either --image (single) or --image1/--image2 (two)
    parser.add_argument("--image", type=str, default=None,
                        help="Path/URL to image (single-image mode if only this is provided)")
    parser.add_argument("--image1", type=str, default=None,
                        help="Path/URL to first image (two-image mode if image2 is also provided)")
    parser.add_argument("--image2", type=str, default=None,
                        help="Path/URL to second image (two-image mode when given)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID,
                        help="DINOv3 model repo id (e.g., facebook/dinov3-vits16-pretrain-lvd1689m)")
    parser.add_argument("--show_grid", action="store_true", help="Draw patch grid")
    parser.add_argument("--annotate_indices", action="store_true", help="Write patch indices on cells")
    parser.add_argument("--overlay_alpha", type=float, default=OVERLAY_ALPHA, help="Heatmap alpha")
    parser.add_argument("--patch_size", type=int, default=(PATCH_SIZE_OVERRIDE or -1),
                        help="Override patch size. Set 16 to force. Default: model's patch size")
    args = parser.parse_args()

    show_grid = args.show_grid or SHOW_GRID
    annotate_indices = args.annotate_indices or ANNOTATE_INDICES
    overlay_alpha = args.overlay_alpha

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    ps = args.patch_size if args.patch_size and args.patch_size > 0 else getattr(getattr(model, "config", object()), "patch_size", 16)
    print(f"[info] Using patch size: {ps}")

    # Routing logic:
    img1 = args.image1 or args.image
    img2 = args.image2

    if img1 and img2:
        run_two_images(img1, img2, model, device, ps, show_grid, annotate_indices, overlay_alpha)
    else:
        img_single = img1 or DEFAULT_URL
        run_single_image(img_single, model, device, ps, show_grid, annotate_indices, overlay_alpha)

if __name__ == "__main__":
    main()
