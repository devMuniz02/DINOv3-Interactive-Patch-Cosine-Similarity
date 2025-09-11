import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA

# ---- Model loading (adapt to your setup) ----
from transformers import AutoModel

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trust_remote_code helps with facebook/dinov3 repos
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()

PATCH_SIZE = 16  # ViT-S/16

# ---- Normalization stats ----
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def preprocess(img_bgr):
    # Do NOT resize: keep original multiples of PATCH_SIZE for proper patch mapping
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)
    return tensor, pil_img

def extract_patches(model, tensor, ps):
    with torch.no_grad():
        out = model(pixel_values=tensor)
    # last_hidden_state: (1, T, D) where T = [CLS] + patches
    hs = out.last_hidden_state.squeeze(0).cpu().numpy()  # (T, D)

    H, W = tensor.shape[2], tensor.shape[3]
    rows, cols = H // ps, W // ps  # integer number of patches per dim
    n_patches = rows * cols
    T, D = hs.shape
    n_special = T - n_patches  # typically 1 for [CLS]
    if n_special < 0:
        # Defensive: if sizes are off, bail gracefully
        n_special = 1
    patches = hs[n_special:, :]  # drop special tokens
    # Safety in case of mismatch due to rounding
    patches = patches[:n_patches, :]
    patches = patches.reshape(rows, cols, D)

    X = patches.reshape(-1, D)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn, rows, cols

# ---- Camera loop ----
selected_patch_vec = None
last_selected_frame = None
r_sel, c_sel = None, None  # initialize explicitly

def mouse_callback(event, x, y, flags, param):
    global selected_patch_vec, last_selected_frame, r_sel, c_sel
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param.get('scale', 1.0)
        x_unscaled = int(x / max(scale, 1e-8))
        y_unscaled = int(y / max(scale, 1e-8))

        cols = param['cols']
        rows = param['Xn'].shape[0] // cols

        # Click must be inside the patch grid area
        if x_unscaled < cols * PATCH_SIZE and y_unscaled < rows * PATCH_SIZE:
            r = y_unscaled // PATCH_SIZE
            c = x_unscaled // PATCH_SIZE
            idx = r * cols + c
            if 0 <= idx < param['Xn'].shape[0]:
                # Keep a normalized copy
                v = param['Xn'][idx].copy()
                v /= (np.linalg.norm(v) + 1e-8)
                selected_patch_vec = v
                last_selected_frame = param['frame'].copy()
                r_sel, c_sel = r, c

cap = cv2.VideoCapture(0)
win_name = "Patch Cosine Similarity (3x2 Grid)"
cv2.namedWindow(win_name)

# Try to get initial window size for scaling
screen_w, screen_h = 1280, 720
try:
    rect = cv2.getWindowImageRect(win_name)
    screen_w, screen_h = rect[2], rect[3]
except:
    pass

pca_model = None
first_frame_pca_fitted = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Ensure frame dims are multiples of PATCH_SIZE so grid aligns perfectly ---
    H0, W0 = frame.shape[:2]
    Hc = (H0 // PATCH_SIZE) * PATCH_SIZE
    Wc = (W0 // PATCH_SIZE) * PATCH_SIZE
    if Hc == 0 or Wc == 0:
        continue  # skip weird frames
    frame_proc = frame[:Hc, :Wc]  # crop to multiples of PATCH_SIZE

    # --- Features ---
    tensor, _ = preprocess(frame_proc)
    Xn, rows, cols = extract_patches(model, tensor, PATCH_SIZE)

    # --- Cosine overlay ---
    overlay_color = np.zeros_like(frame_proc)
    if selected_patch_vec is not None:
        qn = selected_patch_vec
        cos_map = Xn @ qn  # (rows*cols,)
        # Normalize to [0,1] for visualization
        cmin, cmax = 0.0, 1.0
        cos_map = (cos_map - cmin) / (cmax - cmin + 1e-8)
        cos_map_img = cos_map.reshape(rows, cols)
        overlay_small = cv2.resize(
            cos_map_img.astype(np.float32),
            (cols * PATCH_SIZE, rows * PATCH_SIZE),
            interpolation=cv2.INTER_NEAREST
        )
        overlay_color = cv2.applyColorMap((overlay_small * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

        # Blend (same size as frame_proc)
        blended = cv2.addWeighted(frame_proc, 0.5, overlay_color, 0.5, 0.0)
    else:
        blended = frame_proc

    # --- Last selected view with grid and red rectangle ---
    if last_selected_frame is not None and r_sel is not None and c_sel is not None:
        last_grid = last_selected_frame.copy()
        hl, wl = last_grid.shape[:2]
        rows_last = hl // PATCH_SIZE
        cols_last = wl // PATCH_SIZE
        # draw grid
        for rr in range(rows_last):
            y = rr * PATCH_SIZE
            cv2.line(last_grid, (0, y), (wl, y), (200, 200, 200), 1)
        for cc in range(cols_last):
            x = cc * PATCH_SIZE
            cv2.line(last_grid, (x, 0), (x, hl), (200, 200, 200), 1)
        # selected patch
        x0 = c_sel * PATCH_SIZE
        y0 = r_sel * PATCH_SIZE
        x1 = x0 + PATCH_SIZE
        y1 = y0 + PATCH_SIZE
        cv2.rectangle(last_grid, (x0, y0), (x1, y1), (0, 0, 255), 2)
        # match current size
        last_view = cv2.resize(last_grid, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
    else:
        last_view = np.zeros_like(frame_proc)

    # --- PCA Visualization ---
    if not first_frame_pca_fitted and Xn.shape[1] >= 3:
        pca_model = PCA(n_components=3)
        pca_model.fit(Xn)
        first_frame_pca_fitted = True

    # If a patch is selected, refit PCA on features of the frame at selection
    if selected_patch_vec is not None and last_selected_frame is not None:
        tensor_sel, _ = preprocess(last_selected_frame[:Hc, :Wc])  # safe crop if needed
        Xn_sel, _, _ = extract_patches(model, tensor_sel, PATCH_SIZE)
        if Xn_sel.shape[1] >= 3:
            pca_model = PCA(n_components=3)
            pca_model.fit(Xn_sel)

    if pca_model is not None and Xn.shape[1] >= 3 and Xn.shape[0] == rows * cols:
        pca_feats = pca_model.transform(Xn)
        lo = np.percentile(pca_feats, 1, axis=0, keepdims=True)
        hi = np.percentile(pca_feats, 99, axis=0, keepdims=True)
        pca_feats = (pca_feats - lo) / (hi - lo + 1e-8)
        pca_feats = np.clip(pca_feats, 0, 1)
        pca_small = (pca_feats.reshape(rows, cols, 3) * 255).astype(np.uint8)
        pca_img = cv2.resize(pca_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
    else:
        pca_img = np.zeros_like(frame_proc)

    # --- Titles ---
    def add_title(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        return out

    frame_title   = add_title(frame_proc, "Original")
    blended_title = add_title(blended, "Original + Cosine Overlay")
    overlay_title = add_title(overlay_color, "Cosine Overlay Only")
    last_title    = add_title(last_view, "Last Selection (Grid + Red)")
    pca_title     = add_title(pca_img, "PCA Visualization")
    black_title   = add_title(np.zeros_like(frame_proc), "Empty")

    # --- 3x2 grid ---
    row1 = np.concatenate([frame_title, blended_title], axis=1)
    row2 = np.concatenate([overlay_title, last_title], axis=1)
    row3 = np.concatenate([pca_title, black_title], axis=1)
    grid = np.concatenate([row1, row2, row3], axis=0)

    # --- Fit to window once grid exists (fix: do NOT touch 'grid' before this) ---
    gh, gw = grid.shape[:2]
    scale = min(screen_w / gw, screen_h / gh, 1.0)
    if scale < 1.0 or (gw > screen_w or gh > screen_h):
        grid = cv2.resize(grid, (int(gw * scale), int(gh * scale)), interpolation=cv2.INTER_AREA)

    # Register mouse callback ONCE per frame, after 'scale' is known
    cv2.setMouseCallback(
        win_name,
        mouse_callback,
        param={
            'cols': cols,
            'Xn': Xn,
            'frame': frame_proc,           # use the cropped/processed frame
            'scale': scale,
        }
    )

    cv2.imshow(win_name, grid)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
