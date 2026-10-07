# =====================================================================
#  RIPON COMPUTER - Photo Studio (Passport / Visa / Stamp + AI Retouch)
#  Install:  pip install PySide6 opencv-python numpy pillow rembg onnxruntime
#  Optional AI (GPU/torch lagbe, na thakleo software cholbe):
#            pip install gfpgan realesrgan
#  Dress/Suit PNG: ei file er pashe "dress" naam er folder banao, tate
#  transparent PNG rakho - Studio menu te auto ashbe.
# =====================================================================
import sys, os, math, io, datetime
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageFont
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                               QComboBox, QFileDialog, QMessageBox, QSpinBox, QCheckBox, QSlider,
                               QGroupBox, QScrollArea, QStackedWidget, QColorDialog, QMenuBar, QLineEdit,
                               QProgressBar, QFrame)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QPainterPath, QPolygonF,
                           QShortcut, QKeySequence, QAction)
from PySide6.QtCore import Qt, QRectF, QPointF, QObject, Signal, Slot, QThread, QTimer

# ---- Dokan er size ekhane change korte paro (mm) ----
SIZES = {
    "Passport 35x45 mm": (35, 45),
    "Visa 40x50 mm": (40, 50),
    "Stamp 25x30 mm": (25, 30),
    "Small 20x25 mm": (20, 25),
}
BG_COLORS = {"White": (255, 255, 255), "Light Blue": (173, 216, 230), "Blue": (70, 130, 200)}
DPI = 300
A4 = (2480, 3508)
APP_NAME = "Ripon Computer"
PAPERS = {"A4 (210x297 mm)": (2480, 3508), "A5 (148x210 mm)": (1748, 2480),
          "4R (4x6 inch)": (1200, 1800), "5R (5x7 inch)": (1500, 2100)}
HAIR_COLORS = {"Black (Kalo)": (28, 24, 24), "Dark Brown": (62, 42, 32), "Brown": (110, 72, 44),
               "Golden Blonde": (190, 150, 84), "Grey": (140, 140, 142), "Silver White": (205, 205, 208),
               "Henna Red": (140, 52, 34)}
RT_TOOLS = ["heal", "hair_add", "hair_erase"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRESS_DIR = os.path.join(SCRIPT_DIR, "dress")

def mm2px(mm): return int(round(mm / 25.4 * DPI))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
    if hasattr(cv2, "CascadeClassifier") else None
MODELS = {
    "u2net_human_seg (fast, halka)": "u2net_human_seg",
    "birefnet-portrait (chul e best)": "birefnet-portrait",
    "bria-rmbg (RMBG-2.0)": "bria-rmbg",
}
_sessions = {}

def remove_bg(pil_img, model="u2net_human_seg"):
    from rembg import remove, new_session
    if model not in _sessions:
        _sessions[model] = new_session(model)  # prothom bar model download hobe
    return remove(pil_img, session=_sessions[model]).convert("RGBA")

# ------------------- AUTO FEATURES -------------------
def auto_enhance(rgb):
    f = rgb.astype(np.float32)
    avg = f.reshape(-1, 3).mean(0)
    scale = np.clip(avg.mean() / np.maximum(avg, 1), 0.85, 1.15)
    rgb = np.clip(f * scale, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.5)
    return cv2.addWeighted(rgb, 1.4, blur, -0.4, 0)

def skin_mask(rgb, face):
    x, y, w, h = face
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    m = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    region = np.zeros(m.shape, np.uint8)
    x0, x1 = max(0, int(x - 0.2 * w)), min(m.shape[1], int(x + 1.2 * w))
    y0, y1 = max(0, int(y - 0.3 * h)), min(m.shape[0], int(y + 1.4 * h))
    region[y0:y1, x0:x1] = 255
    m = cv2.bitwise_and(m, region)
    k = max(3, (w // 15) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.GaussianBlur(m, (k * 2 + 1, k * 2 + 1), 0)
    return (m.astype(np.float32) / 255.0)[..., None]

def skin_smooth(rgb, mask, face, strength):
    d = max(5, face[2] // 20)
    sm = cv2.bilateralFilter(rgb, d, 40, d * 2)
    sm = cv2.bilateralFilter(sm, d, 40, d * 2)
    amt = mask * (strength / 100.0) * 0.85
    return np.clip(rgb * (1 - amt) + sm * amt, 0, 255).astype(np.uint8)

def skin_tone_up(rgb, mask, strength):
    s = strength / 100.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = lab[..., 0:1], lab[..., 1:2], lab[..., 2:3]
    sel = mask[..., 0] > 0.5
    if sel.sum() > 100:
        ma, mb = a[sel].mean(), b[sel].mean()
        a = a + (ma - a) * mask * 0.35 * s
        b = b + (mb - b) * mask * 0.35 * s
    l = l + 10 * s * mask
    lab = np.concatenate([l, a, b], axis=2)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

# ------------------- RIPON: AI / RETOUCH / STUDIO TOOLS -------------------
_ai = {}

def _torch_device():
    try:
        import torch
        return ("cuda" if torch.cuda.is_available() else "cpu"), torch
    except Exception:
        return "cpu", None

def _gfpgan():
    if "gfpgan" not in _ai:
        try:
            from gfpgan import GFPGANer
            device, _ = _torch_device()
            _ai["gfpgan"] = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
                device=device)
            _ai["gfpgan_device"] = device
        except TypeError:
            try:
                from gfpgan import GFPGANer
                _ai["gfpgan"] = GFPGANer(
                    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                    upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
                _ai["gfpgan_device"] = _torch_device()[0]
            except Exception as e:
                _ai["gfpgan"] = None; _ai["gfpgan_err"] = str(e)
        except Exception as e:
            _ai["gfpgan"] = None; _ai["gfpgan_err"] = str(e)
    return _ai["gfpgan"]

def ai_face_restore(rgb, weight=0.5):
    g = _gfpgan()
    if g is None:
        raise RuntimeError("GFPGAN install nai ba load hoyni.\nCMD e chalao:  pip install gfpgan\n" + _ai.get("gfpgan_err", "")[:150])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, _, out = g.enhance(bgr, has_aligned=False, only_center_face=True, paste_back=True, weight=weight)
    if out is None: raise RuntimeError("Face paoa jayni")
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def _esrgan():
    if "esr" not in _ai:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            device, torch = _torch_device()
            m = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            _ai["esr"] = RealESRGANer(
                scale=4, model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model=m, tile=400, tile_pad=10, pre_pad=0,
                half=bool(device == "cuda"), gpu_id=0 if device == "cuda" else None)
            _ai["esr_device"] = device
        except Exception as e:
            _ai["esr"] = None
            _ai["esr_err"] = str(e)
    return _ai["esr"]

def upscale2x(pil_img):
    """AI (Real-ESRGAN) thakle seta, na thakle Lanczos + sharpen"""
    esr = _esrgan()
    if esr is not None:
        try:
            out, _ = esr.enhance(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), outscale=2)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), "AI (Real-ESRGAN)"
        except Exception:
            pass
    big = pil_img.resize((pil_img.width * 2, pil_img.height * 2), Image.LANCZOS)
    a = np.array(big)
    a = cv2.addWeighted(a, 1.5, cv2.GaussianBlur(a, (0, 0), 1.6), -0.5, 0)
    return Image.fromarray(a), "Normal (Lanczos + sharpen)"

def enhance_pro(rgb, strength=60):
    """noise kombay + color/contrast + shadow uthay + vibrance + sharpen"""
    s = strength / 100.0
    x = cv2.bilateralFilter(rgb, 5, 10 + 20 * s, 5)
    x = auto_enhance(x)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32) / 255.0
    lab[..., 0] = np.clip((l ** (1.0 - 0.2 * s)) * 255, 0, 255).astype(np.uint8)
    x = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.3 * s * (1 - sat)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def skin_retouch(rgb, mask, face, strength):
    """Frequency-separation style: dag/rong smooth hoy kintu pore texture thake (natural dekhay)"""
    s = strength / 100.0
    w = max(face[2], 80)
    d = max(5, int(w // 25) | 1)
    base = cv2.bilateralFilter(rgb, d, 20 + 40 * s, d * 2)
    base = cv2.bilateralFilter(base, d, 20 + 40 * s, d * 2).astype(np.float32)
    f = rgb.astype(np.float32)
    tex = f - cv2.GaussianBlur(f, (0, 0), max(1.0, w / 70.0))
    tex = cv2.GaussianBlur(tex, (0, 0), 0.7)
    out = base + tex * (1.0 - 0.6 * s)
    amt = mask * (0.45 + 0.5 * s)
    return np.clip(f * (1 - amt) + out * amt, 0, 255).astype(np.uint8)

def skin_shine_fix(rgb, mask, strength):
    """mukher oily/chokchoke bhab kombay"""
    s = strength / 100.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[..., 0]; sel = mask[..., 0] > 0.5
    if sel.sum() < 100: return rgb
    th = np.percentile(l[sel], 75)
    lab[..., 0] = l - np.clip(l - th, 0, None) * 0.75 * s * mask[..., 0]
    conv = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    return np.clip(rgb.astype(np.float32) * (1 - mask) + conv * mask, 0, 255).astype(np.uint8)

def heal_spot(arr, cx, cy, r):
    """Spot Healing: pashe er porishkar skin theke patch niye dag dhake (arr inplace bodlay).
    return (x0, y0, old_patch) undo er jonno, na hole None"""
    H, W = arr.shape[:2]
    r = int(max(3, r)); pad = int(r * 1.5) + 1; S = 2 * pad
    cx, cy = int(round(cx)), int(round(cy))
    def grab(px, py):
        x0, y0 = int(round(px)) - pad, int(round(py)) - pad
        if x0 < 0 or y0 < 0 or x0 + S > W or y0 + S > H: return None
        return x0, y0
    t = grab(cx, cy)
    if t is None: return None
    x0, y0 = t
    tgt = arr[y0:y0 + S, x0:x0 + S]
    yy, xx = np.mgrid[0:S, 0:S]
    dist = np.hypot(xx - pad + 0.5, yy - pad + 0.5)
    ring = (dist > r * 1.1) & (dist < r * 1.5)
    inner = dist < r
    ref = tgt[ring].astype(np.float32).mean(0)
    best, best_sc = None, 1e18
    for ang in range(0, 360, 30):
        for mult in (2.6, 3.6):
            g = grab(cx + math.cos(math.radians(ang)) * r * mult, cy + math.sin(math.radians(ang)) * r * mult)
            if g is None: continue
            p = arr[g[1]:g[1] + S, g[0]:g[0] + S]
            sc = np.abs(p[ring].astype(np.float32).mean(0) - ref).sum() + 0.6 * p[inner].astype(np.float32).std(0).sum()
            if sc < best_sc: best, best_sc = p, sc
    if best is None: return None
    src = best.astype(np.float32)
    src += ref - src[ring].mean(0)                       # tone match
    m = np.clip((r - dist) / (0.45 * r), 0, 1)[..., None]
    old = tgt.copy()
    arr[y0:y0 + S, x0:x0 + S] = np.clip(tgt.astype(np.float32) * (1 - m) + src * m, 0, 255).astype(np.uint8)
    return x0, y0, old

def find_blemishes(rgb, face, sens):
    """auto pimple/dag khuje. return [(cx, cy, radius), ...]"""
    H, W = rgb.shape[:2]; x, y, w, h = face
    x0, y0 = max(0, int(x - 0.1 * w)), max(0, int(y - 0.3 * h))
    x1, y1 = min(W, int(x + 1.1 * w)), min(H, int(y + 1.1 * h))
    if x1 - x0 < 20 or y1 - y0 < 20: return []
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2LAB)
    med = cv2.medianBlur(lab, max(9, int(w // 7) | 1))
    d = lab.astype(np.int16) - med.astype(np.int16)
    t = 1.0 - sens / 100.0
    spot = (d[..., 1] > 5 + 9 * t) | (d[..., 0] < -(9 + 15 * t))
    sk = (skin_mask(rgb, face)[y0:y1, x0:x1, 0] > 0.5).astype(np.uint8)
    e = max(3, int(w * 0.05) | 1)
    spot &= cv2.erode(sk, np.ones((e, e), np.uint8)) > 0          # mukher kinara (jaw/chul er line) e chhuo na
    ex = np.zeros(spot.shape, bool)
    def zone(a, b, c, e):          # chokh/bhru + nak/thot e chhuo na
        ex[int(y + b * h) - y0:int(y + e * h) - y0, int(x + a * w) - x0:int(x + c * w) - x0] = True
    zone(0.0, 0.20, 1.0, 0.48); zone(0.35, 0.48, 0.65, 0.72); zone(0.30, 0.72, 0.70, 0.88)
    spot &= ~ex
    m8 = cv2.morphologyEx(spot.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(m8, connectivity=8)
    amin, amax = (w * 0.008) ** 2, (w * 0.09) ** 2
    out = []
    for i in range(1, n):
        bw, bh, ar = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        if ar < amin or ar > amax or max(bw, bh) > 2.5 * min(bw, bh) + 2: continue
        out.append((x0 + ce[i][0], y0 + ce[i][1], max(bw, bh) + 3))
    return out[:150]

def remove_blemishes(rgb, face, sens):
    arr = rgb.copy(); spots = find_blemishes(arr, face, sens)
    for cx, cy, r in spots: heal_spot(arr, cx, cy, r)
    return arr, len(spots)

def hair_mask_auto(rgb, face, person=None):
    """chuler jaiga (float 0-1). person = bg-remove er alpha (0-1) thakle bhalo; na thakle border color theke ondaj.
    skin chena hoy mukher majher rong er sathe milie (tai kalo/brown chul skin hishebe dhora porbe na)"""
    H, W = rgb.shape[:2]; x, y, w, h = face
    rx0, rx1 = max(0, int(x - 0.5 * w)), min(W, int(x + 1.5 * w))
    ry0, ry1 = max(0, int(y - 1.0 * h)), min(H, int(y + 0.85 * h))
    if rx1 - rx0 < 10 or ry1 - ry0 < 10: return np.zeros((H, W), np.float32)
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[ry0:ry1, rx0:rx1]), cv2.COLOR_RGB2LAB).astype(np.float32)
    # mukher majher (gal/nak) skin color
    sy0, sy1 = max(0, int(y + 0.52 * h)), min(H, int(y + 0.66 * h))
    sx0, sx1 = max(0, int(x + 0.25 * w)), min(W, int(x + 0.75 * w))
    sk = cv2.cvtColor(np.ascontiguousarray(rgb[sy0:sy1, sx0:sx1]), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    med = np.median(sk, 0) if len(sk) else np.array([180, 140, 145], np.float32)
    skin = np.sqrt((lab[..., 0] - med[0]) ** 2 + (2 * (lab[..., 1] - med[1])) ** 2 + (2 * (lab[..., 2] - med[2])) ** 2) < 45
    if person is None:
        a, b = max(2, H // 30), max(2, W // 30)
        strips = [rgb[:a], rgb[:, :b], rgb[:, W - b:]]
        border = np.concatenate([cv2.cvtColor(np.ascontiguousarray(s_), cv2.COLOR_RGB2LAB).reshape(-1, 3) for s_ in strips]).astype(np.float32)
        person_r = np.linalg.norm(lab - np.median(border, 0), axis=2) > 28
    else:
        person_r = person[ry0:ry1, rx0:rx1] > 0.5
    mr = (person_r & ~skin).astype(np.float32)
    m = np.zeros((H, W), np.float32); m[ry0:ry1, rx0:rx1] = mr
    m[max(0, int(y + 0.04 * h)):min(H, int(y + 1.05 * h)), max(0, int(x + 0.03 * w)):min(W, int(x + 0.97 * w))] = 0   # mukh er bhitor baad
    m8 = (m * 255).astype(np.uint8)
    k = max(3, int(w // 40) | 1)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((k * 2 + 1, k * 2 + 1), np.uint8))
    n, labs, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    keep = np.zeros_like(m8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 0.02 * w * h: keep[labs == i] = 255
    keep = cv2.GaussianBlur(keep, (0, 0), max(1.0, w / 120))
    return keep.astype(np.float32) / 255.0

def apply_hair_color(rgb, hm, color, strength=100):
    """chuler shading/texture rekhe rong bodlay. color = (r, g, b)"""
    s = strength / 100.0
    sel = hm > 0.5
    if sel.sum() < 50: return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tl = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    l = lab[..., 0]; ml = l[sel].mean()
    new = lab.copy()
    new[..., 0] = tl[0] + (l - ml) * 0.65
    new[..., 1] = tl[1] + (lab[..., 1] - 128) * 0.15
    new[..., 2] = tl[2] + (lab[..., 2] - 128) * 0.15
    conv = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    m = (np.clip(hm, 0, 1) * s)[..., None]
    return np.clip(rgb.astype(np.float32) * (1 - m) + conv * m, 0, 255).astype(np.uint8)

def add_dress(im, dress, scale, yoff):
    """dress = transparent PNG (RGBA). scale = photo width er %, yoff = upor/niche (% of height)"""
    W, H = im.size
    dw = max(1, int(W * scale / 100.0)); dh = max(1, int(dress.height * dw / dress.width))
    d = dress.resize((dw, dh), Image.LANCZOS)
    out = im.convert("RGBA")
    out.paste(d, ((W - dw) // 2, H - dh + int(H * yoff / 100.0)), d)
    return out.convert("RGB")

def add_caption(im, text):
    if not text.strip(): return im
    im = im.copy(); d = ImageDraw.Draw(im); W, H = im.size
    size = max(10, int(H * 0.055))
    def font_of(sz):
        for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
            try: return ImageFont.truetype(name, sz)
            except Exception: pass
        return ImageFont.load_default()
    font = font_of(size)
    try:
        while d.textlength(text, font=font) > W * 0.94 and size > 8:
            size -= 1; font = font_of(size)
    except Exception: pass
    bar = int(size * 1.8)
    d.rectangle([0, H - bar, W, H], fill=(255, 255, 255))
    try: d.text((W / 2, H - bar / 2), text, fill=(0, 0, 0), font=font, anchor="mm")
    except Exception: d.text((6, H - bar + 4), text, fill=(0, 0, 0), font=font)
    return im

def pack_sheet(items, paper, gap_mm=3, margin_mm=4, border=True):
    """items = [(photo, copies), ...]. Portrait/landscape duto try kore, jeta te beshi dhoke setai nei.
    return (sheet, placed, total)"""
    gap, mg = mm2px(gap_mm), mm2px(margin_mm)
    total = sum(c for _, c in items); best = None
    for PW, PH in (paper, (paper[1], paper[0])):
        sheet = Image.new("RGB", (PW, PH), (255, 255, 255)); d = ImageDraw.Draw(sheet)
        y = mg; placed = 0; full = False
        for photo, copies in items:
            pw, ph = photo.size; n = 0
            while n < copies and not full:
                if y + ph > PH - mg: full = True; break
                x = mg; row_n = 0
                while n < copies and x + pw <= PW - mg:
                    sheet.paste(photo, (x, y))
                    if border: d.rectangle([x, y, x + pw - 1, y + ph - 1], outline=(200, 200, 200))
                    x += pw + gap; n += 1; placed += 1; row_n += 1
                if row_n == 0: full = True; break
                y += ph + gap
        if best is None or placed > best[1]: best = (sheet, placed)
        if placed == total: break
    return best[0], best[1], total

def save_online(im, path, kb, size_px=None):
    """online form er jonno: KB limit er moddhe JPG save. size_px=(w, h) dile oi pixel e (center crop). return (bytes, (w, h)) ba None"""
    fixed = bool(size_px)
    if fixed: im = ImageOps.fit(im, size_px, Image.LANCZOS)
    limit = kb * 1024; scale = 1.0
    for _ in range(1 if fixed else 8):
        cur = im if scale == 1.0 else im.resize((max(50, int(im.width * scale)), max(50, int(im.height * scale))), Image.LANCZOS)
        lo, hi, best = 15, 95, None
        while lo <= hi:
            q = (lo + hi) // 2; buf = io.BytesIO()
            cur.save(buf, "JPEG", quality=q, optimize=True, dpi=(DPI, DPI))
            if buf.tell() <= limit: best, lo = buf.getvalue(), q + 1
            else: hi = q - 1
        if best:
            with open(path, "wb") as fh: fh.write(best)
            return len(best), cur.size
        scale *= 0.88
    return None

# ------------------- STATE HELPERS -------------------
def detect_face(rgb):
    if face_cascade is None: return None
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        s = min(1.0, 1200.0 / max(gray.shape))
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        for sf, nb, ms in ((1.1, 5, 60), (1.05, 3, 40)):
            faces = face_cascade.detectMultiScale(g, sf, nb, minSize=(ms, ms))
            if len(faces):
                f = max(faces, key=lambda f: f[2] * f[3])
                return tuple(int(v / s) for v in f)
    except Exception:
        pass
    return None

def auto_box(cut, face, ratio, use_face=True):
    """box = [center_x, center_y, height] (image pixel)"""
    iw, ih = cut.size
    if face is not None and use_face:
        x, y, w, h = face
        return [x + w / 2, y + h / 2 + h * 0.05, h / 0.55, 0.0]
    return [iw / 2, ih / 2, min(ih, iw / ratio), 0.0]

def render_preview(small, bg, angle):
    im = small.rotate(angle, resample=Image.BICUBIC) if angle else small
    canvas = Image.new("RGB", im.size, bg)
    canvas.paste(im, (0, 0), im)
    return canvas

def crop_to(cut, bg, size_mm, box):
    """Rotated box theke final photo. box = [cx, cy, height, angle_deg]"""
    cx, cy, bh, ang = box
    W, H = size_mm
    pw, ph = mm2px(W), mm2px(H)
    RW, RH = pw * 2, ph * 2                      # 2x e banai, tarpor chhoto kori (smooth)
    k = bh / RH
    if k > 1.0:                                  # boro photo age chhoto kore nei (aliasing eray)
        f = 1.0 / k
        cut = cut.resize((max(1, int(cut.width * f)), max(1, int(cut.height * f))), Image.LANCZOS)
        cx, cy, bh = cx * f, cy * f, bh * f
        k = bh / RH
    t = math.radians(ang); c, s_ = math.cos(t), math.sin(t)
    data = (c * k, -s_ * k, cx - c * k * RW / 2 + s_ * k * RH / 2,
            s_ * k, c * k, cy - s_ * k * RW / 2 - c * k * RH / 2)
    res = cut.transform((RW, RH), Image.AFFINE, data, resample=Image.BICUBIC)
    canvas = Image.new("RGB", (RW, RH), bg)
    canvas.paste(res, (0, 0), res)
    return canvas.resize((pw, ph), Image.LANCZOS)

def make_a4(photo, copies):
    sheet = Image.new("RGB", A4, (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    pw, ph = photo.size
    gap, margin = mm2px(3), mm2px(8)
    cols = max(1, (A4[0] - 2 * margin + gap) // (pw + gap))
    n, y = 0, margin
    while n < copies and y + ph <= A4[1] - margin:
        for c in range(cols):
            if n >= copies: break
            x = margin + c * (pw + gap)
            sheet.paste(photo, (x, y))
            d.rectangle([x, y, x + pw, y + ph], outline=(200, 200, 200))
            n += 1
        y += ph + gap
    return sheet


# ------------------- V2 PERFORMANCE ENGINE -------------------
# All long-running AI/CPU jobs live in one persistent worker thread. The UI thread
# is reserved for Qt painting/input, so sliders, crop, menus and preview stay responsive.

def run_auto_pipeline(arr, face, settings, model_name, hair_rgb):
    """Pure processing pipeline used by the worker. No Qt/UI access here."""
    notes = []
    arr = np.ascontiguousarray(arr)
    alpha = None
    if settings.get("enh"):
        arr = enhance_pro(arr, settings.get("enh_strength", 60))
    if settings.get("face_ai"):
        try:
            arr = ai_face_restore(arr, settings.get("face_strength", 50) / 100.0)
        except Exception as e:
            notes.append("AI Face skip: " + (str(e).splitlines() or [""])[0][:90])
    skin_on = settings.get("smooth") or settings.get("tone") or settings.get("shine")
    if face is not None:
        mask = skin_mask(arr, face) if skin_on else None
        if settings.get("pimple"):
            arr, _ = remove_blemishes(arr, face, settings.get("pimple_strength", 55))
        if settings.get("shine"):
            if mask is None: mask = skin_mask(arr, face)
            arr = skin_shine_fix(arr, mask, 60)
        if settings.get("smooth"):
            if mask is None: mask = skin_mask(arr, face)
            arr = skin_retouch(arr, mask, face, settings.get("smooth_strength", 50))
        if settings.get("tone"):
            if mask is None: mask = skin_mask(arr, face)
            arr = skin_tone_up(arr, mask, settings.get("tone_strength", 50))
    elif skin_on or settings.get("pimple"):
        notes.append("Face paini, skin/pimple skip")
    person = None
    if settings.get("bg") or settings.get("hair"):
        try:
            person = remove_bg(Image.fromarray(arr), model_name).getchannel("A")
        except Exception as e:
            if settings.get("bg"):
                raise
            notes.append("Hair segmentation skip: " + (str(e).splitlines() or [""])[0][:90])
    hair_mask = None
    if settings.get("hair"):
        if face is None:
            notes.append("Face paini, hair skip")
        else:
            person_np = None if person is None else np.array(person, np.float32) / 255.0
            hair_mask = hair_mask_auto(arr, face, person_np)
            arr = apply_hair_color(arr, hair_mask, hair_rgb, settings.get("hair_strength", 100))
    if settings.get("bg"):
        alpha = person
    return arr, alpha, hair_mask, notes

class AIWorker(QObject):
    request = Signal(str, object)
    finished = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(str)

    def __init__(self):
        super().__init__()
        self.request.connect(self.execute)

    @Slot(str, object)
    def execute(self, kind, payload):
        try:
            if kind == "preload":
                self.progress.emit("AI startup: preparing models...")
                # Load the default/selected segmentation model first because it is
                # the most commonly used shop operation.
                model = payload.get("model", "u2net_human_seg")
                try:
                    remove_bg(Image.new("RGB", (32, 32), (255, 255, 255)), model)
                except Exception:
                    # Optional AI must never prevent the application from opening.
                    pass
                self.progress.emit("AI startup: loading face restore...")
                try: _gfpgan()
                except Exception: pass
                self.progress.emit("AI startup: loading upscale...")
                try: _esrgan()
                except Exception: pass
                self.finished.emit(kind, {"ready": True})
                return

            if kind == "bg":
                self.progress.emit("Background remove processing...")
                out = remove_bg(payload["image"], payload["model"]).getchannel("A")
                self.finished.emit(kind, out)
                return

            if kind == "face":
                self.progress.emit("AI Face Restore processing...")
                out = ai_face_restore(payload["arr"], payload["weight"])
                self.finished.emit(kind, out)
                return

            if kind == "upscale":
                self.progress.emit("AI Upscale processing...")
                out, how = upscale2x(payload["image"])
                self.finished.emit(kind, (out, how))
                return

            if kind == "auto":
                self.progress.emit("Auto processing...")
                result = run_auto_pipeline(payload["arr"], payload["face"], payload["settings"], payload["model"], payload["hair_rgb"])
                self.finished.emit(kind, result)
                return

            if kind == "hairmask":
                self.progress.emit("Chuler jaiga khujchi...")
                person = None
                if payload.get("alpha") is not None:
                    person = payload["alpha"]
                else:
                    try:
                        person = np.array(remove_bg(Image.fromarray(payload["arr"]), payload["model"]).getchannel("A"), np.float32) / 255.0
                    except Exception:
                        person = None
                result = hair_mask_auto(payload["arr"], payload["face"], person)
                self.finished.emit(kind, result)
                return

            # CPU operations also run off the GUI thread. This is important for
            # large photos because bilateral filters and connected-components can
            # otherwise freeze Qt even without AI models.
            arr = payload["arr"]
            face = payload.get("face")
            if kind == "enhance":
                result = enhance_pro(arr, payload["strength"])
            elif kind == "smooth":
                mask = skin_mask(arr, face)
                result = skin_smooth(arr, mask, face, payload["strength"])
            elif kind == "tone":
                mask = skin_mask(arr, face)
                result = skin_tone_up(arr, mask, payload["strength"])
            elif kind == "retouch":
                mask = skin_mask(arr, face)
                result = skin_retouch(arr, mask, face, payload["strength"])
            elif kind == "shine":
                mask = skin_mask(arr, face)
                result = skin_shine_fix(arr, mask, payload["strength"])
            elif kind == "pimple":
                result = remove_blemishes(arr, face, payload["strength"])
            elif kind == "hair":
                hm = payload["mask"]
                result = apply_hair_color(arr, hm, payload["color"], payload["strength"])
            else:
                raise RuntimeError("Unknown worker task: " + kind)
            self.finished.emit(kind, result)
        except Exception as e:
            self.failed.emit(kind, str(e))

# ------------------- GUI -------------------
def pil_to_pixmap(im):
    im = im.convert("RGB")
    q = QImage(im.tobytes(), im.width, im.height, 3 * im.width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q)

class CropView(QWidget):
    """Photoshop style crop: corner/edge tene resize, bhitore drag = sorano,
    baire drag = ghurano, scroll = zoom, double click ba Enter = done."""
    HIT = 14
    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 650); self.setMouseTracking(True)
        self.pm = None; self.iw = self.ih = 1
        self.ratio = 0.78; self.box = [0, 0, 1, 0.0]
        self.changed = None; self.on_confirm = None
        self._mode = None; self._info = None; self._st = None

    def set_data(self, full_size, ratio, box):
        self.iw, self.ih = full_size; self.ratio = ratio; self.box = box; self.update()

    def set_preview(self, pil):
        self.pm = pil_to_pixmap(pil); self.update()

    def _geom(self):
        s = min(self.width() / (self.iw * 1.3), self.height() / (self.ih * 1.3))
        return s, (self.width() - self.iw * s) / 2, (self.height() - self.ih * s) / 2

    def _w2i(self, pt):
        s, ox, oy = self._geom(); return (pt.x() - ox) / s, (pt.y() - oy) / s

    def _i2w(self, x, y):
        s, ox, oy = self._geom(); return QPointF(ox + x * s, oy + y * s)

    @staticmethod
    def _local(ix, iy, box):
        t = math.radians(box[3]); dx, dy = ix - box[0], iy - box[1]
        return dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)

    @staticmethod
    def _to_img(lx, ly, box):
        t = math.radians(box[3])
        return box[0] + lx * math.cos(t) - ly * math.sin(t), box[1] + lx * math.sin(t) + ly * math.cos(t)

    def _hit(self, pt):
        s = self._geom()[0]; ix, iy = self._w2i(pt); lx, ly = self._local(ix, iy, self.box)
        bh = self.box[2]; bw = bh * self.ratio; r = self.HIT / s
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            if abs(lx - sx * bw / 2) < r and abs(ly - sy * bh / 2) < r: return "resize", (sx, sy)
        if abs(lx) <= bw / 2 and abs(ly) <= bh / 2: return "move", None
        return "rotate", None

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#cfcfcf"))
        if self.pm is None:
            p.end(); return
        s, ox, oy = self._geom()
        p.drawPixmap(QRectF(ox, oy, self.iw * s, self.ih * s), self.pm, QRectF(self.pm.rect()))
        cx, cy, bh, ang = self.box; bw = bh * self.ratio
        pts = [self._i2w(*self._to_img(sx * bw / 2, sy * bh / 2, self.box)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        path = QPainterPath(); path.addRect(QRectF(self.rect())); path.addPolygon(QPolygonF(pts)); path.closeSubpath()
        p.fillPath(path, QColor(0, 0, 0, 130))                 # box er baire andhokar
        w, h = bw * s, bh * s
        p.save(); p.translate(self._i2w(cx, cy)); p.rotate(ang)
        p.setPen(QPen(QColor(255, 255, 255, 230), 1, Qt.DashLine))   # mathar guide
        p.drawEllipse(QRectF(-w * 0.27, -h / 2 + h * 0.11, w * 0.54, h * 0.71))
        p.drawLine(0, int(-h / 2), 0, int(h / 2))
        p.setPen(QPen(QColor(255, 255, 255), 2)); p.drawRect(QRectF(-w / 2, -h / 2, w, h))
        p.setPen(QPen(QColor(40, 40, 40), 1))
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            r = QRectF(sx * w / 2 - 5, sy * h / 2 - 5, 10, 10)
            p.fillRect(r, QColor(255, 255, 255)); p.drawRect(r)
        p.restore(); p.end()

    def _notify(self):
        self.update()
        if self.changed: self.changed()

    def mousePressEvent(self, e):
        pt = e.position(); self._mode, self._info = self._hit(pt)
        ix, iy = self._w2i(pt); b0 = list(self.box)
        self._st = dict(box=b0, ix=ix, iy=iy, a0=math.degrees(math.atan2(iy - b0[1], ix - b0[0])))

    def mouseDoubleClickEvent(self, e):
        if self.on_confirm: self.on_confirm()

    def mouseMoveEvent(self, e):
        pt = e.position()
        if self._mode is None:
            mode, _ = self._hit(pt)
            self.setCursor({"move": Qt.SizeAllCursor, "resize": Qt.SizeFDiagCursor,
                            "rotate": Qt.OpenHandCursor}[mode]); return
        ix, iy = self._w2i(pt); st = self._st; b0 = st["box"]
        if self._mode == "move":
            self.box[0] = b0[0] + ix - st["ix"]; self.box[1] = b0[1] + iy - st["iy"]
        elif self._mode == "rotate":
            a = math.degrees(math.atan2(iy - b0[1], ix - b0[0])) - st["a0"] + b0[3]
            self.box[3] = (a + 180) % 360 - 180
        else:
            sx, sy = self._info; bh0 = b0[2]; bw0 = bh0 * self.ratio
            lx, ly = self._local(ix, iy, b0)
            ax = -sx * bw0 / 2 if sx else 0; ay = -sy * bh0 / 2 if sy else 0
            if sx and sy: h = max((ly - ay) * sy, (lx - ax) * sx / self.ratio)
            elif sx: h = (lx - ax) * sx / self.ratio
            else: h = (ly - ay) * sy
            h = max(h, 30.0); w = h * self.ratio
            clx = ax + sx * w / 2 if sx else 0; cly = ay + sy * h / 2 if sy else 0
            self.box[0], self.box[1] = self._to_img(clx, cly, b0); self.box[2] = h
        self._notify()

    def mouseReleaseEvent(self, e): self._mode = None

    def wheelEvent(self, e):
        self.box[2] *= 0.95 if e.angleDelta().y() > 0 else 1.05
        self._notify()

class PaintView(QWidget):
    """Retouch Studio: click diye pimple heal, chuler jaiga brush. Scroll = zoom, right/middle drag = sorano"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 650); self.setMouseTracking(True)
        self.pm = None; self.iw = self.ih = 1
        self.ov = None; self.tool = "heal"; self.radius = 12; self.on_action = None
        self.zoom = 1.0; self.pan = QPointF(0, 0); self.dirty_hair = False
        self._paint = False; self._panning = False; self._last = None; self._lastimg = None; self._cur = None

    def set_image(self, arr, keep_view=False):
        arr = np.ascontiguousarray(arr); h, w = arr.shape[:2]
        self.pm = QPixmap.fromImage(QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888))
        self.iw, self.ih = w, h
        if not keep_view: self.zoom = 1.0; self.pan = QPointF(0, 0)
        self.update()

    def refresh(self, arr): self.set_image(arr, True)

    def clear(self): self.pm = None; self.ov = None; self.update()

    def set_overlay(self, mask):
        if mask is None: self.ov = None; self.update(); return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0] = 255; rgba[..., 2] = 90
        rgba[..., 3] = (np.clip(mask, 0, 1) * 120).astype(np.uint8)
        self.ov = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).convertToFormat(QImage.Format_ARGB32_Premultiplied)
        self.update()

    def get_overlay_mask(self):
        if self.ov is None: return None
        q = self.ov.convertToFormat(QImage.Format_RGBA8888)
        buf = np.frombuffer(q.constBits(), np.uint8)
        a = buf.reshape(q.height(), q.bytesPerLine())[:, :q.width() * 4].reshape(q.height(), q.width(), 4)[..., 3]
        return np.clip(a.astype(np.float32) / 120.0, 0, 1)

    def _geom(self):
        s = min(self.width() / self.iw, self.height() / self.ih) * self.zoom
        return s, (self.width() - self.iw * s) / 2 + self.pan.x(), (self.height() - self.ih * s) / 2 + self.pan.y()

    def _w2i(self, pt):
        s, ox, oy = self._geom(); return (pt.x() - ox) / s, (pt.y() - oy) / s

    def _stroke(self, p1, p2, erase):
        if self.ov is None:
            self.ov = QImage(self.iw, self.ih, QImage.Format_ARGB32_Premultiplied); self.ov.fill(0)
        col = QColor(255, 0, 90, 120); r = self.radius
        qp = QPainter(self.ov)
        qp.setCompositionMode(QPainter.CompositionMode_Clear if erase else QPainter.CompositionMode_Source)
        qp.setPen(QPen(col, r * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)); qp.setBrush(col)
        qp.drawLine(QPointF(*p1), QPointF(*p2)); qp.drawEllipse(QPointF(*p2), r, r); qp.end()
        self.dirty_hair = True

    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#cfcfcf"))
        if self.pm is None:
            p.end(); return
        s, ox, oy = self._geom(); target = QRectF(ox, oy, self.iw * s, self.ih * s)
        p.setRenderHint(QPainter.SmoothPixmapTransform, s < 1)
        p.drawPixmap(target, self.pm, QRectF(self.pm.rect()))
        if self.ov is not None and self.tool.startswith("hair"): p.drawImage(target, self.ov)
        if self._cur is not None:
            r = self.radius * s; p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255), 2)); p.drawEllipse(self._cur, r, r)
            p.setPen(QPen(QColor(0, 0, 0), 1, Qt.DashLine)); p.drawEllipse(self._cur, r, r)
        p.end()

    def mousePressEvent(self, e):
        if e.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = True; self._last = e.position(); return
        if e.button() == Qt.LeftButton and self.pm is not None:
            ix, iy = self._w2i(e.position())
            if not (0 <= ix < self.iw and 0 <= iy < self.ih): return
            if self.tool == "heal":
                if self.on_action: self.on_action("heal", ix, iy)
            else:
                self._paint = True; self._lastimg = (ix, iy)
                self._stroke((ix, iy), (ix, iy), self.tool == "hair_erase"); self.update()

    def mouseMoveEvent(self, e):
        self._cur = e.position()
        if self._panning:
            d = e.position() - self._last; self._last = e.position()
            self.pan = QPointF(self.pan.x() + d.x(), self.pan.y() + d.y())
        elif self._paint:
            ix, iy = self._w2i(e.position())
            self._stroke(self._lastimg, (ix, iy), self.tool == "hair_erase"); self._lastimg = (ix, iy)
        self.update()

    def mouseReleaseEvent(self, e): self._panning = False; self._paint = False

    def leaveEvent(self, e): self._cur = None; self.update()

    def wheelEvent(self, e):
        if self.pm is None: return
        pt = e.position(); ix, iy = self._w2i(pt)
        self.zoom = max(0.2, min(30.0, self.zoom * (1.25 if e.angleDelta().y() > 0 else 0.8)))
        s, ox, oy = self._geom()
        self.pan = QPointF(self.pan.x() + pt.x() - (ox + ix * s), self.pan.y() + pt.y() - (oy + iy * s))
        self.update()

def hslider(lo, hi, val):
    s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val); return s

def btn(text, fn, style=""):
    b = QPushButton(text); b.setMinimumHeight(38); b.clicked.connect(fn)
    if style: b.setStyleSheet(style)
    return b

def group(title, items):
    g = QGroupBox(title); l = QVBoxLayout(g)
    for it in items:
        l.addLayout(it) if hasattr(it, "addWidget") else l.addWidget(it)
    return g

def row(*ws):
    l = QHBoxLayout()
    for w in ws: l.addWidget(w)
    return l

class App(QWidget):
    """State: self.rgb (photo) + self.alpha (bg remove hole mask, None = bg ache).
    AUTO ar MANUAL duto-i ei state e kaj kore, tai mix kora jay."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME + " - Photo Studio (AI)")
        self.resize(1300, 880)
        self.orig = None; self.rgb = None; self.alpha = None; self.face = None
        self.history = []; self.redo_stack = []
        self.hair_mask = None; self.hair_rgb = tuple(HAIR_COLORS["Black (Kalo)"]); self.dress = None
        self._pre = None; self.rt_work = None; self.rt_stack = []
        self.box = None; self.base_h = 1.0
        self.bg_rgb = (255, 255, 255)
        self.photo_base = None; self.photo = None; self.sheet = None; self.showing_orig = False

        self.preview = QLabel("Photo select koro")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(560, 650)
        self.preview.setStyleSheet("border:1px solid #888; background:#eee;")
        self.crop_view = CropView(); self.crop_view.changed = self.on_box_changed
        self.crop_view.on_confirm = self.on_enter
        self.stack = QStackedWidget()
        self.stack.addWidget(self.preview); self.stack.addWidget(self.crop_view)
        self.paint_view = PaintView(); self.paint_view.on_action = self.on_paint_action
        self.stack.addWidget(self.paint_view)

        self.size_box = QComboBox(); self.size_box.addItems(SIZES.keys())
        self.size_box.currentIndexChanged.connect(self.on_size_change)
        self.model_box = QComboBox(); self.model_box.addItems(MODELS.keys())
        self.copies = QSpinBox(); self.copies.setRange(1, 60); self.copies.setValue(8)
        self.swatch = QLabel(""); self.swatch.setFixedHeight(28)

        # AUTO ticks
        self.c_crop = QCheckBox("Auto Crop (face dhore)")
        self.c_manual = QCheckBox("Auto er por crop manual adjust korbo")
        self.c_bg = QCheckBox("Auto Background Remove")
        self.c_enh = QCheckBox("Auto Enhance")
        self.c_smooth = QCheckBox("Auto Skin Retouch (natural)")
        self.c_tone = QCheckBox("Auto Skin Tone Up")
        for c in (self.c_crop, self.c_bg, self.c_enh, self.c_smooth, self.c_tone): c.setChecked(True)

        # sliders
        self.s_smooth = hslider(0, 100, 50); self.s_tone = hslider(0, 100, 50)
        self.s_zoom = hslider(40, 300, 100); self.s_zoom.valueChanged.connect(self.on_zoom)
        self.s_tilt = hslider(-450, 450, 0); self.s_tilt.valueChanged.connect(self.on_tilt)
        self.s_bright = hslider(-50, 50, 0); self.s_bright.valueChanged.connect(self.apply_adjust)
        self.s_contrast = hslider(-50, 50, 0); self.s_contrast.valueChanged.connect(self.apply_adjust)

        self.s_enh = hslider(0, 100, 60); self.s_pimple = hslider(0, 100, 55)
        self.s_aiface = hslider(10, 100, 50); self.s_hair = hslider(20, 100, 100)
        self.s_brush = hslider(2, 120, 12); self.s_brush.valueChanged.connect(self.on_brush)
        self.s_dress = hslider(30, 220, 100); self.s_dress.valueChanged.connect(self.apply_adjust)
        self.s_dress_y = hslider(-40, 60, 0); self.s_dress_y.valueChanged.connect(self.apply_adjust)
        self.c_pimple = QCheckBox("Auto Pimple/Dag Remove"); self.c_pimple.setChecked(True)
        self.c_shine = QCheckBox("Auto Oily Shine kombano")
        self.c_face = QCheckBox("AI Face Restore (GFPGAN, optional)")
        self.c_hair = QCheckBox("Auto Hair Color (niche color select koro)")
        self.hair_box = QComboBox(); self.hair_box.addItems(list(HAIR_COLORS.keys()))
        self.hair_box.currentTextChanged.connect(self.on_hair_pick)
        self.rt_tool_box = QComboBox(); self.rt_tool_box.addItems(["Pimple Heal (click)", "Hair area + (brush)", "Hair area - (eraser)"])
        self.rt_tool_box.currentIndexChanged.connect(self.on_rt_tool)
        self.c_bw = QCheckBox("Black & White photo"); self.c_bw.stateChanged.connect(self.apply_adjust)
        self.cap_edit = QLineEdit(); self.cap_edit.setPlaceholderText("Photo er niche Name (optional, English)")
        self.cap_edit.textChanged.connect(self.apply_adjust)
        self.c_date = QCheckBox("Date add koro"); self.c_date.stateChanged.connect(self.apply_adjust)
        self.paper_box = QComboBox(); self.paper_box.addItems(list(PAPERS.keys()))
        self.c_mix = QCheckBox("Mixed sheet: ar ekta size o dao")
        self.size2_box = QComboBox(); self.size2_box.addItems(list(SIZES.keys())); self.size2_box.setCurrentIndex(2)
        self.copies2 = QSpinBox(); self.copies2.setRange(1, 60); self.copies2.setValue(8); self.copies2.setSuffix(" copy (2nd size)")
        self.c_border = QCheckBox("Cut line (halka border)"); self.c_border.setChecked(True)
        self.sp_kb = QSpinBox(); self.sp_kb.setRange(10, 2000); self.sp_kb.setValue(100); self.sp_kb.setSuffix(" KB max")
        self.sp_ow = QSpinBox(); self.sp_ow.setRange(0, 4000); self.sp_ow.setSuffix(" px width (0 = same size)")
        self.sp_oh = QSpinBox(); self.sp_oh.setRange(0, 4000); self.sp_oh.setSuffix(" px height (0 = same size)")

        side = QVBoxLayout()
        title = QLabel(APP_NAME); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:22px;font-weight:bold;color:#1565c0;padding:4px;")
        side.addWidget(title)
        side.addWidget(btn("1. Photo Open", self.open_photo))
        side.addWidget(group("Size", [self.size_box]))
        presets = [btn(n, lambda _=False, c=col: self.set_bg(c)) for n, col in BG_COLORS.items()]
        side.addWidget(group("BACKGROUND COLOR (jokhon khushi change koro)", [
            self.swatch, row(*presets), btn("Color Picker (je kono color)...", self.pick_bg)]))

        side.addWidget(group("AUTO (ek click e shob)", [
            self.c_crop, self.c_manual, self.c_bg, self.c_enh, self.c_pimple, self.c_smooth, self.c_tone,
            self.c_shine, self.c_face, self.c_hair,
            QLabel("Koto beshi hobe: niche AI section er slider theke ney"),
            row(btn("Select All", lambda: self.set_all(True)), btn("Clear", lambda: self.set_all(False))),
            btn("AUTO CHALAO", self.run_auto, "background:#2e7d32;color:white;font-weight:bold;")]))

        side.addWidget(group("AI ENHANCE / SKIN RETOUCH / PIMPLE", [
            QLabel("Enhance strength"), self.s_enh, btn("Photo Enhance PRO", self.m_enhance),
            row(btn("AI Face Restore", self.m_face_ai), btn("AI Upscale 2x", self.m_upscale)),
            QLabel("AI Face strength (kom = asol mukh er kachhakachhi)"), self.s_aiface,
            QLabel("Skin Retouch / Smooth strength"), self.s_smooth,
            row(btn("Skin Retouch (natural)", self.m_retouch), btn("Skin Smooth (soft)", self.m_smooth)),
            QLabel("Skin Tone Up strength"), self.s_tone, btn("Skin Tone Up", self.m_tone),
            btn("Oily Shine kombao", self.m_shine),
            QLabel("Pimple/Dag sensitivity"), self.s_pimple, btn("Pimple/Dag Auto Remove", self.m_pimple)]))

        side.addWidget(group("HAIR COLOR", [
            self.hair_box, btn("Custom Color...", self.hair_custom), QLabel("Hair color strength"), self.s_hair,
            row(btn("Hair Color Apply", lambda: self.m_hair()), btn("Area Edit (brush)", self.hair_edit))]))

        rt_hint = QLabel("Zoom kore pimple er upor click. Brush = pimple er cheye ektu boro. "
                         "Scroll = zoom | Right-drag = sorano | Enter = Done | Esc = Cancel")
        rt_hint.setWordWrap(True)
        side.addWidget(group("RETOUCH STUDIO (zoom kore click)", [
            btn("Retouch Studio Kholo", lambda: self.open_retouch("heal"), "background:#6a1b9a;color:white;font-weight:bold;"),
            self.rt_tool_box, QLabel("Brush size"), self.s_brush, rt_hint,
            btn("Done (Enter)", lambda: self.close_retouch())]))

        side.addWidget(group("BG REMOVE / UNDO", [
            QLabel("BG Remove Model"), self.model_box,
            row(btn("BG Remove", self.m_bg_remove), btn("BG Restore", self.m_bg_restore)),
            row(btn("Undo (Ctrl+Z)", self.undo), btn("Redo (Ctrl+Y)", self.redo)),
            btn("Reset Original", self.reset_orig)]))

        hint = QLabel("Corner tene resize | bhitore drag = sorao | baire drag = ghurao | "
                      "scroll = zoom | Enter ba double-click = Done | Esc = Cancel")
        hint.setWordWrap(True)
        side.addWidget(group("MANUAL CROP (Photoshop style)", [
            btn("Crop Kholo", self.open_editor, "background:#1565c0;color:white;font-weight:bold;"), hint,
            QLabel("Zoom"), self.s_zoom, QLabel("Rotate (degree)"), self.s_tilt,
            row(btn("⟲ 90", lambda: self.rot90(-90)), btn("⟳ 90", lambda: self.rot90(90))),
            row(btn("←", lambda: self.nudge(-1, 0)), btn("→", lambda: self.nudge(1, 0)),
                btn("↑", lambda: self.nudge(0, -1)), btn("↓", lambda: self.nudge(0, 1))),
            btn("Crop Apply (Enter)", self.apply_crop, "background:#ef6c00;color:white;font-weight:bold;")]))

        side.addWidget(group("STUDIO: Dress / Name / B&W", [
            row(btn("Dress (PNG) Load", self.pick_dress), btn("Dress Remove", lambda: self.load_dress(None))),
            QLabel("Dress size"), self.s_dress, QLabel("Dress upor / niche"), self.s_dress_y,
            self.cap_edit, self.c_date, self.c_bw]))
        side.addWidget(group("Brightness / Contrast (final photo)", [
            QLabel("Brightness"), self.s_bright, QLabel("Contrast"), self.s_contrast]))

        side.addWidget(btn("Original / Result dekho", self.toggle))
        side.addWidget(group("PAPER / SHEET", [
            QLabel("Paper"), self.paper_box, QLabel("Copies (main size)"), self.copies,
            self.c_mix, self.size2_box, self.copies2, self.c_border,
            btn("Sheet Layout", self.do_layout), btn("Print", self.print_sheet),
            row(btn("Save Sheet JPG", self.save_sheet), btn("Save Sheet PDF", self.save_pdf))]))
        side.addWidget(group("ONLINE FORM SAVE (KB limit)", [
            self.sp_kb, self.sp_ow, self.sp_oh, btn("Save for Online", self.save_online_cb)]))
        side.addWidget(btn("Save Single Photo", self.save_single))
        self.status = QLabel(""); self.status.setWordWrap(True)
        side.addWidget(self.status); side.addStretch()

        panel = QWidget(); panel.setLayout(side)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); scroll.setFixedWidth(400)
        body = QHBoxLayout(); body.addWidget(self.stack, 1); body.addWidget(scroll)
        outer = QVBoxLayout(self); outer.setMenuBar(self.build_menu()); outer.addLayout(body)
        for key in (Qt.Key_Return, Qt.Key_Enter):
            QShortcut(QKeySequence(key), self).activated.connect(self.on_enter)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.on_esc)
        self.set_bg(self.bg_rgb)

    # ---------- helpers ----------
    def set_all(self, v):
        for c in (self.c_crop, self.c_bg, self.c_enh, self.c_pimple, self.c_smooth, self.c_tone, self.c_shine): c.setChecked(v)
        if not v:
            self.c_face.setChecked(False); self.c_hair.setChecked(False)

    def ratio(self):
        W, H = SIZES[self.size_box.currentText()]; return W / H

    def busy(self, msg):
        self.status.setText(msg); QApplication.processEvents()

    def need_img(self):
        if self.rgb is None:
            QMessageBox.warning(self, "Oops", "Age 'Photo Open' diye photo nao"); return False
        return True

    def source(self):
        im = self.rgb.convert("RGBA")
        if self.alpha is not None: im.putalpha(self.alpha)
        return im

    def composite(self):
        if self.alpha is None: return self.rgb
        c = Image.new("RGB", self.rgb.size, self.bg_rgb); c.paste(self.rgb, (0, 0), self.alpha); return c

    def show_img(self, im):
        self.stack.setCurrentIndex(0)
        self.preview.setPixmap(pil_to_pixmap(im).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def push(self):
        self.history.append(self._snap()); self.history = self.history[-15:]; self.redo_stack = []

    def _snap(self):
        return dict(rgb=self.rgb, alpha=self.alpha, orig=self.orig, face=self.face,
                    box=list(self.box) if self.box else None, base_h=self.base_h,
                    cropped=self.photo_base is not None)

    def _restore(self, s):
        self.rgb, self.alpha, self.orig, self.face = s["rgb"], s["alpha"], s["orig"], s["face"]
        self.box = list(s["box"]) if s["box"] else None; self.base_h = s["base_h"]
        if not s["cropped"]: self.photo_base = self.photo = self.sheet = None
        if self.hair_mask is not None and self.hair_mask.shape != (self.rgb.height, self.rgb.width): self.hair_mask = None
        if self.box: self.on_box_changed()
        if s["cropped"] and self.photo_base is None and self.stack.currentIndex() == 0: self.apply_crop()
        else: self.after_change()

    def after_change(self):
        """Kono kichu bodlale: editor khola thakle preview update, crop kora thakle abar crop, noyto full photo dekhao"""
        if self.rgb is None: return
        if self.stack.currentIndex() == 1: self.refresh_preview()
        elif self.photo_base is not None: self.apply_crop()
        else: self.show_img(self.composite())

    # ---------- open ----------
    def open_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Photo", "", "Images (*.jpg *.jpeg *.png *.webp)")
        if not p: return
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        if max(im.size) > 4096: im.thumbnail((4096, 4096), Image.LANCZOS)     # onek boro photo halka kore nei (fast cholbe)
        self.orig = im
        self.rgb = self.orig; self.alpha = None; self.history = []; self.redo_stack = []; self.hair_mask = None
        self.face = detect_face(np.array(self.orig))
        self.box = None; self.photo_base = self.photo = self.sheet = None
        self.show_img(self.orig)
        self.status.setText("Photo loaded. " + ("Face detect holo." if self.face else "Face paini (skin effect e face lagbe)."))

    # ---------- background color ----------
    def set_bg(self, rgb):
        self.bg_rgb = tuple(int(v) for v in rgb)
        self.swatch.setStyleSheet("background: rgb(%d,%d,%d); border:1px solid #444;" % self.bg_rgb)
        self.after_change()

    def pick_bg(self):
        c = QColorDialog.getColor(QColor(*self.bg_rgb), self, "Background Color")
        if c.isValid(): self.set_bg((c.red(), c.green(), c.blue()))

    # ---------- MANUAL effects ----------
    def m_bg_remove(self):
        if not self.need_img(): return
        self.busy("BG remove hochche... (prothom bar model download hote pare)")
        try:
            self.push()
            self.alpha = remove_bg(self.rgb, MODELS[self.model_box.currentText()]).getchannel("A")
            self.after_change(); self.status.setText("BG remove holo. Color bodlate upore BACKGROUND COLOR use koro")
        except Exception as e:
            self.history.pop(); QMessageBox.critical(self, "Error", str(e)); self.status.setText("Error")

    def m_bg_restore(self):
        if self.need_img(): self.push(); self.alpha = None; self.after_change(); self.status.setText("Original bg fire ashlo")

    def m_enhance(self):
        if not self.need_img(): return
        self.push(); self.rgb = Image.fromarray(enhance_pro(np.array(self.rgb), self.s_enh.value())); self.after_change()
        self.status.setText("Photo Enhance PRO holo")

    def _need_face(self):
        if not self.need_img(): return False
        if self.face is None:
            QMessageBox.information(self, "Face paini", "Skin effect er jonno face detect lagbe. Shoja-shuji mukh wala photo nao."); return False
        return True

    def m_smooth(self):
        if not self._need_face(): return
        self.push(); a = np.array(self.rgb)
        self.rgb = Image.fromarray(skin_smooth(a, skin_mask(a, self.face), self.face, self.s_smooth.value()))
        self.after_change(); self.status.setText("Skin smooth holo")

    def m_tone(self):
        if not self._need_face(): return
        self.push(); a = np.array(self.rgb)
        self.rgb = Image.fromarray(skin_tone_up(a, skin_mask(a, self.face), self.s_tone.value()))
        self.after_change(); self.status.setText("Skin tone up holo")

    def undo(self):
        if self.stack.currentIndex() == 2: return self.rt_undo()
        if not self.history: return
        self.redo_stack.append(self._snap()); self._restore(self.history.pop()); self.status.setText("Undo")

    def redo(self):
        if self.stack.currentIndex() == 2 or not self.redo_stack: return
        self.history.append(self._snap()); self._restore(self.redo_stack.pop()); self.status.setText("Redo")

    def reset_orig(self):
        if self.need_img(): self.push(); self.rgb = self.orig; self.alpha = None; self.after_change(); self.status.setText("Original e fire")

    # ---------- NEW: AI / RETOUCH / STUDIO ----------
    def m_retouch(self):
        if not self._need_face(): return
        self.push(); a = np.array(self.rgb)
        self.rgb = Image.fromarray(skin_retouch(a, skin_mask(a, self.face), self.face, self.s_smooth.value()))
        self.after_change(); self.status.setText("Skin Retouch holo (natural)")

    def m_shine(self):
        if not self._need_face(): return
        self.push(); a = np.array(self.rgb)
        self.rgb = Image.fromarray(skin_shine_fix(a, skin_mask(a, self.face), 70))
        self.after_change(); self.status.setText("Oily shine kombano holo")

    def m_pimple(self):
        if not self._need_face(): return
        self.busy("Pimple/dag khujchi...")
        self.push(); a = np.array(self.rgb)
        out, n = remove_blemishes(a, self.face, self.s_pimple.value())
        self.rgb = Image.fromarray(out); self.after_change()
        self.status.setText(f"{n} ta pimple/dag remove holo. Baki thakle Retouch Studio te click koro")

    def m_face_ai(self):
        if not self.need_img(): return
        self.busy("AI Face Restore cholche... (prothom bar model download hobe, GPU na thakle somoy lagbe)")
        self.push()
        try:
            self.rgb = Image.fromarray(ai_face_restore(np.array(self.rgb), self.s_aiface.value() / 100.0))
            self.after_change(); self.status.setText("AI Face Restore holo")
        except Exception as e:
            self.history.pop(); self.status.setText("AI Face Restore hoyni")
            QMessageBox.warning(self, "AI Face Restore", "Cholche na.\n\n" + str(e))

    def m_upscale(self):
        if not self.need_img(): return
        if max(self.rgb.size) >= 2400:
            self.status.setText("Photo already boro (2400px+), upscale dorkar nai"); return
        self.busy("Upscale hochche... (AI install thakle prothom bar model download hobe)")
        self.push()
        try:
            big, how = upscale2x(self.rgb)
        except Exception as e:
            self.history.pop(); QMessageBox.critical(self, "Error", str(e)); return
        k = big.width / self.rgb.width
        self.orig = big if self.orig is self.rgb else self.orig.resize(big.size, Image.LANCZOS)
        if self.alpha is not None: self.alpha = self.alpha.resize(big.size, Image.LANCZOS)
        self.rgb = big
        if self.face is not None: self.face = tuple(int(v * k) for v in self.face)
        if self.box is not None: self.box = [self.box[0] * k, self.box[1] * k, self.box[2] * k, self.box[3]]; self.base_h *= k
        self.hair_mask = None
        self.after_change(); self.status.setText("Upscale holo: " + how)

    # ---- hair ----
    def _person_mask(self):
        if self.alpha is not None: return np.array(self.alpha, np.float32) / 255.0
        try: return np.array(remove_bg(self.rgb, "u2net_human_seg").getchannel("A"), np.float32) / 255.0
        except Exception: return None

    def hair_detect(self):
        self.busy("Chuler jaiga khujchi...")
        self.hair_mask = hair_mask_auto(np.array(self.rgb), self.face, self._person_mask())

    def hair_custom(self):
        c = QColorDialog.getColor(QColor(*self.hair_rgb), self, "Hair Color")
        if c.isValid(): self.hair_rgb = (c.red(), c.green(), c.blue()); self.status.setText("Custom hair color set. Ekhon 'Hair Color Apply' chapo")

    def on_hair_pick(self, t):
        if t in HAIR_COLORS: self.hair_rgb = HAIR_COLORS[t]

    def m_hair(self, color=None):
        if not self.need_img(): return
        if self.hair_mask is None or self.hair_mask.shape != (self.rgb.height, self.rgb.width):
            if not self._need_face(): return
            self.hair_detect()
        if (self.hair_mask > 0.5).sum() < 200:
            QMessageBox.information(self, "Chul paini", "Chuler jaiga auto dhora jayni. 'Area Edit (brush)' diye lal brush e chul ankon koro, tarpor abar Apply koro.")
            return
        self.push()
        self.rgb = Image.fromarray(apply_hair_color(np.array(self.rgb), self.hair_mask, color or self.hair_rgb, self.s_hair.value()))
        self.after_change(); self.status.setText("Hair color holo. Jaiga thik na hole 'Area Edit (brush)' diye thik koro")

    def hair_edit(self):
        self.open_retouch("hair_add")

    # ---- Retouch Studio (zoom + click) ----
    def open_retouch(self, tool="heal"):
        if not self.need_img() or self.stack.currentIndex() == 1: return
        if self.stack.currentIndex() != 2:
            self.rt_work = np.ascontiguousarray(np.array(self.rgb)); self.rt_stack = []
            self.paint_view.set_image(self.rt_work); self.paint_view.dirty_hair = False
            ok = self.hair_mask is not None and self.hair_mask.shape == self.rt_work.shape[:2]
            self.paint_view.set_overlay(self.hair_mask if ok else None)
            self.stack.setCurrentIndex(2)
        i = RT_TOOLS.index(tool)
        self.rt_tool_box.blockSignals(True); self.rt_tool_box.setCurrentIndex(i); self.rt_tool_box.blockSignals(False)
        self.on_rt_tool(i)

    def on_rt_tool(self, i):
        if self.stack.currentIndex() != 2: return
        tool = RT_TOOLS[i]; self.paint_view.tool = tool
        if tool.startswith("hair") and self.paint_view.ov is None and self.face is not None:
            if self.hair_mask is not None and self.hair_mask.shape == (self.rgb.height, self.rgb.width):
                self.paint_view.set_overlay(self.hair_mask)
            else:
                self.hair_detect()
        self.paint_view.update()
        self.status.setText("Pimple er upor click koro (brush = pimple er cheye ektu boro). Scroll = zoom, Right-drag = sorano. Enter = Done, Esc = Cancel"
                            if tool == "heal" else
                            "Lal jaiga = chul. Brush + diye jog, eraser diye bad dao. Enter = Done, tarpor 'Hair Color Apply'")

    def on_brush(self, v):
        self.paint_view.radius = v; self.paint_view.update()

    def on_paint_action(self, kind, x, y):
        if kind != "heal" or self.rt_work is None: return
        res = heal_spot(self.rt_work, x, y, self.paint_view.radius)
        if res: self.rt_stack.append(res); self.paint_view.refresh(self.rt_work)
        else: self.status.setText("Eta photo r kinarar kachhe, ektu bhitore click koro")

    def rt_undo(self):
        if not self.rt_stack: return
        x0, y0, old = self.rt_stack.pop(); h, w = old.shape[:2]
        self.rt_work[y0:y0 + h, x0:x0 + w] = old; self.paint_view.refresh(self.rt_work)

    def close_retouch(self, commit=True):
        if self.stack.currentIndex() != 2: return
        pv = self.paint_view
        changed = commit and bool(self.rt_stack)
        if commit and pv.dirty_hair:
            m = pv.get_overlay_mask()
            if m is not None and m.shape == (self.rgb.height, self.rgb.width):
                self.hair_mask = cv2.GaussianBlur(m, (0, 0), 1.5)
        if changed:
            self.push(); self.rgb = Image.fromarray(self.rt_work)
        self.rt_work = None; self.rt_stack = []; pv.clear()
        self.stack.setCurrentIndex(0)
        if changed: self.after_change()
        else: self.show_img(self.photo if self.photo is not None else self.composite())
        self.status.setText("Retouch holo" if changed else ("Chuler jaiga save holo" if commit and self.hair_mask is not None else "Retouch bondho"))

    # ---- dress ----
    def pick_dress(self):
        p, _ = QFileDialog.getOpenFileName(self, "Dress PNG", DRESS_DIR if os.path.isdir(DRESS_DIR) else "", "PNG (*.png)")
        if p: self.load_dress(p)

    def load_dress(self, p):
        try: self.dress = Image.open(p).convert("RGBA") if p else None
        except Exception as e: QMessageBox.critical(self, "Error", str(e)); return
        self.apply_adjust(); self.status.setText("Dress set holo (crop kora thakle sathe sathe boshbe)" if p else "Dress remove holo")

    def fill_dress_menu(self):
        mn = self.dress_menu; mn.clear()
        files = sorted(f for f in os.listdir(DRESS_DIR) if f.lower().endswith(".png")) if os.path.isdir(DRESS_DIR) else []
        for f in files:
            mn.addAction(os.path.splitext(f)[0]).triggered.connect(lambda _=False, p=os.path.join(DRESS_DIR, f): self.load_dress(p))
        if files: mn.addSeparator()
        mn.addAction("Onno PNG khujo...").triggered.connect(lambda _=False: self.pick_dress())
        mn.addAction("Dress Remove").triggered.connect(lambda _=False: self.load_dress(None))

    # ---- menu ----
    def build_menu(self):
        mb = QMenuBar()
        def add(menu, text, fn, key=None):
            a = QAction(text, self); a.triggered.connect(lambda _=False, f=fn: f())
            if key: a.setShortcut(QKeySequence(key))
            menu.addAction(a)
        m = mb.addMenu("&File")
        add(m, "Photo Open...", self.open_photo, "Ctrl+O")
        add(m, "Save Single Photo (JPG)", self.save_single, "Ctrl+S")
        add(m, "Save Sheet (JPG)", self.save_sheet, "Ctrl+Shift+S")
        add(m, "Save Sheet (PDF)", self.save_pdf)
        add(m, "Save for Online Form (KB limit)", self.save_online_cb)
        add(m, "Print Sheet", self.print_sheet, "Ctrl+P")
        m.addSeparator(); add(m, "Exit", self.close)
        m = mb.addMenu("&Edit")
        add(m, "Undo", self.undo, "Ctrl+Z"); add(m, "Redo", self.redo, "Ctrl+Y")
        add(m, "Reset Original", self.reset_orig); add(m, "Original / Result dekho", self.toggle)
        m = mb.addMenu("&AI Tools")
        add(m, "AUTO CHALAO (shob feature ekshathe)", self.run_auto, "F5")
        m.addSeparator()
        add(m, "Photo Enhance PRO", self.m_enhance)
        add(m, "AI Face Restore (GFPGAN)", self.m_face_ai)
        add(m, "AI Upscale 2x (chhoto photo boro kora)", self.m_upscale)
        m.addSeparator()
        add(m, "Skin Retouch (natural)", self.m_retouch)
        add(m, "Skin Smooth (soft)", self.m_smooth)
        add(m, "Skin Tone Up", self.m_tone)
        add(m, "Oily Shine kombao", self.m_shine)
        add(m, "Pimple / Dag Auto Remove", self.m_pimple)
        add(m, "Retouch Studio (click diye pimple heal)", lambda: self.open_retouch("heal"), "F6")
        m.addSeparator()
        hm = m.addMenu("Hair Color")
        for name, col in HAIR_COLORS.items(): add(hm, name, lambda c=col: self.m_hair(c))
        hm.addSeparator()
        add(hm, "Custom Color...", self.hair_custom)
        add(hm, "Hair Color Apply (selected color)", lambda: self.m_hair())
        add(hm, "Area Edit (brush)", self.hair_edit)
        m.addSeparator()
        add(m, "Background Remove", self.m_bg_remove); add(m, "Background Restore", self.m_bg_restore)
        m = mb.addMenu("&Studio")
        add(m, "Crop Kholo (Photoshop style)", self.open_editor)
        self.dress_menu = m.addMenu("Dress / Suit (PNG)")
        self.dress_menu.aboutToShow.connect(self.fill_dress_menu)
        add(m, "Black && White on/off", lambda: self.c_bw.setChecked(not self.c_bw.isChecked()))
        add(m, "Sheet Layout", self.do_layout, "Ctrl+L")
        m = mb.addMenu("&Help")
        add(m, "About Ripon Computer", self.about)
        return mb

    def about(self):
        QMessageBox.about(self, APP_NAME, "<h3>Ripon Computer - Photo Studio</h3>"
                          "Passport / Visa / Stamp photo, AI enhance, skin retouch, pimple remove, hair color, "
                          "BG remove, dress, A4/4R sheet, online form save.")

    # ---------- AUTO ----------
    def run_auto(self):
        if not self.need_img(): return
        self.busy("Auto processing... (prothom bar model download hote pare)")
        notes = []
        try:
            self.push(); self.alpha = None
            arr = np.array(self.orig)
            if self.c_enh.isChecked(): arr = enhance_pro(arr, self.s_enh.value())
            if self.c_face.isChecked():
                try: arr = ai_face_restore(arr, self.s_aiface.value() / 100.0)
                except Exception as e: notes.append("AI Face skip: " + (str(e).splitlines() or [""])[0][:70])
            skin_on = self.c_smooth.isChecked() or self.c_tone.isChecked() or self.c_shine.isChecked()
            if self.face is not None:
                if self.c_pimple.isChecked(): arr, _ = remove_blemishes(arr, self.face, self.s_pimple.value())
                mask = skin_mask(arr, self.face) if skin_on else None
                if self.c_shine.isChecked(): arr = skin_shine_fix(arr, mask, 60)
                if self.c_smooth.isChecked(): arr = skin_retouch(arr, mask, self.face, self.s_smooth.value())
                if self.c_tone.isChecked(): arr = skin_tone_up(arr, mask, self.s_tone.value())
            elif skin_on or self.c_pimple.isChecked(): notes.append("Face paini, skin/pimple skip")
            person = None
            if self.c_bg.isChecked() or self.c_hair.isChecked():
                try: person = remove_bg(Image.fromarray(arr), MODELS[self.model_box.currentText()]).getchannel("A")
                except Exception:
                    if self.c_bg.isChecked(): raise
            if self.c_hair.isChecked():
                if self.face is None: notes.append("Face paini, hair skip")
                else:
                    self.hair_mask = hair_mask_auto(arr, self.face, None if person is None else np.array(person, np.float32) / 255.0)
                    arr = apply_hair_color(arr, self.hair_mask, self.hair_rgb, self.s_hair.value())
            self.rgb = Image.fromarray(arr)
            self.alpha = person if self.c_bg.isChecked() else None
            self.box = auto_box(self.rgb, self.face, self.ratio(), self.c_crop.isChecked())
            self.base_h = self.box[2]
            for sl, v in ((self.s_zoom, 100), (self.s_tilt, 0)):
                sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)
            if self.c_manual.isChecked() or (self.face is None and self.c_crop.isChecked()):
                self.open_editor()
                if self.face is None: self.status.setText("Face paini: manual box sorao, tarpor Enter chapo")
            else:
                self.apply_crop()
            if notes: self.status.setText(self.status.text() + "   [" + " | ".join(notes) + "]")
        except Exception as e:
            self.history and self.history.pop()
            QMessageBox.critical(self, "Error", str(e)); self.status.setText("Error")

    # ---------- MANUAL crop ----------
    def ensure_box(self):
        if self.box is None:
            self.box = auto_box(self.rgb, self.face, self.ratio(), True); self.base_h = self.box[2]
            for sl, v in ((self.s_zoom, 100), (self.s_tilt, 0)):
                sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)

    def open_editor(self):
        if not self.need_img(): return
        self.ensure_box(); self._pre = self._snap()
        self.crop_view.set_data(self.rgb.size, self.ratio(), self.box)
        self.refresh_preview(); self.stack.setCurrentIndex(1)
        self.status.setText("Photoshop er moto: corner tano, bhitore drag = sorao, baire drag = ghurao. Enter = Done, Esc = Cancel")

    def refresh_preview(self):
        small = self.source(); small.thumbnail((1400, 1400))
        self.crop_view.set_preview(render_preview(small, self.bg_rgb, 0))

    def on_box_changed(self):
        v = int(round(self.base_h * 100 / self.box[2]))
        self.s_zoom.blockSignals(True); self.s_zoom.setValue(max(40, min(300, v))); self.s_zoom.blockSignals(False)
        self.s_tilt.blockSignals(True); self.s_tilt.setValue(max(-450, min(450, int(round(self.box[3] * 10))))); self.s_tilt.blockSignals(False)

    def on_zoom(self, v):
        if self.box: self.box[2] = self.base_h * 100 / v; self.crop_view.update()

    def on_tilt(self, v):
        if self.box: self.box[3] = v / 10.0; self.crop_view.update()

    def rot90(self, d):
        if self.box: self.box[3] = (self.box[3] + d + 180) % 360 - 180; self.crop_view.update(); self.on_box_changed()

    def nudge(self, dx, dy):
        if self.box:
            self.box[0] += dx * self.box[2] * 0.01; self.box[1] += dy * self.box[2] * 0.01; self.crop_view.update()

    def on_size_change(self, *_):
        self.crop_view.ratio = self.ratio(); self.crop_view.update()
        if self.photo_base is not None and self.stack.currentIndex() == 0: self.apply_crop()

    def on_enter(self, *_):
        i = self.stack.currentIndex()
        if i == 2: self.close_retouch()
        elif i == 1:
            pre = self._pre
            if pre is not None and pre["box"] != self.box:       # crop undo er jonno
                self.history.append(pre); self.history = self.history[-15:]; self.redo_stack = []
            self.apply_crop()

    def on_esc(self, *_):
        i = self.stack.currentIndex()
        if i == 2: self.close_retouch(commit=False); return
        if i == 1:
            pre = self._pre
            if pre is not None and pre["box"] is not None and self.box is not None:
                self.box[:] = pre["box"]; self.base_h = pre["base_h"]; self.on_box_changed()
            if self.photo is not None: self.show_img(self.photo)
            elif self.rgb is not None: self.show_img(self.composite())
            else: self.stack.setCurrentIndex(0)
            self.status.setText("Crop cancel")

    def apply_crop(self):
        if not self.need_img(): return
        self.ensure_box()
        self.photo_base = crop_to(self.source(), self.bg_rgb, SIZES[self.size_box.currentText()], self.box)
        self.sheet = None; self.showing_orig = False
        self.stack.setCurrentIndex(0)
        self.apply_adjust()
        self.status.setText("Crop holo. Thik na hole 'Crop Kholo' chapo")

    # ---------- final adjust ----------
    def adjust(self, im):
        b, c = self.s_bright.value(), self.s_contrast.value()
        if b: im = ImageEnhance.Brightness(im).enhance(1 + b / 100.0)
        if c: im = ImageEnhance.Contrast(im).enhance(1 + c / 100.0)
        if self.c_bw.isChecked(): im = ImageOps.grayscale(im).convert("RGB")
        if self.dress is not None: im = add_dress(im, self.dress, self.s_dress.value(), self.s_dress_y.value())
        txt = self.cap_edit.text().strip()
        if self.c_date.isChecked(): txt = (txt + "   " if txt else "") + datetime.date.today().strftime("%d-%m-%Y")
        return add_caption(im, txt)

    def apply_adjust(self, *_):
        if self.photo_base is None: return
        self.photo = self.adjust(self.photo_base); self.sheet = None
        if self.stack.currentIndex() == 0: self.show_img(self.photo)

    def toggle(self):
        if self.orig is None or self.stack.currentIndex() == 1: return
        self.showing_orig = not self.showing_orig
        self.show_img(self.orig if self.showing_orig else (self.photo if self.photo is not None else self.composite()))

    # ---------- output ----------
    def do_layout(self):
        if self.photo is None:
            return QMessageBox.warning(self, "Oops", "Age crop koro ('Crop Kholo' ba AUTO CHALAO)")
        items = [(self.photo, self.copies.value())]
        if self.c_mix.isChecked() and self.box is not None:
            b2 = crop_to(self.source(), self.bg_rgb, SIZES[self.size2_box.currentText()], self.box)
            items.append((self.adjust(b2), self.copies2.value()))
        self.sheet, placed, total = pack_sheet(items, PAPERS[self.paper_box.currentText()], border=self.c_border.isChecked())
        self.show_img(self.sheet)
        self.status.setText(f"{self.paper_box.currentText()} ready: {placed}/{total} ta photo boshlo")
        if placed < total:
            QMessageBox.information(self, "Jayga kom", f"Paper e {placed}/{total} ta photo dhoke. Copy kombao ba boro paper nao.")

    def save_single(self):
        if self.photo is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "photo.jpg", "JPG (*.jpg)")
        if p: self.photo.save(p, quality=95, dpi=(DPI, DPI)); self.status.setText("Saved")

    def save_sheet(self):
        if self.sheet is None: self.do_layout()
        if self.sheet is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "sheet.jpg", "JPG (*.jpg)")
        if p: self.sheet.save(p, quality=95, dpi=(DPI, DPI)); self.status.setText("Saved")

    def save_pdf(self):
        if self.sheet is None: self.do_layout()
        if self.sheet is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Save PDF", "sheet.pdf", "PDF (*.pdf)")
        if p: self.sheet.save(p, "PDF", resolution=DPI); self.status.setText("PDF saved")

    def save_online_cb(self):
        if self.photo is None:
            return QMessageBox.warning(self, "Oops", "Age crop koro ('Crop Kholo' ba AUTO CHALAO)")
        p, _ = QFileDialog.getSaveFileName(self, "Save for Online", "online_photo.jpg", "JPG (*.jpg)")
        if not p: return
        sz = (self.sp_ow.value(), self.sp_oh.value())
        r = save_online(self.photo, p, self.sp_kb.value(), sz if sz[0] and sz[1] else None)
        if r is None: QMessageBox.warning(self, "Boro", "Ei KB e photo ana gelo na. KB beshi dao ba pixel size chhoto koro.")
        else: self.status.setText(f"Saved: {r[0] / 1024:.0f} KB, {r[1][0]}x{r[1][1]} px")

    def print_sheet(self):
        if self.sheet is None: self.do_layout()
        if self.sheet is None: return
        p = os.path.join(os.environ.get("TEMP", "."), "print_sheet.jpg")
        self.sheet.save(p, quality=95, dpi=(DPI, DPI))
        if hasattr(os, "startfile"): os.startfile(p, "print")
        else: QMessageBox.information(self, "Print", "Saved: " + p + "\nEta khule print koro.")



# ------------------- FIXED PROCESSING INDICATOR -------------------
class ProcessingBanner(QFrame):
    """Editor-er ekdom upore fixed processing indicator.
    Heavy job cholche kina chokhe dekha jay: spinner + moving progress bar + live message.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ProcessingBanner")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame#ProcessingBanner {
                background: #17212b;
                border: 1px solid #3d5366;
                border-radius: 8px;
            }
            QLabel#ProcessTitle { color: #ffffff; font-size: 14px; font-weight: 700; }
            QLabel#ProcessMessage { color: #d7e5ef; font-size: 13px; }
            QLabel#ProcessSpinner { color: #42a5f5; font-size: 18px; font-weight: 700; min-width: 24px; }
            QProgressBar {
                border: 0;
                background: #263746;
                border-radius: 3px;
                height: 5px;
            }
            QProgressBar::chunk { background: #42a5f5; border-radius: 3px; }
        """)
        self.spinner = QLabel("●")
        self.spinner.setObjectName("ProcessSpinner")
        self.title = QLabel("READY")
        self.title.setObjectName("ProcessTitle")
        self.message = QLabel("Photo Studio ready")
        self.message.setObjectName("ProcessMessage")
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # indeterminate / animated busy bar
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(5)
        self.bar.setVisible(False)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        text_col.addWidget(self.title)
        text_col.addWidget(self.message)

        top = QHBoxLayout()
        top.setContentsMargins(10, 7, 10, 3)
        top.setSpacing(8)
        top.addWidget(self.spinner)
        top.addLayout(text_col, 1)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().addLayout(top)
        self.layout().addWidget(self.bar)

        self._timer = QTimer(self)
        self._timer.setInterval(140)
        self._timer.timeout.connect(self._tick)
        self._frames = ["●", "●", "●", "●"]
        self._frame = 0
        self.setVisible(False)

    def _tick(self):
        # Subtle spinner animation; no GIF/image overhead.
        self._frame = (self._frame + 1) % len(self._frames)
        dots = "." * (self._frame + 1)
        self.spinner.setText("◉" if self._frame % 2 else "●")
        self.title.setText("PROCESSING" + dots)

    def start(self, message):
        self.message.setText(message or "Processing...")
        self.title.setText("PROCESSING...")
        self.bar.setVisible(True)
        self.setVisible(True)
        self._frame = 0
        self._timer.start()
        self.raise_()

    def update_message(self, message):
        if message:
            self.message.setText(message)
        if not self.isVisible():
            self.setVisible(True)
            self.bar.setVisible(True)
            self._timer.start()

    def stop(self, message="Ready"):
        self._timer.stop()
        self.spinner.setText("✓")
        self.title.setText("DONE")
        self.message.setText(message)
        self.bar.setVisible(False)
        # Keep it visible briefly so the user sees the result, then hide.
        QTimer.singleShot(900, self.hide)

# ------------------- APP V2 -------------------
class AppV2(App):
    """Performance-focused V2 built directly on the original App feature set.

    Design rules:
      * Qt thread = UI only.
      * One persistent worker = AI/large CPU jobs.
      * AI models are warmed up at startup.
      * Preview is generated at display resolution, never from a full-size QPixmap.
      * Final export still uses full resolution.
      * Adjustment sliders are debounced instead of recalculating every mouse tick.
      * Expensive masks are cached per working-image revision.
    """
    PREVIEW_MAX = 1100
    ADJUST_DEBOUNCE_MS = 70

    def __init__(self):
        self._v2_revision = 0
        self._cache = {}
        self._job_active = False
        self._pending_adjust = False
        self._adjust_timer = QTimer()
        self._adjust_timer.setSingleShot(True)
        self._adjust_timer.setInterval(self.ADJUST_DEBOUNCE_MS)
        self._adjust_timer.timeout.connect(self._apply_adjust_now)
        self._last_display_key = None
        self._last_display_pixmap = None
        super().__init__()

        # Fixed top-of-editor processing banner. The old status label remains as
        # a permanent history/status line, but active processing is always visible.
        self.process_banner = ProcessingBanner(self)
        self.layout().insertWidget(0, self.process_banner)

        self._ai_thread = QThread(self)
        self._ai_worker = AIWorker()
        self._ai_worker.moveToThread(self._ai_thread)
        self._ai_worker.finished.connect(self._job_finished)
        self._ai_worker.failed.connect(self._job_failed)
        self._ai_worker.progress.connect(self._show_progress)
        self._ai_thread.started.connect(lambda: self._ai_worker.request.emit("preload", {
            "model": MODELS[self.model_box.currentText()]
        }))
        self._ai_thread.start()
        self.status.setText("AI startup: loading models in background... UI ready.")

    # ---------- cache / state ----------
    def _invalidate_cache(self):
        self._v2_revision += 1
        self._cache.clear()
        self._last_display_key = None
        self._last_display_pixmap = None

    def _set_working_rgb(self, arr_or_img):
        self.rgb = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(np.ascontiguousarray(arr_or_img))
        self._invalidate_cache()

    def _display_size(self):
        w = max(320, self.preview.width() - 12)
        h = max(320, self.preview.height() - 12)
        return min(self.PREVIEW_MAX, w), min(self.PREVIEW_MAX, h)

    def show_img(self, im):
        if im is None: return
        self.stack.setCurrentIndex(0)
        max_w, max_h = self._display_size()
        key = (id(im), im.size, max_w, max_h)
        if key == self._last_display_key and self._last_display_pixmap is not None:
            self.preview.setPixmap(self._last_display_pixmap)
            return
        # Resize PIL first. Converting a 4000-6000px image to QImage on every
        # preview update was one of the biggest hidden UI costs in V1.
        if im.width > max_w or im.height > max_h:
            view = im.copy()
            view.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        else:
            view = im
        pm = pil_to_pixmap(view)
        self._last_display_key = key
        self._last_display_pixmap = pm
        self.preview.setPixmap(pm)

    def source(self):
        im = self.rgb.convert("RGBA")
        if self.alpha is not None:
            im.putalpha(self.alpha)
        return im

    def composite(self):
        if self.alpha is None: return self.rgb
        c = Image.new("RGB", self.rgb.size, self.bg_rgb)
        c.paste(self.rgb, (0, 0), self.alpha)
        return c

    def after_change(self):
        if self.rgb is None: return
        self._invalidate_cache()
        if self.stack.currentIndex() == 1:
            self.refresh_preview()
        elif self.photo_base is not None:
            self.apply_crop()
        else:
            self.show_img(self.composite())

    def push(self):
        # The image objects are immutable from the editor's point of view: every
        # edit assigns a new PIL image/array. Therefore history does not need an
        # immediate deep copy of every 4096px frame.
        self.history.append(self._snap())
        self.history = self.history[-12:]
        self.redo_stack = []

    def _snap(self):
        return dict(rgb=self.rgb, alpha=self.alpha, orig=self.orig, face=self.face,
                    box=list(self.box) if self.box else None, base_h=self.base_h,
                    cropped=self.photo_base is not None)

    # ---------- non-blocking jobs ----------
    def _show_progress(self, message):
        self.status.setText(message)
        if hasattr(self, "process_banner"):
            self.process_banner.update_message(message)

    def _submit(self, kind, payload, message=None, history=False):
        if self._job_active:
            self.status.setText("Ekta processing already cholche. Oita shesh hole abar chapo.")
            return False
        if history:
            self.push()
        self._job_active = True
        msg = message or "Processing..."
        self.status.setText(msg)
        self.process_banner.start(msg)
        self._ai_worker.request.emit(kind, payload)
        return True

    def _job_finished(self, kind, result):
        self._job_active = False
        try:
            if kind == "preload":
                self.status.setText("AI Ready — models warm. Photo Studio is ready.")
                self.process_banner.stop("AI models ready — Photo Studio is ready")
                return
            if kind == "bg":
                self.alpha = result
                self._invalidate_cache(); self.after_change()
                self.status.setText("Background remove holo.")
                self.process_banner.stop("Background remove complete")
                return
            if kind == "face":
                self.push(); self._set_working_rgb(result)
                self.after_change(); self.status.setText("AI Face Restore holo.")
                self.process_banner.stop("AI Face Restore complete")
                return
            if kind == "upscale":
                big, how = result
                old_size = self.rgb.size
                self.push()
                k = big.width / max(1, old_size[0])
                if self.orig is not None:
                    self.orig = self.orig.resize(big.size, Image.Resampling.LANCZOS)
                if self.alpha is not None:
                    self.alpha = self.alpha.resize(big.size, Image.Resampling.LANCZOS)
                self._set_working_rgb(big)
                if self.face is not None: self.face = tuple(int(v * k) for v in self.face)
                if self.box is not None:
                    self.box = [self.box[0] * k, self.box[1] * k, self.box[2] * k, self.box[3]]
                    self.base_h *= k
                self.hair_mask = None
                self.after_change(); self.status.setText("Upscale holo: " + how)
                self.process_banner.stop("Upscale complete — " + how)
                return
            if kind == "auto":
                arr, alpha, hm, notes = result
                self._set_working_rgb(arr)
                self.alpha = alpha; self.hair_mask = hm
                self.box = auto_box(self.rgb, self.face, self.ratio(), self.c_crop.isChecked())
                self.base_h = self.box[2]
                self.s_zoom.blockSignals(True); self.s_zoom.setValue(100); self.s_zoom.blockSignals(False)
                self.s_tilt.blockSignals(True); self.s_tilt.setValue(0); self.s_tilt.blockSignals(False)
                if self.c_manual.isChecked() or (self.face is None and self.c_crop.isChecked()):
                    self.open_editor()
                    if self.face is None: self.status.setText("Face paini: manual box sorao, tarpor Enter chapo")
                else:
                    self.apply_crop()
                if notes: self.status.setText(self.status.text() + "   [" + " | ".join(notes) + "]")
                self.process_banner.stop("Auto processing complete")
                return
            if kind == "hairmask":
                self.hair_mask = result
                if self.stack.currentIndex() == 2:
                    self.paint_view.set_overlay(self.hair_mask)
                self.status.setText("Hair area ready.")
                self.process_banner.stop("Hair area detection complete")
                return
            if kind in ("enhance", "smooth", "tone", "retouch", "shine", "hair"):
                self._set_working_rgb(result)
                self.after_change()
                labels = {"enhance":"Photo Enhance PRO holo", "smooth":"Skin smooth holo",
                          "tone":"Skin tone up holo", "retouch":"Skin retouch holo",
                          "shine":"Oily shine komano holo", "hair":"Hair color holo"}
                self.status.setText(labels[kind])
                self.process_banner.stop("Operation complete")
                return
            if kind == "pimple":
                arr, n = result
                self._set_working_rgb(arr); self.after_change()
                self.status.setText(f"{n} ta pimple/dag remove holo.")
                self.process_banner.stop(f"{n} ta pimple/dag remove complete")
        except Exception as e:
            self._job_failed(kind, str(e))

    def _job_failed(self, kind, error):
        self._job_active = False
        if self.history and kind in ("bg", "face", "upscale", "auto", "enhance", "smooth", "tone", "retouch", "shine", "pimple", "hair"):
            # Failed jobs produced no new state, so discard their pending history snapshot.
            self.history.pop()
        self.status.setText("Processing failed")
        if hasattr(self, "process_banner"):
            self.process_banner.stop("Processing failed — details below")
        QMessageBox.warning(self, "Ripon Computer", f"{kind} hoyni.\n\n{error[:1000]}")

    # ---------- open / masks ----------
    def open_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Photo", "", "Images (*.jpg *.jpeg *.png *.webp)")
        if not p: return
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            if max(im.size) > 4096:
                im.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        except Exception as e:
            QMessageBox.critical(self, "Open", str(e)); return
        self.orig = im; self.rgb = im; self.alpha = None
        self.history = []; self.redo_stack = []; self.hair_mask = None
        self.face = detect_face(np.array(im))
        self.box = None; self.photo_base = self.photo = self.sheet = None
        self._invalidate_cache(); self.show_img(im)
        self.status.setText("Photo loaded. " + ("Face detect holo." if self.face else "Face paini."))

    def _cached_skin_mask(self):
        key = ("skin", self._v2_revision, self.face)
        if key not in self._cache:
            self._cache[key] = skin_mask(np.array(self.rgb), self.face)
        return self._cache[key]

    # ---------- async manual effects ----------
    def m_bg_remove(self):
        if not self.need_img(): return
        self._submit("bg", {"image": self.rgb.copy(), "model": MODELS[self.model_box.currentText()]},
                     "Background remove hochche... AI startup e preload hocche.", history=True)

    def m_bg_restore(self):
        if self.need_img():
            self.push(); self.alpha = None; self.after_change(); self.status.setText("Original background fire ashlo")

    def m_enhance(self):
        if self.need_img():
            self._submit("enhance", {"arr": np.array(self.rgb), "strength": self.s_enh.value()},
                         "Photo Enhance PRO...", history=True)

    def _need_face(self):
        if not self.need_img(): return False
        if self.face is None:
            QMessageBox.information(self, "Face paini", "Skin effect er jonno face detect lagbe.")
            return False
        return True

    def m_smooth(self):
        if self._need_face():
            self._submit("smooth", {"arr": np.array(self.rgb), "face": self.face, "strength": self.s_smooth.value()},
                         "Skin smooth...", history=True)

    def m_tone(self):
        if self._need_face():
            self._submit("tone", {"arr": np.array(self.rgb), "face": self.face, "strength": self.s_tone.value()},
                         "Skin tone...", history=True)

    def m_retouch(self):
        if self._need_face():
            self._submit("retouch", {"arr": np.array(self.rgb), "face": self.face, "strength": self.s_smooth.value()},
                         "Skin retouch...", history=True)

    def m_shine(self):
        if self._need_face():
            self._submit("shine", {"arr": np.array(self.rgb), "face": self.face, "strength": 70},
                         "Oily shine fix...", history=True)

    def m_pimple(self):
        if self._need_face():
            self._submit("pimple", {"arr": np.array(self.rgb), "face": self.face, "strength": self.s_pimple.value()},
                         "Pimple/dag khujchi...", history=True)

    def m_face_ai(self):
        if self.need_img():
            self._submit("face", {"arr": np.array(self.rgb), "weight": self.s_aiface.value() / 100.0},
                         "AI Face Restore...", history=True)

    def m_upscale(self):
        if not self.need_img(): return
        if max(self.rgb.size) >= 2400:
            self.status.setText("Photo already boro (2400px+), upscale dorkar nai"); return
        self._submit("upscale", {"image": self.rgb.copy()}, "AI Upscale 2x...", history=True)

    def hair_detect(self):
        if not self._need_face(): return False
        if self.hair_mask is not None and self.hair_mask.shape == (self.rgb.height, self.rgb.width):
            return True
        return self._submit("hairmask", {
            "arr": np.array(self.rgb), "face": self.face, "alpha": None if self.alpha is None else np.array(self.alpha),
            "model": MODELS[self.model_box.currentText()]
        }, "Chuler jaiga khujchi...")

    def m_hair(self, color=None):
        if not self.need_img(): return
        if self.hair_mask is None or self.hair_mask.shape != (self.rgb.height, self.rgb.width):
            if not self.hair_detect(): return
            self.status.setText("Hair mask ready hole abar Hair Color Apply chapo.")
            return
        if (self.hair_mask > 0.5).sum() < 200:
            QMessageBox.information(self, "Chul paini", "Chuler jaiga auto dhora jayni. Area Edit diye brush koro.")
            return
        self._submit("hair", {"arr": np.array(self.rgb), "mask": self.hair_mask.copy(),
                               "color": color or self.hair_rgb, "strength": self.s_hair.value()},
                     "Hair color...", history=True)

    # ---------- fast preview adjustment ----------
    def apply_adjust(self, *_):
        if self.photo_base is None: return
        self._pending_adjust = True
        self._adjust_timer.start()

    def _apply_adjust_now(self):
        self._pending_adjust = False
        if self.photo_base is None: return
        self.photo = self.adjust(self.photo_base)
        self.sheet = None
        if self.stack.currentIndex() == 0: self.show_img(self.photo)

    def refresh_preview(self):
        if self.rgb is None: return
        small = self.source()
        small.thumbnail((self.PREVIEW_MAX, self.PREVIEW_MAX), Image.Resampling.LANCZOS)
        self.crop_view.set_preview(render_preview(small, self.bg_rgb, 0))

    # ---------- async AUTO ----------
    def run_auto(self):
        if not self.need_img(): return
        if self._job_active:
            self.status.setText("Ekta processing already cholche."); return
        self.alpha = None
        settings = {
            "enh": self.c_enh.isChecked(), "enh_strength": self.s_enh.value(),
            "face_ai": self.c_face.isChecked(), "face_strength": self.s_aiface.value(),
            "pimple": self.c_pimple.isChecked(), "pimple_strength": self.s_pimple.value(),
            "smooth": self.c_smooth.isChecked(), "smooth_strength": self.s_smooth.value(),
            "tone": self.c_tone.isChecked(), "tone_strength": self.s_tone.value(),
            "shine": self.c_shine.isChecked(), "bg": self.c_bg.isChecked(),
            "hair": self.c_hair.isChecked(), "hair_strength": self.s_hair.value(),
        }
        ok = self._submit("auto", {"arr": np.array(self.orig if self.orig is not None else self.rgb),
                                     "face": self.face, "settings": settings,
                                     "model": MODELS[self.model_box.currentText()], "hair_rgb": self.hair_rgb},
                          "Auto processing... UI responsive thakbe.", history=True)

    # ---------- shutdown ----------
    def closeEvent(self, event):
        try:
            self._adjust_timer.stop()
            if hasattr(self, "_ai_thread"):
                self._ai_thread.quit()
                self._ai_thread.wait(3000)
        finally:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    w = AppV2(); w.show(); sys.exit(app.exec())
