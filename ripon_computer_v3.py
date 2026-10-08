# =====================================================================
#  RIPON COMPUTER  -  AI PHOTO STUDIO   (v3)
#  One-click AI passport photo, print studio, document studio,
#  customer jobs, shop accounts.  Built on top of Ripon Computer v2.
#
#  Run:      python ripon_computer_v3.py
#  Required: pip install PySide6 opencv-python numpy pillow rembg onnxruntime
#  Optional AI (everything works without these, just with simpler tools):
#      GPU background removal : pip uninstall onnxruntime && pip install onnxruntime-gpu
#      Face detect/landmarks  : pip install insightface
#      Face restore / upscale : pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#                               pip install gfpgan realesrgan basicsr
#  Models are stored in:   %LOCALAPPDATA%\RiponComputer\models   (see "AI Models" button)
#  Dress/Suit PNG: put transparent PNGs in the "dress" folder next to this file.
#                  Sub-folders Shirt / Suit / Tie / Coat become tabs automatically.
# =====================================================================
import sys, os, math, io, re, json, time, glob, shutil, sqlite3, queue, threading, datetime, traceback, urllib.request
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageFont, ImageFilter

# ---------------- app folders (created once, never inside Program Files) ----------------
def _app_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "RiponComputer")
    os.makedirs(d, exist_ok=True)
    return d

APP_NAME = "Ripon Computer"
APP_TITLE = "RIPON COMPUTER - AI PHOTO STUDIO"
APP_DIR = _app_dir()
MODELS_DIR = os.path.join(APP_DIR, "models")
JOBS_DIR = os.path.join(APP_DIR, "jobs")
RECOVERY_DIR = os.path.join(APP_DIR, "recovery")
DB_PATH = os.path.join(APP_DIR, "shop.db")
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
PRESETS_PATH = os.path.join(APP_DIR, "presets.json")
LOG_PATH = os.path.join(APP_DIR, "ripon.log")
for _d in (MODELS_DIR, JOBS_DIR, RECOVERY_DIR):
    os.makedirs(_d, exist_ok=True)

# All AI libraries must store their weights under MODELS_DIR (set BEFORE they are imported).
os.environ.setdefault("U2NET_HOME", os.path.join(MODELS_DIR, "rembg"))
os.environ.setdefault("TORCH_HOME", os.path.join(MODELS_DIR, "torch"))
os.makedirs(os.environ["U2NET_HOME"], exist_ok=True)

import logging
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ripon")

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                               QComboBox, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
                               QGroupBox, QScrollArea, QStackedWidget, QColorDialog, QMenuBar, QLineEdit,
                               QProgressBar, QFrame, QTabWidget, QDialog, QTableWidget, QTableWidgetItem,
                               QHeaderView, QListWidget, QListWidgetItem, QAbstractItemView, QFormLayout,
                               QDialogButtonBox, QInputDialog, QSizePolicy, QGridLayout, QTextEdit, QSplitter)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QPainterPath, QPolygonF, QIcon,
                           QShortcut, QKeySequence, QAction, QFont, QPageSize, QPageLayout)
from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, QObject, Signal, Slot, QTimer, QSize, QMarginsF
from PySide6.QtPrintSupport import QPrinter, QPrintDialog, QPrintPreviewDialog

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, "frozen", False) else __file__))
DRESS_DIR = os.path.join(SCRIPT_DIR, "dress")

# ---------------- settings (tiny JSON file) ----------------
_DEFAULT_SETTINGS = {"model": "u2net_human_seg", "warmup": True, "gpu": True, "last_dir": "",
                     "shop_name": "Ripon Computer", "autosave_sec": 30, "extra_save_dir": ""}

def load_settings():
    s = dict(_DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f: s.update(json.load(f))
    except Exception:
        pass
    return s

SETTINGS = load_settings()

def save_settings():
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f: json.dump(SETTINGS, f, indent=1)
    except Exception as e:
        log.warning("settings save failed: %s", e)

# ---------------- sizes / presets ----------------
# SIZES / BG_COLORS kept from v2 so old habits still work. Presets (below) extend them.
SIZES = {"Bangladesh Passport 35x45 mm": (35, 45), "Visa 40x50 mm": (40, 50),
         "Stamp 25x30 mm": (25, 30), "Small 20x25 mm": (20, 25)}
BG_COLORS = {"White": (255, 255, 255), "Light Blue": (173, 216, 230), "Blue": (70, 130, 200)}
DPI = 300                                      # current output DPI (a preset can change it)
PAPERS_MM = {"A4 (210x297 mm)": (210, 297), "A5 (148x210 mm)": (148, 210),
             "4R (4x6 inch)": (101.6, 152.4), "5R (5x7 inch)": (127, 177.8)}
CUSTOM_PAPER = "Custom paper..."
HAIR_COLORS = {"Black (Kalo)": (28, 24, 24), "Dark Brown": (62, 42, 32), "Brown": (110, 72, 44),
               "Golden Blonde": (190, 150, 84), "Grey": (140, 140, 142), "Silver White": (205, 205, 208),
               "Henna Red": (140, 52, 34)}
RT_TOOLS = ["heal", "hair_add", "hair_erase"]
SERVICES = ["Passport Photo", "Visa Photo", "NID/ID Photo", "Print", "Document", "Scan/PDF", "Online Form", "Other"]

def set_dpi(v):
    global DPI
    DPI = int(max(72, min(1200, v)))

def mm2px(mm, dpi=None):
    return int(round(mm / 25.4 * (dpi or DPI)))

def paper_px(name, custom_mm=None, dpi=None):
    mm = custom_mm if name == CUSTOM_PAPER and custom_mm else PAPERS_MM.get(name, PAPERS_MM["A4 (210x297 mm)"])
    return mm2px(mm[0], dpi), mm2px(mm[1], dpi)

# A preset describes EVERYTHING the operator normally picks by hand.
#   head  = chin-to-crown height as a fraction of photo height
#   top   = gap above the crown as a fraction of photo height
BUILTIN_PRESETS = {
    "🇧🇩 Bangladesh Passport": dict(w=35, h=45, dpi=300, bg=[255, 255, 255], copies=6, paper="4R (4x6 inch)",
                                   head=0.74, top=0.09, enhance=50, retouch=40, builtin=True, service="Passport Photo"),
    "Visa Photo":              dict(w=40, h=50, dpi=300, bg=[255, 255, 255], copies=4, paper="4R (4x6 inch)",
                                   head=0.72, top=0.09, enhance=50, retouch=40, builtin=True, service="Visa Photo"),
    "NID Photo":               dict(w=35, h=45, dpi=300, bg=[255, 255, 255], copies=4, paper="4R (4x6 inch)",
                                   head=0.72, top=0.10, enhance=45, retouch=35, builtin=True, service="NID/ID Photo"),
    "Job Application Photo":   dict(w=35, h=45, dpi=300, bg=[173, 216, 230], copies=4, paper="A4 (210x297 mm)",
                                   head=0.66, top=0.10, enhance=55, retouch=45, builtin=True, service="Passport Photo"),
    "CV Photo":                dict(w=40, h=50, dpi=300, bg=[173, 216, 230], copies=2, paper="A4 (210x297 mm)",
                                   head=0.62, top=0.11, enhance=55, retouch=45, builtin=True, service="Passport Photo"),
    "School/College Photo":    dict(w=30, h=40, dpi=300, bg=[70, 130, 200], copies=8, paper="4R (4x6 inch)",
                                   head=0.68, top=0.10, enhance=45, retouch=35, builtin=True, service="Passport Photo"),
    "Stamp 25x30 mm":          dict(w=25, h=30, dpi=300, bg=[255, 255, 255], copies=12, paper="4R (4x6 inch)",
                                   head=0.70, top=0.10, enhance=45, retouch=35, builtin=True, service="Passport Photo"),
    "Small 20x25 mm":          dict(w=20, h=25, dpi=300, bg=[255, 255, 255], copies=16, paper="4R (4x6 inch)",
                                   head=0.70, top=0.10, enhance=45, retouch=35, builtin=True, service="Passport Photo"),
}
CUSTOM_PRESET_NAME = "Custom size..."

def load_presets():
    """built-in presets + operator's saved presets (presets.json). Built-ins can be edited; edits are stored as overrides."""
    out = {k: dict(v) for k, v in BUILTIN_PRESETS.items()}
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as f: saved = json.load(f)
        for k, v in saved.items():
            base = out.get(k, {})
            merged = dict(base); merged.update(v)
            out[k] = merged
    except Exception:
        pass
    return out

def store_presets(presets):
    """only user-made or edited presets are saved; built-ins that are unchanged are skipped."""
    save = {}
    for k, v in presets.items():
        if k in BUILTIN_PRESETS:
            b = BUILTIN_PRESETS[k]
            if any(v.get(x) != b.get(x) for x in b if x != "builtin"): save[k] = {x: y for x, y in v.items() if x != "builtin"}
        else:
            save[k] = {x: y for x, y in v.items() if x != "builtin"}
    with open(PRESETS_PATH, "w", encoding="utf-8") as f: json.dump(save, f, indent=1, ensure_ascii=False)

def safe_name(s, fallback="Customer"):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", str(s or "")).strip().strip(".")
    s = re.sub(r"\s+", "_", s)
    return s[:40] or fallback

def unique_path(p):
    """never overwrite an existing file: name.jpg -> name_2.jpg"""
    if not os.path.exists(p): return p
    root, ext = os.path.splitext(p); i = 2
    while os.path.exists("%s_%d%s" % (root, i, ext)): i += 1
    return "%s_%d%s" % (root, i, ext)


# =====================================================================
#  AI MODEL HUB  (modular, optional, cached, GPU first with CPU fallback)
# =====================================================================
import gc, collections

def _shim_torchvision():
    """basicsr/gfpgan import torchvision.transforms.functional_tensor which new torchvision removed."""
    try:
        import torchvision.transforms.functional_tensor  # noqa
    except Exception:
        try:
            import torchvision.transforms.functional as F
            sys.modules["torchvision.transforms.functional_tensor"] = F
        except Exception:
            pass

MODEL_REGISTRY = collections.OrderedDict([
    ("face_detect", dict(label="Face + Eye Detect (InsightFace / SCRFD)", size="~280 MB", kind="insightface",
                         note="Best landmarks. InsightFace weights are free for non-commercial use only.")),
    ("yunet", dict(label="Face + Eye Detect (OpenCV YuNet, small)", size="~230 KB", kind="file",
                   file=os.path.join("yunet", "face_detection_yunet_2023mar.onnx"),
                   url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                   note="Light and commercial-friendly. Used when InsightFace is missing.")),
    ("u2net_human_seg", dict(label="Background Remove: Fast (u2net human)", size="~170 MB", kind="rembg",
                             note="Fast. Good enough for most photos.")),
    ("birefnet-portrait", dict(label="Background Remove: Best hair (BiRefNet portrait)", size="~970 MB", kind="rembg",
                               note="Best hair edges. Needs more GPU memory.")),
    ("bria-rmbg", dict(label="Background Remove: RMBG-2.0", size="~1 GB", kind="rembg",
                       note="Very good. Licence is non-commercial - check before using in a shop.")),
    ("gfpgan", dict(label="Face Restore (GFPGAN 1.4)", size="~350 MB", kind="file",
                    file=os.path.join("gfpgan", "GFPGANv1.4.pth"),
                    url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                    note="Only for blurry / low-quality faces. Needs torch + gfpgan.")),
    ("realesrgan", dict(label="Upscale (Real-ESRGAN x4)", size="~64 MB", kind="file",
                        file=os.path.join("realesrgan", "RealESRGAN_x4plus.pth"),
                        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                        note="Only for small photos. Needs torch + realesrgan.")),
])
BG_MODEL_KEYS = ["u2net_human_seg", "birefnet-portrait", "bria-rmbg"]
MODELS = {   # v2 names -> rembg ids (kept so old code/menus still work)
    "u2net_human_seg (fast, halka)": "u2net_human_seg",
    "birefnet-portrait (chul e best)": "birefnet-portrait",
    "bria-rmbg (RMBG-2.0)": "bria-rmbg",
}

class Cancelled(Exception):
    pass

class ModelHub:
    """Loads each model once, keeps at most MAX_LOADED in memory (LRU), reports GPU/CPU status.
    Nothing is ever downloaded unless the operator presses Download (or a worker asks explicitly)."""
    MAX_LOADED = 4

    def __init__(self):
        self.lock = threading.RLock()
        self.cache = collections.OrderedDict()
        self.errors = {}
        self.notes = []                    # short messages about CPU fallbacks etc. (read by UI)
        self._dev = None

    # ---------- device ----------
    def device_info(self, refresh=False):
        if self._dev is not None and not refresh: return self._dev
        d = dict(cuda=False, gpu="", torch=False, ort_gpu=False, providers=["CPUExecutionProvider"])
        try:
            import onnxruntime as ort
            av = ort.get_available_providers()
            d["ort_gpu"] = "CUDAExecutionProvider" in av
            if d["ort_gpu"]: d["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        try:
            import torch
            d["torch"] = True
            d["cuda"] = bool(torch.cuda.is_available())
            if d["cuda"]: d["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        self._dev = d
        return d

    def use_gpu(self):
        return bool(SETTINGS.get("gpu", True))

    def torch_device(self):
        return "cuda" if (self.use_gpu() and self.device_info()["cuda"]) else "cpu"

    def ort_providers(self):
        return self.device_info()["providers"] if (self.use_gpu() and self.device_info()["ort_gpu"]) else ["CPUExecutionProvider"]

    # ---------- install state ----------
    def path_of(self, key):
        r = MODEL_REGISTRY[key]
        return os.path.join(MODELS_DIR, r["file"]) if r["kind"] == "file" else None

    def is_installed(self, key):
        r = MODEL_REGISTRY[key]
        if r["kind"] == "file": return os.path.isfile(self.path_of(key)) and os.path.getsize(self.path_of(key)) > 1000
        if r["kind"] == "rembg": return os.path.isfile(os.path.join(os.environ["U2NET_HOME"], key + ".onnx"))
        if r["kind"] == "insightface":
            return os.path.isfile(os.path.join(MODELS_DIR, "insightface", "models", "buffalo_l", "det_10g.onnx"))
        return False

    def lib_available(self, key):
        """is the python package for this model importable? (cheap check, no heavy import)"""
        import importlib.util as iu
        r = MODEL_REGISTRY[key]
        need = {"insightface": ["insightface"], "rembg": ["rembg"]}.get(r["kind"], [])
        if key == "gfpgan": need = ["torch", "gfpgan"]
        if key == "realesrgan": need = ["torch", "realesrgan", "basicsr"]
        if key == "yunet": return hasattr(cv2, "FaceDetectorYN")
        try: return all(iu.find_spec(n) is not None for n in need)
        except Exception: return False

    def status_rows(self):
        dev = self.device_info(); rows = []
        for k, r in MODEL_REGISTRY.items():
            inst = self.is_installed(k); lib = self.lib_available(k)
            with self.lock: loaded = k in self.cache
            if not lib: st = "Package missing"
            elif not inst: st = "Not downloaded"
            elif loaded: st = "Loaded"
            else: st = "Installed"
            where = "-"
            if loaded:
                where = "GPU" if getattr(self.cache[k], "_ripon_gpu", False) else "CPU"
            rows.append(dict(key=k, label=r["label"], status=st, installed=inst, lib=lib, loaded=loaded,
                             where=where, size=r["size"], note=r["note"], err=self.errors.get(k, "")))
        return rows, dev

    # ---------- download ----------
    def download(self, key, progress=None):
        """progress(msg, frac_or_None). Raises on failure."""
        r = MODEL_REGISTRY[key]
        if r["kind"] == "file":
            os.makedirs(os.path.dirname(self.path_of(key)), exist_ok=True)
            tmp = self.path_of(key) + ".part"
            req = urllib.request.Request(r["url"], headers={"User-Agent": "RiponComputer/3"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0); got = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk: break
                    out.write(chunk); got += len(chunk)
                    if progress: progress("Downloading %s ..." % r["label"], (got / total) if total else None)
            os.replace(tmp, self.path_of(key))
        elif r["kind"] == "rembg":
            if progress: progress("Downloading %s (no % available) ..." % r["label"], None)
            self._rembg_session(key, force_cpu=True, load_only=True)
        elif r["kind"] == "insightface":
            if progress: progress("Downloading InsightFace models ...", None)
            self._insight(force_cpu=True)
        self.errors.pop(key, None)

    def unload(self, key=None):
        with self.lock:
            keys = [key] if key else list(self.cache.keys())
            for k in keys: self.cache.pop(k, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass

    def _remember(self, key, obj, gpu):
        try: obj._ripon_gpu = gpu
        except Exception: pass
        with self.lock:
            self.cache[key] = obj; self.cache.move_to_end(key)
            while len(self.cache) > self.MAX_LOADED:
                self.cache.popitem(last=False)
        gc.collect()

    # ---------- loaders ----------
    def _insight(self, force_cpu=False):
        with self.lock:
            if "face_detect" in self.cache and not force_cpu:
                self.cache.move_to_end("face_detect"); return self.cache["face_detect"]
            from insightface.app import FaceAnalysis
            gpu = self.use_gpu() and self.device_info()["ort_gpu"] and not force_cpu
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", root=os.path.join(MODELS_DIR, "insightface"),
                               allowed_modules=["detection", "landmark_2d_106"], providers=prov)
            app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
            self._remember("face_detect", app, gpu)
            return app

    def _yunet(self, w, h):
        det = cv2.FaceDetectorYN.create(self.path_of("yunet"), "", (w, h), 0.6, 0.3, 50)
        return det

    def _rembg_session(self, name, force_cpu=False, load_only=False):
        with self.lock:
            if name in self.cache and not force_cpu:
                self.cache.move_to_end(name); return self.cache[name]
            from rembg import new_session
            gpu = self.use_gpu() and self.device_info()["ort_gpu"] and not force_cpu
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
            try: sess = new_session(name, providers=prov)
            except TypeError: sess = new_session(name); gpu = False
            self._remember(name, sess, gpu)
            return sess

    def _gfpgan(self, force_cpu=False):
        with self.lock:
            if "gfpgan" in self.cache and not force_cpu:
                self.cache.move_to_end("gfpgan"); return self.cache["gfpgan"]
            _shim_torchvision()
            from gfpgan import GFPGANer
            dev = "cpu" if force_cpu else self.torch_device()
            g = GFPGANer(model_path=self.path_of("gfpgan"), upscale=1, arch="clean", channel_multiplier=2,
                         bg_upsampler=None, device=dev)
            self._remember("gfpgan", g, dev == "cuda")
            return g

    def _esrgan(self, force_cpu=False):
        with self.lock:
            if "realesrgan" in self.cache and not force_cpu:
                self.cache.move_to_end("realesrgan"); return self.cache["realesrgan"]
            _shim_torchvision()
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            dev = "cpu" if force_cpu else self.torch_device()
            m = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            e = RealESRGANer(scale=4, model_path=self.path_of("realesrgan"), model=m, tile=400, tile_pad=10,
                             pre_pad=0, half=(dev == "cuda"), gpu_id=0 if dev == "cuda" else None)
            self._remember("realesrgan", e, dev == "cuda")
            return e

    def warmup(self, bg_model, progress=None):
        """load only what is already installed; never downloads"""
        try:
            if self.is_installed("face_detect") and self.lib_available("face_detect"):
                if progress: progress("Starting AI: face detector...", None)
                self._insight()
        except Exception as e:
            self.errors["face_detect"] = str(e)[:200]
        try:
            if self.is_installed(bg_model) and self.lib_available(bg_model):
                if progress: progress("Starting AI: background model...", None)
                self._rembg_session(bg_model)
        except Exception as e:
            self.errors[bg_model] = str(e)[:200]

    # ---------- operations ----------
    def segment(self, rgb, model="u2net_human_seg", max_side=1280):
        """rgb uint8 HxWx3 -> float32 alpha HxW (0..1). GPU first, automatic CPU retry."""
        H, W = rgb.shape[:2]
        s = min(1.0, max_side / float(max(H, W)))
        small = cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
        from rembg import remove
        pil = Image.fromarray(small)
        try:
            sess = self._rembg_session(model)
            mask = remove(pil, session=sess, only_mask=True)
        except Exception as e:
            if getattr(self.cache.get(model), "_ripon_gpu", False) or self.ort_providers()[0] != "CPUExecutionProvider":
                self.notes.append("GPU background removal failed, used CPU (%s)" % (str(e).splitlines() or [""])[0][:60])
                log.warning("segment GPU fail: %s", e)
                self.unload(model)
                sess = self._rembg_session(model, force_cpu=True)
                mask = remove(pil, session=sess, only_mask=True)
            else:
                raise
        a = np.array(mask.convert("L"), np.float32) / 255.0
        if a.shape != (H, W): a = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
        return np.clip(a, 0, 1)

    def restore_face(self, rgb, weight=0.5):
        try:
            g = self._gfpgan()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, _, out = g.enhance(bgr, has_aligned=False, only_center_face=True, paste_back=True, weight=weight)
        except Exception as e:
            low = str(e).lower()
            if "out of memory" in low or "cuda" in low:          # GPU problem -> CPU retry
                self.notes.append("GPU face restore failed, used CPU")
                self.unload("gfpgan"); g = self._gfpgan(force_cpu=True)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                _, _, out = g.enhance(bgr, has_aligned=False, only_center_face=True, paste_back=True, weight=weight)
            else:
                raise
        if out is None: raise RuntimeError("Face not found for restore")
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def upscale(self, pil_img, outscale=2):
        """Real-ESRGAN when installed, otherwise Lanczos + sharpen. returns (PIL, how)"""
        if self.is_installed("realesrgan") and self.lib_available("realesrgan"):
            try:
                e = self._esrgan()
                out, _ = e.enhance(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), outscale=outscale)
                return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), "AI (Real-ESRGAN)"
            except Exception as ex:
                self.notes.append("AI Upscale unavailable, used normal upscale (%s)" % (str(ex).splitlines() or [""])[0][:50])
                log.warning("esrgan fail: %s", ex)
        big = pil_img.resize((pil_img.width * outscale, pil_img.height * outscale), Image.LANCZOS)
        a = np.array(big)
        a = cv2.addWeighted(a, 1.5, cv2.GaussianBlur(a, (0, 0), 1.6), -0.5, 0)
        return Image.fromarray(a), "Normal (Lanczos + sharpen)"

HUB = ModelHub()

# ---- v2-compatible wrappers (old code paths still call these names) ----
def remove_bg(pil_img, model="u2net_human_seg"):
    a = HUB.segment(np.array(pil_img.convert("RGB")), model)
    out = pil_img.convert("RGBA"); out.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    return out

def ai_face_restore(rgb, weight=0.5):
    if not HUB.is_installed("gfpgan") or not HUB.lib_available("gfpgan"):
        raise RuntimeError("AI Face Restore is not installed.\nOpen 'AI Models' and press Download for GFPGAN (needs: pip install torch gfpgan).")
    return HUB.restore_face(rgb, weight)

def upscale2x(pil_img):
    return HUB.upscale(pil_img, 2)

# =====================================================================
#  FACE ANALYSIS  (face box, eyes, nose, mouth, chin, crown, tilt)
# =====================================================================
_CASCADES = None
def _load_cascades():
    global _CASCADES
    if _CASCADES is None:
        _CASCADES = {}
        for key, fn in (("default", "haarcascade_frontalface_default.xml"), ("alt2", "haarcascade_frontalface_alt2.xml"),
                        ("alt", "haarcascade_frontalface_alt.xml"), ("profile", "haarcascade_profileface.xml"),
                        ("eye", "haarcascade_eye.xml")):
            try:
                c = cv2.CascadeClassifier(cv2.data.haarcascades + fn)
                if not c.empty(): _CASCADES[key] = c
            except Exception:
                pass
    return _CASCADES
def detect_face(rgb):
    """Face (x, y, w, h) image pixel e. Onek rokom cascade + halka ghuriye + profile try kore.
    Na pele None (tokhon user 'Face Manual' diye nijei box ankon korbe)."""
    try:
        cas = _load_cascades()
        if not cas: return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        s = min(1.0, 1200.0 / max(gray.shape))
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        g = cv2.equalizeHist(g)
        H, W = g.shape
        # (angle, cascades, scaleFactor, minNeighbors, minSize)
        attempts = [
            (0, ("default",), 1.1, 5, 60),
            (0, ("default", "alt2"), 1.05, 3, 40),
            (0, ("alt2", "alt", "default"), 1.04, 2, 30),
            (-15, ("default", "alt2"), 1.05, 3, 40), (15, ("default", "alt2"), 1.05, 3, 40),
            (-30, ("default", "alt2"), 1.05, 3, 40), (30, ("default", "alt2"), 1.05, 3, 40),
            (0, ("profile",), 1.05, 3, 40),
        ]
        for ang, names, sf, nb, ms in attempts:
            if ang:
                M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
                gr = cv2.warpAffine(g, M, (W, H)); inv = cv2.invertAffineTransform(M)
            else:
                gr = g; inv = None
            for nm in names:
                c = cas.get(nm)
                if c is None: continue
                for flip in ((False, True) if nm == "profile" else (False,)):
                    gg = cv2.flip(gr, 1) if flip else gr
                    faces = c.detectMultiScale(gg, sf, nb, minSize=(ms, ms))
                    if not len(faces): continue
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    if flip: x = W - (x + w)
                    if inv is not None:
                        cx, cy = x + w / 2.0, y + h / 2.0
                        cx, cy = inv[0, 0] * cx + inv[0, 1] * cy + inv[0, 2], inv[1, 0] * cx + inv[1, 1] * cy + inv[1, 2]
                        x, y = cx - w / 2.0, cy - h / 2.0
                    x, y = max(0.0, x), max(0.0, y)
                    w, h = min(w, W - x), min(h, H - y)
                    if w < 10 or h < 10: continue
                    return tuple(int(v / s) for v in (x, y, w, h))
    except Exception:
        pass
    return None


def _fi(bbox, le, re_, nose, ml, mr, chin, src, conf=1.0, est=False):
    """FaceInfo dict. Left/right = as seen in the image (le = eye on the left side of the picture)."""
    x, y, w, h = [float(v) for v in bbox]
    if le[0] > re_[0]: le, re_ = re_, le
    if ml[0] > mr[0]: ml, mr = mr, ml
    ang = math.degrees(math.atan2(re_[1] - le[1], re_[0] - le[0])) if not est else 0.0
    return dict(bbox=(x, y, w, h), le=tuple(map(float, le)), re=tuple(map(float, re_)), nose=tuple(map(float, nose)),
                ml=tuple(map(float, ml)), mr=tuple(map(float, mr)), chin=tuple(map(float, chin)),
                angle=ang, src=src, conf=float(conf), est=est, crown=None)

def _scale_fi(fi, k, dx=0.0, dy=0.0):
    """move/scale a FaceInfo into another image space: p' = p*k + (dx, dy)"""
    if fi is None: return None
    o = dict(fi)
    for key in ("le", "re", "nose", "ml", "mr", "chin", "crown"):
        if fi.get(key) is not None: o[key] = (fi[key][0] * k + dx, fi[key][1] * k + dy)
    x, y, w, h = fi["bbox"]; o["bbox"] = (x * k + dx, y * k + dy, w * k, h * k)
    return o

def fi_face_tuple(fi):
    """FaceInfo -> v2 style (x, y, w, h) ints that all old skin/pimple/hair tools use"""
    if fi is None: return None
    x, y, w, h = fi["bbox"]; return (int(max(0, x)), int(max(0, y)), int(w), int(h))

def _fi_from_haar(rgb):
    face = detect_face(rgb)
    if face is None: return None
    x, y, w, h = face
    le = re_ = None
    try:
        eye_c = _load_cascades().get("eye")
        if eye_c is not None:
            g = cv2.cvtColor(np.ascontiguousarray(rgb[y:y + int(h * 0.62), x:x + w]), cv2.COLOR_RGB2GRAY)
            eyes = eye_c.detectMultiScale(cv2.equalizeHist(g), 1.08, 4, minSize=(max(8, w // 10), max(8, w // 10)))
            cand = [(ex + ew / 2.0, ey + eh / 2.0, ew * eh) for ex, ey, ew, eh in eyes if ey + eh / 2.0 < h * 0.5]
            lefts = [c for c in cand if c[0] < w * 0.5]; rights = [c for c in cand if c[0] >= w * 0.5]
            if lefts and rights:
                a = max(lefts, key=lambda c: c[2]); b = max(rights, key=lambda c: c[2])
                if abs(a[1] - b[1]) < w * 0.25:
                    le = (x + a[0], y + a[1]); re_ = (x + b[0], y + b[1])
    except Exception:
        le = re_ = None
    est = le is None
    if est: le = (x + 0.30 * w, y + 0.40 * h); re_ = (x + 0.70 * w, y + 0.40 * h)
    return _fi((x, y, w, h), le, re_, (x + 0.5 * w, y + 0.62 * h), (x + 0.35 * w, y + 0.78 * h),
               (x + 0.65 * w, y + 0.78 * h), (x + 0.5 * w, y + 1.02 * h), "haar", 0.5, est)

def detect_face_info(rgb, hub=None):
    """Best detector available: InsightFace -> YuNet -> OpenCV Haar. Returns FaceInfo or None. Never raises."""
    hub = hub or HUB
    H, W = rgb.shape[:2]
    s = min(1.0, 1600.0 / max(H, W))
    small = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    best = None
    try:
        if hub.is_installed("face_detect") and hub.lib_available("face_detect"):
            app = hub._insight()
            faces = app.get(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x0, y0, x1, y1 = [float(v) for v in f.bbox]; k = f.kps
                chin = ((x0 + x1) / 2.0, y1 + 0.02 * (y1 - y0))
                lm = getattr(f, "landmark_2d_106", None)
                if lm is not None and len(lm) >= 33:
                    j = lm[0:33]; i = int(np.argmax(j[:, 1])); chin = (float(j[i, 0]), float(j[i, 1]))
                best = _fi((x0, y0, x1 - x0, y1 - y0), k[0], k[1], k[2], k[3], k[4], chin, "insightface", float(f.det_score))
    except Exception as e:
        log.warning("insightface detect failed: %s", e); hub.errors["face_detect"] = str(e)[:200]
    if best is None:
        try:
            if hub.is_installed("yunet") and hasattr(cv2, "FaceDetectorYN"):
                h2, w2 = small.shape[:2]
                det = hub._yunet(w2, h2); det.setInputSize((w2, h2))
                _, res = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
                if res is not None and len(res):
                    r = max(res, key=lambda r: r[2] * r[3])
                    x, y, w, h = [float(v) for v in r[:4]]
                    best = _fi((x, y, w, h), (r[4], r[5]), (r[6], r[7]), (r[8], r[9]), (r[10], r[11]), (r[12], r[13]),
                               (x + w / 2.0, y + h * 1.03), "yunet", float(r[14]))
        except Exception as e:
            log.warning("yunet detect failed: %s", e); hub.errors["yunet"] = str(e)[:200]
    if best is None:
        try: best = _fi_from_haar(small)
        except Exception as e: log.warning("haar detect failed: %s", e)
    if best is None: return None
    return _scale_fi(best, 1.0 / s) if s < 1 else best

# ---------------- head geometry in the head's own (rotated) frame ----------------
def _local(pt, origin, ang_deg):
    t = math.radians(ang_deg); dx, dy = pt[0] - origin[0], pt[1] - origin[1]
    return dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)

def _to_img(lx, ly, origin, ang_deg):
    t = math.radians(ang_deg)
    return origin[0] + lx * math.cos(t) - ly * math.sin(t), origin[1] + lx * math.sin(t) + ly * math.cos(t)

def crown_from_alpha(alpha, fi):
    """top of the head (incl. hair) from the person mask. alpha float 0..1. returns (x, y) or None"""
    try:
        x, y, w, h = fi["bbox"]; H, W = alpha.shape
        x0, x1 = max(0, int(x - 0.1 * w)), min(W, int(x + 1.1 * w))
        y1 = min(H, int(y + 0.6 * h))
        sub = alpha[:y1, x0:x1] > 0.5
        rows = np.where(sub.sum(1) > max(3, 0.04 * w))[0]
        if not len(rows): return None
        ty = int(rows[0]); cols = np.where(sub[ty])[0]
        return (x0 + float(cols.mean()) if len(cols) else x + w / 2.0, float(ty))
    except Exception:
        return None

def head_geometry(fi, alpha=None):
    """Returns dict(origin, angle, chin_ly, crown_ly, head_h, cx_l) in the head frame (origin = eye middle).
    chin_ly > 0 (below the eyes), crown_ly < 0 (above the eyes)."""
    E = ((fi["le"][0] + fi["re"][0]) / 2.0, (fi["le"][1] + fi["re"][1]) / 2.0)
    ang = fi["angle"]
    chin_l = _local(fi["chin"], E, ang)
    mouth = ((fi["ml"][0] + fi["mr"][0]) / 2.0, (fi["ml"][1] + fi["mr"][1]) / 2.0)
    mouth_l = _local(mouth, E, ang)
    eye_to_chin = max(chin_l[1], 1.0)
    head_h = eye_to_chin / 0.54                      # eyes sit ~46% down from the crown
    crown_ly = eye_to_chin - head_h                  # negative
    crown_pt = None
    if alpha is not None:
        crown_pt = crown_from_alpha(alpha, fi)
        if crown_pt is not None:
            cl = _local(crown_pt, E, ang)
            if crown_ly - 0.30 * head_h <= cl[1] <= crown_ly + 0.20 * head_h:    # sane? use the real hair top
                crown_ly = cl[1]; head_h = eye_to_chin - crown_ly
            else:
                crown_pt = None
    cx_l = (0.0 + mouth_l[0] + chin_l[0]) / 3.0
    return dict(origin=E, angle=ang, chin_ly=eye_to_chin, crown_ly=crown_ly, head_h=head_h, cx_l=cx_l, real_crown=crown_pt is not None)

def ideal_box(fi, preset, alpha=None, level=True):
    """Professional passport crop -> [cx, cy, box_height, angle] (v2 crop box format)."""
    g = head_geometry(fi, alpha)
    head, top = float(preset.get("head", 0.74)), float(preset.get("top", 0.09))
    bh = g["head_h"] / head
    top_ly = g["crown_ly"] - top * bh
    cx, cy = _to_img(g["cx_l"], top_ly + bh / 2.0, g["origin"], g["angle"])
    ang = g["angle"] if level else 0.0
    if not level:                                            # keep the same on-screen framing without rotating
        cx, cy = _to_img(g["cx_l"], top_ly + bh / 2.0, g["origin"], 0.0)
    return [float(cx), float(cy), float(bh), float(ang)]

def crop_metrics(fi, box, ratio, alpha=None):
    """where do head/eyes land inside the crop (0..1 of the crop)?  box = [cx, cy, h, ang]"""
    cx, cy, bh, ang = box; bw = bh * ratio
    def uv(p):
        lx, ly = _local(p, (cx, cy), ang); return 0.5 + lx / bw, 0.5 + ly / bh
    g = head_geometry(fi, alpha)
    crown_img = _to_img(g["cx_l"], g["crown_ly"], g["origin"], g["angle"])
    E = g["origin"]; chin = fi["chin"]
    mouth = ((fi["ml"][0] + fi["mr"][0]) / 2.0, (fi["ml"][1] + fi["mr"][1]) / 2.0)
    u_c = np.mean([uv(E)[0], uv(mouth)[0], uv(chin)[0]])
    return dict(crown_v=uv(crown_img)[1], chin_v=uv(chin)[1], eye_v=uv(E)[1], u_center=float(u_c),
                head_frac=uv(chin)[1] - uv(crown_img)[1], tilt=((fi["angle"] - ang + 180) % 360) - 180)

def passport_check(fi, box, ratio, preset, alpha=None):
    """List of (level, code, text). level: ok | warn | bad.  Used for the live 'Passport Position' chip."""
    if fi is None:
        return [("bad", "noface", "⚠ Face not found - use Face Manual")]
    m = crop_metrics(fi, box, ratio, alpha)
    head, top = float(preset.get("head", 0.74)), float(preset.get("top", 0.09))
    items = []
    tol_h = 0.08
    if m["head_frac"] > head + tol_h: items.append(("warn", "framing", "⚠ Head too large"))
    elif m["head_frac"] < head - tol_h - 0.04: items.append(("warn", "framing", "⚠ Head too small"))
    eye_target = top + head * 0.46
    if m["eye_v"] > eye_target + 0.07: items.append(("warn", "framing", "⚠ Face too low"))
    elif m["eye_v"] < eye_target - 0.07: items.append(("warn", "framing", "⚠ Face too high"))
    if m["u_center"] > 0.5 + 0.045: items.append(("warn", "framing", "⚠ Face too far right"))
    elif m["u_center"] < 0.5 - 0.045: items.append(("warn", "framing", "⚠ Face too far left"))
    if abs(m["tilt"]) > 2.5 and not fi.get("est"): items.append(("warn", "framing", "⚠ Face tilted"))
    if m["crown_v"] < 0.0: items.append(("bad", "framing", "⚠ Top of head cut off"))
    if m["chin_v"] > 1.0: items.append(("bad", "framing", "⚠ Chin cut off"))
    if not items:
        items.append(("ok", "ok", "✓ Passport Position OK"))
    return items

def shoulder_row(alpha, fi):
    """y of the shoulder line (first row under the chin where the person is ~1.7x face width). None if unknown"""
    try:
        x, y, w, h = fi["bbox"]; H, W = alpha.shape
        for yy in range(int(fi["chin"][1]), H, max(1, H // 300)):
            if (alpha[yy] > 0.5).sum() > 1.7 * w: return float(yy)
    except Exception:
        pass
    return None

# =====================================================================
#  IMAGE QUALITY HELPERS  (analysis first -> cheapest fix only when needed)
# =====================================================================
def analyze_image(rgb, fi=None):
    """quick numbers deciding which processing steps are really needed"""
    H, W = rgb.shape[:2]
    s = min(1.0, 700.0 / max(H, W))
    sm = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    g = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
    out = dict(mean=float(g.mean()), std=float(g.std()), blur=float(cv2.Laplacian(g, cv2.CV_64F).var()))
    if fi is not None:
        x, y, w, h = [int(v * s) for v in fi["bbox"]]
        fr = g[max(0, y):max(1, y + h), max(0, x):max(1, x + w)]
        if fr.size > 100:
            out["face_mean"] = float(fr.mean()); out["face_blur"] = float(cv2.Laplacian(cv2.resize(fr, (160, 160)), cv2.CV_64F).var())
    out["face_px"] = float(fi["bbox"][3]) if fi is not None else 0.0
    return out

def exposure_fix(rgb, fi, strength=1.0):
    """Gentle luminance-only correction so the face is neither dark nor washed out. Colour/skin tone untouched."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    x, y, w, h = [int(v) for v in fi["bbox"]] if fi is not None else (0, 0, rgb.shape[1], rgb.shape[0])
    reg = L[max(0, y + h // 4):max(1, y + 3 * h // 4), max(0, x + w // 5):max(1, x + 4 * w // 5)]
    if reg.size < 50: return rgb
    cur = float(np.median(reg))
    target = float(np.clip(cur, 118, 190))                     # only pull extremes towards a sane range
    if abs(target - cur) < 6: return rgb
    gamma = math.log(max(target, 1) / 255.0) / math.log(max(cur, 1) / 255.0)
    gamma = float(np.clip(1 + (gamma - 1) * strength, 0.65, 1.5))
    lab[..., 0] = np.clip(255.0 * (L / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def gentle_enhance(rgb, level=50, need_light=True):
    """natural clean-up: tiny denoise, local contrast, mild sharpen. level 0-100"""
    s = level / 100.0
    x = cv2.bilateralFilter(rgb, 5, 8 + 14 * s, 5)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.2 + 0.8 * s, tileGridSize=(8, 8)).apply(l)
    x = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(x, (0, 0), 1.3)
    return cv2.addWeighted(x, 1.0 + 0.35 * s, blur, -0.35 * s, 0)

def feature_guard(mask, fi):
    """take eyes / brows / lips / nostrils out of the skin mask so retouch never smears them"""
    if fi is None: return mask
    out = mask.copy(); w = fi["bbox"][2]
    m8 = np.zeros(mask.shape[:2], np.uint8)
    for p, r in ((fi["le"], 0.17), (fi["re"], 0.17)):
        cv2.ellipse(m8, (int(p[0]), int(p[1] - 0.03 * w)), (int(r * w * 1.1), int(r * w * 0.75)), 0, 0, 360, 255, -1)
    mc = ((fi["ml"][0] + fi["mr"][0]) / 2.0, (fi["ml"][1] + fi["mr"][1]) / 2.0)
    cv2.ellipse(m8, (int(mc[0]), int(mc[1])), (int(0.23 * w), int(0.08 * w)), 0, 0, 360, 255, -1)
    cv2.ellipse(m8, (int(fi["nose"][0]), int(fi["nose"][1])), (int(0.10 * w), int(0.05 * w)), 0, 0, 360, 255, -1)
    k = max(3, int(w * 0.06) | 1)
    m8 = cv2.GaussianBlur(m8, (k, k), 0).astype(np.float32) / 255.0
    return out * (1.0 - m8[..., None]) if out.ndim == 3 else out * (1.0 - m8)

# =====================================================================
#  BACKGROUND EDGE QUALITY  (hair edge refine, halo removal)
# =====================================================================
def _guided(I, p, r, eps):
    """fast guided filter (He et al.). I, p float32 HxW in 0..1"""
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k); mp = cv2.boxFilter(p, -1, k)
    cIp = cv2.boxFilter(I * p, -1, k) - mI * mp
    vI = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cIp / (vI + eps); b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)

def refine_alpha(rgb, alpha, strength=60, keep_face=None):
    """Clean the cut-out: edge-aware smoothing that follows hair strands, stray-blob removal,
    small-hole fill and a sub-pixel shrink so no old-background halo is left. alpha float 0..1"""
    H, W = alpha.shape
    s = strength / 100.0
    a = np.clip(alpha, 0, 1).astype(np.float32)
    # 1. keep the main person: component that touches the face (or the largest one)
    m8 = (a > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n > 2:
        keep = None
        if keep_face is not None:
            fx, fy = int(np.clip(keep_face[0], 0, W - 1)), int(np.clip(keep_face[1], 0, H - 1))
            if lab[fy, fx] > 0: keep = lab[fy, fx]
        if keep is None: keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        big = st[keep, cv2.CC_STAT_AREA]
        ok = np.zeros(n, bool); ok[keep] = True
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] > 0.25 * big: ok[i] = True       # e.g. a hair bun separated by a gap
        valid = ok[lab].astype(np.float32)
        valid = cv2.dilate(valid, np.ones((9, 9), np.uint8))
        a = a * valid
    # 2. fill tiny holes inside the person
    inv = (a < 0.5).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n2 > 1:
        border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
        small = 0.004 * H * W
        for i in range(1, n2):
            if i not in border and st2[i, cv2.CC_STAT_AREA] < small: a[lab2 == i] = 1.0
    # 3. guided filter (colour image steers the edge -> hair strands stay sharp)
    r = max(2, int(round(min(H, W) / 400 * (2 + 4 * s))))
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    a = np.clip(_guided(guide, a, r, 1e-3 + 4e-3 * (1 - s)), 0, 1)
    # 4. tidy: kill faint haze, solidify core, tiny shrink against halos
    a = np.where(a < 0.04, 0.0, a); a = np.where(a > 0.97, 1.0, a)
    if s > 0.3:
        k = max(1, int(round(min(H, W) / 1500 * (s * 2))))
        if k: a = cv2.erode(a, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        a = cv2.GaussianBlur(a, (0, 0), 0.6 + 0.6 * s)
    return np.clip(a, 0, 1).astype(np.float32)

def decontaminate(rgb, alpha):
    """remove coloured fringe (old background colour) on semi-transparent edge pixels by pulling in the nearest solid-person colour"""
    a = alpha
    solid = (a > 0.96).astype(np.uint8)
    if solid.sum() < 100: return rgb
    edge = (a > 0.02) & (a < 0.96)
    if not edge.any(): return rgb
    _, labels = cv2.distanceTransformWithLabels(1 - solid, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(solid > 0)
    lut = np.zeros((labels.max() + 1, 3), np.uint8)
    lab_of_solid = labels[ys, xs]
    lut[lab_of_solid] = rgb[ys, xs]
    near = lut[labels]
    k = cv2.GaussianBlur(near, (0, 0), 1.0)
    w = np.clip((1.0 - a) * 1.2, 0, 0.85)[..., None] * edge[..., None]
    return np.clip(rgb.astype(np.float32) * (1 - w) + k.astype(np.float32) * w, 0, 255).astype(np.uint8)

def bg_uniformity(rgb_img):
    """std-dev of the 4 border strips of a final photo. low = uniform. returns float"""
    a = np.array(rgb_img.convert("RGB").resize((200, 250))); lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    strips = [lab[:12, :], lab[:, :10], lab[:, -10:]]
    return float(np.mean([s.reshape(-1, 3).std(0).mean() for s in strips]))


# =====================================================================
#  PHOTO TOOLS (from v2, unchanged behaviour)  - skin / enhance / pimple / hair
# =====================================================================
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


def add_dress(im, dress, scale, yoff, xoff=0):
    """dress = transparent PNG (RGBA). scale = photo width %, yoff = up/down (% of height), xoff = left/right (% of width)"""
    W, H = im.size
    dw = max(1, int(W * scale / 100.0)); dh = max(1, int(dress.height * dw / dress.width))
    d = dress.resize((dw, dh), Image.LANCZOS)
    out = im.convert("RGBA")
    out.paste(d, ((W - dw) // 2 + int(W * xoff / 100.0), H - dh + int(H * yoff / 100.0)), d)
    return out.convert("RGB")

# =====================================================================
#  PRINT STUDIO  (automatic best layout, cut lines, multi-sheet)
# =====================================================================
def grid_fit(pw, ph, PW, PH, gap, mg):
    """how many photos of size pw x ph fit on a PW x PH sheet?  returns (cols, rows)"""
    cols = max(0, (PW - 2 * mg + gap) // (pw + gap)); rows = max(0, (PH - 2 * mg + gap) // (ph + gap))
    return int(cols), int(rows)

def best_layout(pw, ph, paper, gap, mg):
    """try both paper orientations and both photo orientations; keep the one that fits the most photos.
    returns dict(PW, PH, rotate, cols, rows, cap)"""
    best = None
    for PW, PH in (paper, (paper[1], paper[0])):
        for rot in (False, True):
            w, h = (ph, pw) if rot else (pw, ph)
            c, r = grid_fit(w, h, PW, PH, gap, mg)
            cand = dict(PW=PW, PH=PH, rotate=rot, cols=c, rows=r, cap=c * r)
            key = (cand["cap"], not rot, PH >= PW)          # more photos, then no rotation, then portrait paper
            if best is None or key > best[0]: best = (key, cand)
    return best[1]

def draw_cut_marks(d, x, y, w, h, style):
    if style == "Light border":
        d.rectangle([x, y, x + w - 1, y + h - 1], outline=(190, 190, 190))
    elif style == "Cut marks":
        L = max(6, mm2px(2.5)); col = (120, 120, 120)
        for cx, cy, sx, sy in ((x, y, -1, -1), (x + w, y, 1, -1), (x, y + h, -1, 1), (x + w, y + h, 1, 1)):
            d.line([cx, cy, cx + sx * L, cy], fill=col, width=1); d.line([cx, cy, cx, cy + sy * L], fill=col, width=1)

def pack_sheet(items, paper, gap_mm=3, margin_mm=4, border=True, style=None, orient="Auto", center=True):
    """items = [(photo, copies), ...]. Single size: automatic best grid (both orientations, photo rotation).
    Mixed sizes: row packing. return (sheet, placed, total)  - old v2 signature still works."""
    if style is None: style = "Light border" if border else "None"
    gap, mg = mm2px(gap_mm), mm2px(margin_mm)
    total = sum(c for _, c in items)
    papers = [paper, (paper[1], paper[0])]
    if orient == "Portrait": papers = [(min(paper), max(paper))]
    elif orient == "Landscape": papers = [(max(paper), min(paper))]
    if len(items) == 1:
        photo, copies = items[0]; pw, ph = photo.size
        bl = best_layout(pw, ph, paper if orient == "Auto" else papers[0], gap, mg) if orient == "Auto" else None
        if bl is None:
            PW, PH = papers[0]; bl = None; best = None
            for rot in (False, True):
                w, h = (ph, pw) if rot else (pw, ph); c, r = grid_fit(w, h, PW, PH, gap, mg)
                if best is None or c * r > best["cap"]: best = dict(PW=PW, PH=PH, rotate=rot, cols=c, rows=r, cap=c * r)
            bl = best
        PW, PH = bl["PW"], bl["PH"]
        ph_img = photo.rotate(90, expand=True) if bl["rotate"] else photo
        w, h = ph_img.size
        sheet = Image.new("RGB", (PW, PH), (255, 255, 255)); d = ImageDraw.Draw(sheet)
        n = min(copies, bl["cap"]); cols = max(1, bl["cols"])
        used_rows = int(math.ceil(n / cols)) if n else 0
        used_cols = min(cols, n)
        bw = used_cols * w + max(0, used_cols - 1) * gap; bh = used_rows * h + max(0, used_rows - 1) * gap
        x0 = (PW - bw) // 2 if center else mg; y0 = (PH - bh) // 2 if center else mg
        for i in range(n):
            x = x0 + (i % cols) * (w + gap); y = y0 + (i // cols) * (h + gap)
            sheet.paste(ph_img, (x, y)); draw_cut_marks(d, x, y, w, h, style)
        return sheet, n, total
    best = None
    for PW, PH in papers:
        sheet = Image.new("RGB", (PW, PH), (255, 255, 255)); d = ImageDraw.Draw(sheet)
        y = mg; placed = 0; full = False
        for photo, copies in items:
            pw, ph = photo.size; n = 0
            while n < copies and not full:
                if y + ph > PH - mg: full = True; break
                x = mg; row_n = 0
                while n < copies and x + pw <= PW - mg:
                    sheet.paste(photo, (x, y)); draw_cut_marks(d, x, y, pw, ph, style)
                    x += pw + gap; n += 1; placed += 1; row_n += 1
                if row_n == 0: full = True; break
                y += ph + gap
        if best is None or placed > best[1]: best = (sheet, placed)
        if placed == total: break
    return best[0], best[1], total

def make_sheets(photo, copies, paper, gap_mm=3, margin_mm=4, style="Light border", orient="Auto"):
    """as many sheets as needed for 'copies'. returns list of sheet images"""
    gap, mg = mm2px(gap_mm), mm2px(margin_mm)
    bl = best_layout(photo.width, photo.height, paper, gap, mg)
    cap = max(1, bl["cap"]); sheets = []; left = copies
    while left > 0:
        s, placed, _ = pack_sheet([(photo, min(left, cap))], paper, gap_mm, margin_mm, style=style, orient=orient)
        sheets.append(s); left -= max(1, placed)
        if len(sheets) > 50: break
    return sheets

def layout_text(photo_mm, paper_name, paper_px_, gap_mm, margin_mm, copies, orient="Auto"):
    """short human text: '4 x 3 = 12 fit (4R landscape). 8 copies = 1 sheet'"""
    pw, ph = mm2px(photo_mm[0]), mm2px(photo_mm[1])
    bl = best_layout(pw, ph, paper_px_, mm2px(gap_mm), mm2px(margin_mm))
    if bl["cap"] <= 0: return "Photo does not fit on this paper"
    sheets = int(math.ceil(copies / float(bl["cap"])))
    return "%d x %d = %d photos fit (%s). %d copies = %d sheet%s" % (
        bl["cols"], bl["rows"], bl["cap"], "landscape" if bl["PW"] > bl["PH"] else "portrait", copies, sheets, "" if sheets == 1 else "s")

# =====================================================================
#  FILES / EXPORT  (never overwrites, never touches the original)
# =====================================================================
def out_name(customer, kind, size_txt="", ext="jpg", sheet=False, paper=""):
    """Rahim_Passport_35x45.jpg / Rahim_Passport_Sheet_4R.jpg / Rahim_Document.pdf"""
    parts = [safe_name(customer), kind]
    if sheet: parts += ["Sheet", paper] if paper else ["Sheet"]
    elif size_txt: parts.append(size_txt)
    return "_".join(p for p in parts if p) + "." + ext

def save_image_file(im, path, dpi=None):
    ext = os.path.splitext(path)[1].lower(); dpi = dpi or DPI
    im = im.convert("RGB")
    if ext in (".jpg", ".jpeg"): im.save(path, "JPEG", quality=95, dpi=(dpi, dpi), subsampling=0)
    elif ext == ".png": im.save(path, "PNG", dpi=(dpi, dpi))
    elif ext == ".pdf": save_pdf_pages([im], path, dpi)
    else: im.save(path)

def save_pdf_pages(pages, path, dpi=None):
    pages = [p.convert("RGB") for p in pages]
    if not pages: raise ValueError("No pages")
    pages[0].save(path, "PDF", resolution=float(dpi or DPI), save_all=True, append_images=pages[1:])

def a4_page(im, margin_mm=8):
    """fit one image on a white A4 page (300 dpi)"""
    W, H = 2480, 3508; page = Image.new("RGB", (W, H), (255, 255, 255)); m = mm2px(margin_mm, 300)
    im = im.convert("RGB"); s = min((W - 2 * m) / im.width, (H - 2 * m) / im.height)
    r = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    page.paste(r, ((W - r.width) // 2, (H - r.height) // 2)); return page

# =====================================================================
#  DOCUMENT STUDIO  tools (scan photo -> clean page)
# =====================================================================
def order_quad(pts):
    pts = np.array(pts, np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)   # tl tr br bl

def detect_doc_quad(rgb):
    """find the paper/document corners. returns 4x2 float array (tl,tr,br,bl) in image pixels, or the full frame if not found"""
    H, W = rgb.shape[:2]; s = 800.0 / max(H, W)
    sm = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY), (7, 7), 0)
    best = None
    for mode in range(2):
        if mode == 0:
            e = cv2.Canny(g, 40, 140); e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2)
        else:
            _, e = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(c)
            if area < 0.2 * sm.shape[0] * sm.shape[1] or area > 0.995 * sm.shape[0] * sm.shape[1]: continue
            ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                if best is None or area > best[0]: best = (area, ap.reshape(4, 2))
        if best is not None: break
    if best is None:
        return np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    return order_quad(best[1] / s)

def warp_quad(rgb, quad):
    """straighten a perspective document. output size from the quad edge lengths"""
    q = order_quad(quad)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    w, h = max(50, w), max(50, h)
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32))
    return cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

def remove_shadows(rgb):
    """divide by the estimated paper brightness -> even lighting, shadows gone"""
    out = np.empty_like(rgb)
    k = max(15, (min(rgb.shape[:2]) // 25) | 1)
    for c in range(3):
        ch = rgb[..., c]
        bgc = cv2.medianBlur(cv2.dilate(ch, np.ones((7, 7), np.uint8)), k)
        out[..., c] = cv2.divide(ch, bgc, scale=255)
    return out

def whiten_background(rgb, strength=70):
    """push paper to pure white, keep ink dark"""
    s = strength / 100.0
    x = rgb.astype(np.float32)
    lo, hi = np.percentile(x, 2), np.percentile(x, 100 - 12 * (0.4 + s))
    x = np.clip((x - lo) / max(hi - lo, 1.0), 0, 1)
    x = np.clip(x ** (1.0 + 0.5 * s), 0, 1)
    return (x * 255).astype(np.uint8)

def sharpen_doc(rgb, amount=60):
    a = amount / 100.0
    return cv2.addWeighted(rgb, 1 + 0.9 * a, cv2.GaussianBlur(rgb, (0, 0), 1.2), -0.9 * a, 0)

def bw_scan(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, max(15, (min(g.shape) // 30) | 1), 12)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)

def signature_extract(rgb, transparent=False):
    """crop to the ink and clean the paper. transparent -> RGBA with only the ink"""
    g = cv2.cvtColor(remove_shadows(rgb), cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(ink > 0)
    if len(xs) < 20: return Image.fromarray(rgb)
    pad = max(6, int(0.03 * max(rgb.shape[:2])))
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad); y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
    crop = rgb[y0:y1, x0:x1]; a = ink[y0:y1, x0:x1]
    if transparent:
        out = np.zeros(crop.shape[:2] + (4,), np.uint8); out[..., :3] = (20, 20, 40)
        out[..., 3] = cv2.GaussianBlur(a, (0, 0), 0.8); return Image.fromarray(out, "RGBA")
    return Image.fromarray(whiten_background(crop, 90))

def doc_adjust(rgb, bright=0, contrast=0, mode="Color"):
    im = Image.fromarray(rgb)
    if bright: im = ImageEnhance.Brightness(im).enhance(1 + bright / 100.0)
    if contrast: im = ImageEnhance.Contrast(im).enhance(1 + contrast / 100.0)
    if mode == "Gray": im = ImageOps.grayscale(im).convert("RGB")
    elif mode == "Black & White": return Image.fromarray(bw_scan(np.array(im)))
    return im


# =====================================================================
#  LOCAL DATABASE  (customer jobs + shop accounts)  - SQLite, offline
# =====================================================================
class ShopDB:
    def __init__(self, path=DB_PATH):
        self.lock = threading.RLock()
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        with self.lock:
            self.con.executescript("""
            CREATE TABLE IF NOT EXISTS jobs(
                id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, updated TEXT,
                customer TEXT, phone TEXT, service TEXT, copies INTEGER, price REAL DEFAULT 0,
                preset TEXT, size_txt TEXT, paper TEXT, notes TEXT, folder TEXT, source TEXT, state TEXT);
            CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, title TEXT, amount REAL, note TEXT);
            CREATE INDEX IF NOT EXISTS ix_jobs_created ON jobs(created);
            CREATE INDEX IF NOT EXISTS ix_jobs_customer ON jobs(customer);
            """)
            self.con.commit()

    @staticmethod
    def now(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_job(self, **kw):
        kw.setdefault("customer", "Walk-in"); kw.setdefault("service", "Passport Photo")
        kw.setdefault("copies", 1); kw.setdefault("price", 0)
        kw["created"] = kw.get("created") or self.now(); kw["updated"] = kw["created"]
        cols = list(kw.keys())
        with self.lock:
            cur = self.con.execute("INSERT INTO jobs(%s) VALUES(%s)" % (",".join(cols), ",".join("?" * len(cols))), [kw[c] for c in cols])
            self.con.commit(); return cur.lastrowid

    def update_job(self, jid, **kw):
        if not kw: return
        kw["updated"] = self.now()
        with self.lock:
            self.con.execute("UPDATE jobs SET %s WHERE id=?" % ",".join(k + "=?" for k in kw), list(kw.values()) + [jid]); self.con.commit()

    def get_job(self, jid):
        with self.lock:
            r = self.con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(r) if r else None

    def delete_job(self, jid):
        with self.lock:
            self.con.execute("DELETE FROM jobs WHERE id=?", (jid,)); self.con.commit()

    def search_jobs(self, q="", limit=300):
        q = (q or "").strip()
        with self.lock:
            if q:
                like = "%" + q + "%"
                rows = self.con.execute("SELECT * FROM jobs WHERE customer LIKE ? OR phone LIKE ? OR service LIKE ? OR CAST(id AS TEXT)=? "
                                        "ORDER BY id DESC LIMIT ?", (like, like, like, q, limit)).fetchall()
            else:
                rows = self.con.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def duplicate_job(self, jid):
        j = self.get_job(jid)
        if not j: return None
        j.pop("id"); j["created"] = None
        return self.add_job(**{k: v for k, v in j.items() if v is not None})

    def add_expense(self, title, amount, note="", day=None):
        with self.lock:
            self.con.execute("INSERT INTO expenses(day,title,amount,note) VALUES(?,?,?,?)",
                             (day or datetime.date.today().isoformat(), title, float(amount), note)); self.con.commit()

    def delete_expense(self, eid):
        with self.lock:
            self.con.execute("DELETE FROM expenses WHERE id=?", (eid,)); self.con.commit()

    def expenses_on(self, day):
        with self.lock:
            return [dict(r) for r in self.con.execute("SELECT * FROM expenses WHERE day=? ORDER BY id DESC", (day,)).fetchall()]

    def sales_by_service(self, start, end):
        """start/end: 'YYYY-MM-DD' inclusive. -> [(service, total, count)]"""
        with self.lock:
            rows = self.con.execute("SELECT service, SUM(price) t, COUNT(*) c FROM jobs WHERE date(created)>=? AND date(created)<=? "
                                    "GROUP BY service ORDER BY t DESC", (start, end)).fetchall()
        return [(r["service"] or "Other", float(r["t"] or 0), int(r["c"])) for r in rows]

    def totals(self, start, end):
        inc = sum(t for _, t, _ in self.sales_by_service(start, end))
        with self.lock:
            exp = self.con.execute("SELECT SUM(amount) FROM expenses WHERE day>=? AND day<=?", (start, end)).fetchone()[0] or 0.0
        return float(inc), float(exp)

    def monthly(self, n=12):
        with self.lock:
            inc = {r[0]: (r[1] or 0, r[2]) for r in self.con.execute(
                "SELECT strftime('%Y-%m',created) m, SUM(price), COUNT(*) FROM jobs GROUP BY m").fetchall()}
            exp = {r[0]: r[1] or 0 for r in self.con.execute("SELECT strftime('%Y-%m',day) m, SUM(amount) FROM expenses GROUP BY m").fetchall()}
        months = sorted(set(inc) | set(exp), reverse=True)[:n]
        return [(m, float(inc.get(m, (0, 0))[0]), float(exp.get(m, 0)), int(inc.get(m, (0, 0))[1])) for m in months]

def taka(v):
    v = float(v or 0)
    return "৳%s" % (("%d" % round(v)) if abs(v - round(v)) < 0.005 else ("%.2f" % v))

# =====================================================================
#  ONE-CLICK AI PASSPORT PIPELINE   (runs in a worker thread, no Qt here)
# =====================================================================
def fi_from_box(face):
    """manual face box (x,y,w,h) -> estimated FaceInfo"""
    x, y, w, h = [float(v) for v in face]
    return _fi((x, y, w, h), (x + 0.30 * w, y + 0.40 * h), (x + 0.70 * w, y + 0.40 * h), (x + 0.5 * w, y + 0.62 * h),
               (x + 0.35 * w, y + 0.78 * h), (x + 0.65 * w, y + 0.78 * h), (x + 0.5 * w, y + 1.02 * h), "manual", 1.0, True)

class NoFace(Exception):
    pass

def _roi_for(fi, box, shape, margin=1.55):
    """region of the photo worth processing: the planned crop (rotated) + margin, so manual re-crop still has room"""
    H, W = shape
    cx, cy, bh, ang = box; bw = bh * 0.8
    r = math.radians(ang)
    ext_x = abs(math.cos(r)) * bw / 2 + abs(math.sin(r)) * bh / 2
    ext_y = abs(math.sin(r)) * bw / 2 + abs(math.cos(r)) * bh / 2
    x0, x1 = cx - ext_x * margin, cx + ext_x * margin; y0, y1 = cy - ext_y * margin, cy + ext_y * margin
    fx, fy, fw, fh = fi["bbox"]
    x0, y0 = min(x0, fx - 0.2 * fw), min(y0, fy - 0.6 * fh); x1, y1 = max(x1, fx + 1.2 * fw), max(y1, fy + 1.8 * fh)
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 64 or y1 - y0 < 64: return 0, 0, W, H
    return x0, y0, x1, y1

def run_ai_pipeline(img, preset, opts, ctx, hub=None):
    """img: PIL RGB original. preset: size/bg/head settings. opts: dict
         model (rembg id), bg (bool), enhance (bool), retouch (bool), restore ('auto'|True|False), upscale ('auto'|True|False),
         bg_rgb, face (optional manual (x,y,w,h) or FaceInfo)
       ctx.step(i, n, text) reports REAL progress (completed steps), ctx.check() raises Cancelled.
       Returns dict(rgb, alpha, fi, face, box, photo_base, notes, timings)."""
    hub = hub or HUB; notes = []; tm = {}
    STEPS = ["Detecting face...", "Aligning face...", "Removing background...", "Refining hair...",
             "Enhancing face...", "Retouching skin...", "Creating passport crop..."]
    N = len(STEPS)
    def stage(i):
        ctx.check(); ctx.step(i, N, STEPS[i]); tm[STEPS[i]] = time.time()
    rgb0 = np.ascontiguousarray(np.array(img.convert("RGB")))
    ratio = preset["w"] / float(preset["h"])
    # ---- 1. detect ----
    stage(0)
    fi = opts.get("face")
    if isinstance(fi, (tuple, list)): fi = fi_from_box(fi)
    if fi is None: fi = detect_face_info(rgb0, hub)
    if fi is None: raise NoFace("Face not found")
    if fi.get("src") == "haar" and fi.get("est"): notes.append(("info", "Eyes estimated (basic face detector). Install InsightFace for exact alignment."))
    # ---- 2. align + region of interest ----
    stage(1)
    box0 = ideal_box(fi, preset, None)
    if abs(fi["angle"]) > 25: notes.append(("warn", "Head is strongly tilted - check the result"))
    x0, y0, x1, y1 = _roi_for(fi, box0, rgb0.shape[:2])
    arr = np.ascontiguousarray(rgb0[y0:y1, x0:x1]); fi = _scale_fi(fi, 1.0, -x0, -y0)
    # ---- 3. background ----
    alpha = None; used_model = None
    stage(2)
    if opts.get("bg", True):
        chain = [opts.get("model") or "u2net_human_seg"]
        if chain[0] != "u2net_human_seg": chain.append("u2net_human_seg")
        for mdl in chain:
            ctx.check()
            if not (hub.lib_available(mdl)):
                notes.append(("warn", "Background AI not installed (pip install rembg onnxruntime)")); break
            try:
                alpha = hub.segment(arr, mdl); used_model = mdl; break
            except Cancelled: raise
            except Exception as e:
                log.warning("segment %s failed: %s", mdl, e)
                notes.append(("warn", "Background model '%s' failed: %s" % (mdl, (str(e).splitlines() or [""])[0][:80])))
        if alpha is None: notes.append(("warn", "Background could not be removed - original background kept. Continue manually."))
    # ---- 4. hair edges ----
    stage(3)
    if alpha is not None:
        face_pt = ((fi["le"][0] + fi["re"][0]) / 2.0, (fi["le"][1] + fi["re"][1]) / 2.0 + 0.2 * fi["bbox"][3])
        alpha = refine_alpha(arr, alpha, opts.get("edge", 60), face_pt)
        arr = decontaminate(arr, alpha)
        if alpha[int(np.clip(face_pt[1], 0, alpha.shape[0] - 1)), int(np.clip(face_pt[0], 0, alpha.shape[1] - 1))] < 0.5:
            notes.append(("warn", "Background removal may have cut the face - check the result"))
    # real head top / shoulders from the mask, then the final framing
    g = head_geometry(fi, alpha); fi["crown"] = _to_img(g["cx_l"], g["crown_ly"], g["origin"], g["angle"])
    # ---- 5. enhance face ----
    stage(4)
    stats = analyze_image(arr, fi)
    head_need = float(preset["head"]) * mm2px(preset["h"], preset.get("dpi", DPI))
    ratio_up = head_need / max(g["head_h"], 1.0)
    want_restore = opts.get("restore", "auto"); want_up = opts.get("upscale", "auto")
    if opts.get("enhance", True):
        face_blur = stats.get("face_blur", 999)
        need_restore = (want_restore is True) or (want_restore == "auto" and (face_blur < 45 or ratio_up > 1.6))
        if need_restore:
            if hub.is_installed("gfpgan") and hub.lib_available("gfpgan"):
                try:
                    ctx.check(); res = hub.restore_face(arr, 0.45)
                    arr = np.clip(res.astype(np.float32) * 0.8 + arr.astype(np.float32) * 0.2, 0, 255).astype(np.uint8)
                except Cancelled: raise
                except Exception as e:
                    notes.append(("warn", "AI Face Restore unavailable - used normal enhancement (%s)" % (str(e).splitlines() or [""])[0][:60]))
            elif want_restore is True:
                notes.append(("warn", "AI Face Restore not installed - used normal enhancement"))
        need_up = (want_up is True) or (want_up == "auto" and ratio_up > 1.12)
        if need_up:
            k = 2 if ratio_up <= 2.0 else 4
            if max(arr.shape[:2]) * k > 9000: k = 2
            try:
                ctx.check()
                big, how = hub.upscale(Image.fromarray(arr), k)
                kk = big.width / float(arr.shape[1]); arr = np.ascontiguousarray(np.array(big))
                if alpha is not None: alpha = cv2.resize(alpha, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)
                fi = _scale_fi(fi, kk)
                notes.append(("info", "Small photo upscaled x%d: %s" % (k, how)))
            except Cancelled: raise
            except Exception as e:
                notes.append(("warn", "Upscale failed: %s" % (str(e).splitlines() or [""])[0][:70]))
        arr = gentle_enhance(arr, int(preset.get("enhance", 50)))
        arr = exposure_fix(arr, fi)
    # ---- 6. skin ----
    stage(5)
    face_t = fi_face_tuple(fi)
    if opts.get("retouch", True) and face_t is not None:
        lvl = int(preset.get("retouch", 40))
        try:
            arr, npim = remove_blemishes(arr, face_t, 50)
            m = feature_guard(skin_mask(arr, face_t), fi)
            arr = skin_shine_fix(arr, m, 45)
            arr = skin_retouch(arr, m, face_t, lvl)
            if npim: notes.append(("info", "%d spot(s) cleaned" % npim))
        except Cancelled: raise
        except Exception as e:
            notes.append(("warn", "Skin retouch skipped: %s" % (str(e).splitlines() or [""])[0][:60])); log.exception("retouch")
    # ---- 7. passport crop ----
    stage(6)
    g = head_geometry(fi, alpha); fi["crown"] = _to_img(g["cx_l"], g["crown_ly"], g["origin"], g["angle"])
    box = ideal_box(fi, preset, alpha)
    rgb_img = Image.fromarray(arr)
    alpha_img = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)) if alpha is not None else None
    src = rgb_img.convert("RGBA")
    if alpha_img is not None: src.putalpha(alpha_img)
    bg = tuple(opts.get("bg_rgb") or preset.get("bg") or (255, 255, 255))
    photo_base = crop_to(src, bg, (preset["w"], preset["h"]), box)
    ctx.step(N, N, "Done")
    return dict(rgb=rgb_img, alpha=alpha_img, fi=fi, face=fi_face_tuple(fi), box=box, photo_base=photo_base, notes=notes,
                model=used_model, stats=stats, roi=(x0, y0, x1, y1))

# =====================================================================
#  FINAL QUALITY CHECK  (before save / print)
# =====================================================================
def quality_report(photo, fi, box, ratio, preset, alpha_present, alpha=None):
    """-> list of (level, code, text, fix).  fix in {None,'framing','upscale','bg','exposure','restore'}"""
    out = []
    for lv, code, txt in passport_check(fi, box, ratio, preset, None):
        if lv != "ok": out.append((lv, code, txt, "framing" if fi is not None else None))
    if photo is not None:
        ph = photo.height; src_per_out = box[2] / float(max(1, ph))
        if src_per_out < 0.5: out.append(("bad", "res", "⚠ Image resolution too low", "upscale"))
        elif src_per_out < 0.78: out.append(("warn", "res", "⚠ Image resolution low (looks soft)", "upscale"))
        exp = (mm2px(preset["w"], preset.get("dpi", DPI)), mm2px(preset["h"], preset.get("dpi", DPI)))
        if abs(photo.width - exp[0]) > 2 or abs(photo.height - exp[1]) > 2 + int(ph * 0.12):
            out.append(("warn", "dim", "⚠ Photo size is not %dx%d mm" % (preset["w"], preset["h"]), None))
        if not alpha_present and bg_uniformity(photo) > 12:
            out.append(("warn", "bg", "⚠ Background is not even - use Remove Background", "bg"))
        a = np.array(photo.convert("L")); h, w = a.shape
        mid = a[int(h * 0.22):int(h * 0.62), int(w * 0.3):int(w * 0.7)]
        if mid.size:
            m = float(mid.mean())
            if m < 55: out.append(("warn", "exposure", "⚠ Photo too dark", "exposure"))
            elif m > 222: out.append(("warn", "exposure", "⚠ Photo too bright", "exposure"))
            sharp = cv2.Laplacian(cv2.resize(mid, (200, 200)), cv2.CV_64F).var()
            if sharp < 18: out.append(("warn", "blur", "⚠ Photo looks blurry", "restore"))
    if not out: out.append(("ok", "ok", "✓ Ready to Print", None))
    return out


# =====================================================================
#  BACKGROUND TASKS  (GUI thread never runs AI / heavy CPU work)
# =====================================================================
class TaskCtx:
    """handed to every background function: report REAL progress and honour Cancel"""
    def __init__(self, runner, cancel):
        self._r, self._c = runner, cancel
    def step(self, i, n, text): self._r.progress.emit(text, int(i), int(n))
    def msg(self, text): self._r.progress.emit(text, -1, 0)
    def frac(self, text, f): self._r.progress.emit(text, int(max(0.0, min(1.0, f)) * 1000), -1000)
    def check(self):
        if self._c.is_set(): raise Cancelled()

class TaskRunner(QObject):
    """One job at a time (duplicate clicks are ignored). Daemon thread -> the app can always close,
    even if a model is stuck. Results come back to the GUI thread through queued signals."""
    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.busy = False; self._cancel = threading.Event(); self._id = 0; self._cb = {}
        self.finished.connect(self._on_done); self.failed.connect(self._on_fail)

    def submit(self, fn, on_done, on_fail=None):
        if self.busy: return False
        self.busy = True; self._id += 1; tid = self._id
        self._cancel = threading.Event(); self._cb[tid] = (on_done, on_fail)
        threading.Thread(target=self._run, args=(tid, fn, self._cancel), daemon=True, name="ripon-task").start()
        return True

    def cancel(self):
        self._cancel.set()

    def _run(self, tid, fn, cancel):
        try:
            res = fn(TaskCtx(self, cancel))
            self.finished.emit((tid, res))
        except Cancelled:
            self.failed.emit((tid, "Cancelled", None))
        except Exception as e:
            log.error("task failed: %s\n%s", e, traceback.format_exc())
            self.failed.emit((tid, str(e) or e.__class__.__name__, e))

    @Slot(object)
    def _on_done(self, payload):
        tid, res = payload; self.busy = False
        cb = self._cb.pop(tid, (None, None))[0]
        if cb:
            try: cb(res)
            except Exception as e:
                log.error("done-callback failed: %s\n%s", e, traceback.format_exc())
                QMessageBox.warning(None, APP_NAME, "Something went wrong while showing the result:\n%s" % e)

    @Slot(object)
    def _on_fail(self, payload):
        tid, msg, exc = payload; self.busy = False
        fb = self._cb.pop(tid, (None, None))[1]
        if fb:
            try: fb(msg, exc)
            except Exception as e: log.error("fail-callback failed: %s", e)

# =====================================================================
#  TOP PROCESSING BANNER  (always at the very top of the window)
# =====================================================================
class ProcessingBanner(QFrame):
    cancel_clicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PB"); self.setMinimumHeight(58); self.setMaximumHeight(58)
        self.dot = QLabel("●"); self.dot.setFixedWidth(22)
        self.title = QLabel("READY"); self.msg = QLabel("Open a photo to begin")
        self.right = QLabel(""); self.right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.bar = QProgressBar(); self.bar.setTextVisible(False); self.bar.setFixedHeight(6); self.bar.setRange(0, 1); self.bar.setValue(0)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setFixedWidth(70); self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_clicked.emit)
        col = QVBoxLayout(); col.setSpacing(0); col.setContentsMargins(0, 0, 0, 0)
        self.title.setObjectName("PBT"); self.msg.setObjectName("PBM"); self.right.setObjectName("PBR"); self.dot.setObjectName("PBD")
        col.addWidget(self.title); col.addWidget(self.msg)
        top = QHBoxLayout(); top.setContentsMargins(12, 6, 12, 2)
        top.addWidget(self.dot); top.addLayout(col, 1); top.addWidget(self.right); top.addWidget(self.cancel_btn)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        lay.addLayout(top); lay.addWidget(self.bar)
        self._set_style("#2b3a47", "#8fb8de", "#8fb8de")
        self._t = QTimer(self); self._t.setInterval(220); self._t.timeout.connect(self._tick); self._n = 0; self._label = "PROCESSING"
        self._hold = QTimer(self); self._hold.setSingleShot(True); self._hold.timeout.connect(self.idle)

    def _set_style(self, bg, accent, dot):
        self.setStyleSheet("""QFrame#PB{background:%s;border-bottom:2px solid %s;}
            QLabel#PBT{color:#fff;font-size:13px;font-weight:700;letter-spacing:1px;}
            QLabel#PBM{color:#dbe9f5;font-size:12px;}QLabel#PBR{color:#fff;font-size:13px;font-weight:700;min-width:90px;}
            QLabel#PBD{color:%s;font-size:18px;}
            QProgressBar{border:0;background:rgba(255,255,255,40);}QProgressBar::chunk{background:%s;}
            QPushButton{background:#455a6b;color:#fff;border:0;border-radius:4px;padding:3px;}""" % (bg, accent, dot, accent))

    def _tick(self):
        self._n = (self._n + 1) % 4
        self.dot.setText("●" if self._n % 2 == 0 else "◉"); self.title.setText(self._label + "." * (self._n + 1))

    def start(self, message, cancellable=True, label="PROCESSING"):
        self._hold.stop(); self._label = label; self._n = 0
        self._set_style("#16324a", "#42a5f5", "#42a5f5")
        self.msg.setText(message); self.right.setText(""); self.bar.setRange(0, 0)       # animated, unknown duration
        self.cancel_btn.setVisible(cancellable); self._t.start(); self._tick()

    def progress_update(self, message, i=-1, n=0):
        if message: self.msg.setText(message)
        if n > 0 and i >= 0:                       # REAL progress: finished steps / total steps
            self.bar.setRange(0, n); self.bar.setValue(i)
            self.right.setText("Step %d / %d  (%d%%)" % (min(i + 1, n), n, int(100 * i / n)))
        elif n == 0 and i < 0 and self.bar.maximum() != 0 and not self.right.text():
            self.bar.setRange(0, 0)

    def set_fraction(self, frac, message=None):
        if message: self.msg.setText(message)
        if frac is None: self.bar.setRange(0, 0); self.right.setText("")
        else: self.bar.setRange(0, 1000); self.bar.setValue(int(frac * 1000)); self.right.setText("%d%%" % int(frac * 100))

    def done(self, message="Done"):
        self._t.stop(); self._set_style("#1b5e20", "#66bb6a", "#66bb6a")
        self.dot.setText("✓"); self.title.setText("DONE ✓"); self.msg.setText(message); self.right.setText("")
        self.bar.setRange(0, 1); self.bar.setValue(1); self.cancel_btn.setVisible(False); self._hold.start(4000)

    def warn(self, message):
        self._t.stop(); self._set_style("#7a4a00", "#ffb300", "#ffb300")
        self.dot.setText("⚠"); self.title.setText("DONE WITH WARNINGS"); self.msg.setText(message); self.right.setText("")
        self.bar.setRange(0, 1); self.bar.setValue(1); self.cancel_btn.setVisible(False); self._hold.start(9000)

    def error(self, message):
        self._t.stop(); self._set_style("#7f1d1d", "#ef5350", "#ef5350")
        self.dot.setText("✖"); self.title.setText("FAILED"); self.msg.setText(message); self.right.setText("")
        self.bar.setRange(0, 1); self.bar.setValue(0); self.cancel_btn.setVisible(False); self._hold.start(10000)

    def idle(self, message="Ready"):
        self._t.stop(); self._set_style("#2b3a47", "#8fb8de", "#8fb8de")
        self.dot.setText("●"); self.title.setText("READY"); self.msg.setText(message); self.right.setText("")
        self.bar.setRange(0, 1); self.bar.setValue(0); self.cancel_btn.setVisible(False)

# =====================================================================
#  PRINTING  (real printer dialog + preview, exact physical size)
# =====================================================================
def pil_to_qimage(im):
    im = im.convert("RGB")
    return QImage(im.tobytes(), im.width, im.height, 3 * im.width, QImage.Format_RGB888).copy()

def pil_to_pixmap(im):
    return QPixmap.fromImage(pil_to_qimage(im))

def _setup_printer(printer, sheet, dpi):
    wmm, hmm = sheet.width / float(dpi) * 25.4, sheet.height / float(dpi) * 25.4
    printer.setFullPage(True)
    try:
        printer.setPageSize(QPageSize(QSizeF(min(wmm, hmm), max(wmm, hmm)), QPageSize.Millimeter))
        printer.setPageOrientation(QPageLayout.Landscape if sheet.width > sheet.height else QPageLayout.Portrait)
        printer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)
    except Exception as e:
        log.warning("printer page setup: %s", e)

def _paint_sheets(printer, sheets, dpi, copies=1):
    p = QPainter(printer); res = printer.resolution(); first = True
    try:
        for _ in range(max(1, copies)):
            for s in sheets:
                if not first: printer.newPage()
                first = False
                q = pil_to_qimage(s)
                p.drawImage(QRectF(0, 0, q.width() * res / float(dpi), q.height() * res / float(dpi)), q)     # real size
    finally:
        p.end()

def print_sheets(parent, sheets, dpi, preview=False):
    """sheets: list of PIL images (one per page). Shows the system printer dialog (printer, copies, orientation)."""
    if not sheets: return False
    printer = QPrinter(QPrinter.HighResolution); _setup_printer(printer, sheets[0], dpi)
    if preview:
        dlg = QPrintPreviewDialog(printer, parent); dlg.setWindowTitle("Print Preview")
        dlg.paintRequested.connect(lambda pr: _paint_sheets(pr, sheets, dpi, 1))
        dlg.resize(900, 800); dlg.exec(); return True
    dlg = QPrintDialog(printer, parent); dlg.setWindowTitle("Print")
    if not dlg.exec(): return False
    n = printer.copyCount() if not printer.supportsMultipleCopies() else 1
    if printer.supportsMultipleCopies(): _paint_sheets(printer, sheets, dpi, 1)
    else: _paint_sheets(printer, sheets, dpi, max(1, n))
    return True

# =====================================================================
#  AUTO-SAVE / CRASH RECOVERY
# =====================================================================
class Recovery:
    LOCK = os.path.join(RECOVERY_DIR, "session.lock")
    STATE = os.path.join(RECOVERY_DIR, "state.json")
    RGB = os.path.join(RECOVERY_DIR, "work_rgb.png")
    ALPHA = os.path.join(RECOVERY_DIR, "work_alpha.png")
    _writing = threading.Lock()

    @classmethod
    def crashed(cls):
        return os.path.exists(cls.LOCK) and os.path.exists(cls.STATE) and os.path.exists(cls.RGB)

    @classmethod
    def begin(cls):
        try:
            with open(cls.LOCK, "w") as f: f.write(str(os.getpid()))
        except Exception: pass

    @classmethod
    def end(cls, clear=True):
        for p in ((cls.LOCK, cls.STATE, cls.RGB, cls.ALPHA) if clear else (cls.LOCK,)):
            try: os.remove(p)
            except Exception: pass

    @classmethod
    def save_async(cls, state, rgb, alpha):
        """PNG writes happen in a thread so the UI never stutters. rgb/alpha are treated as immutable images."""
        def work():
            if not cls._writing.acquire(False): return
            try:
                tmp = cls.RGB + ".tmp.png"; rgb.save(tmp, "PNG", compress_level=1); os.replace(tmp, cls.RGB)
                if alpha is not None:
                    tmp = cls.ALPHA + ".tmp.png"; alpha.save(tmp, "PNG", compress_level=1); os.replace(tmp, cls.ALPHA)
                elif os.path.exists(cls.ALPHA): os.remove(cls.ALPHA)
                tmp = cls.STATE + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f: json.dump(state, f)
                os.replace(tmp, cls.STATE)
            except Exception as e:
                log.warning("autosave failed: %s", e)
            finally:
                cls._writing.release()
        threading.Thread(target=work, daemon=True, name="ripon-autosave").start()

    @classmethod
    def load(cls):
        try:
            with open(cls.STATE, "r", encoding="utf-8") as f: st = json.load(f)
            rgb = Image.open(cls.RGB).convert("RGB"); rgb.load()
            alpha = None
            if os.path.exists(cls.ALPHA): alpha = Image.open(cls.ALPHA).convert("L"); alpha.load()
            return st, rgb, alpha
        except Exception as e:
            log.warning("recovery load failed: %s", e); return None

# ---------------- small widget helpers (same as v2) ----------------
def hslider(lo, hi, val):
    s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val); return s

def btn(text, fn, style="", tip=None, h=36):
    b = QPushButton(text); b.setMinimumHeight(h)
    if fn: b.clicked.connect(lambda _=False, f=fn: f())
    if style: b.setStyleSheet(style)
    if tip: b.setToolTip(tip)
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

def ask_yes_no(parent, title, text):
    return QMessageBox.question(parent, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes


# =====================================================================
#  VIEWS  (crop / retouch / face-pick: from v2, with passport guides)
# =====================================================================
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
        self.guide = (0.09, 0.74)            # (gap above crown, head height) as fraction of the crop
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
        gt, gh = self.guide
        p.drawEllipse(QRectF(-w * 0.27, -h / 2 + h * gt, w * 0.54, h * gh))          # head guide (crown to chin)
        p.setPen(QPen(QColor(255, 220, 0, 220), 1, Qt.DashLine))
        ey = -h / 2 + h * (gt + gh * 0.46); p.drawLine(int(-w / 2), int(ey), int(w / 2), int(ey))    # eye line
        p.setPen(QPen(QColor(255, 255, 255, 230), 1, Qt.DashLine))
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

class FacePickView(QWidget):
    """Manual face box: mouse drag = notun box, box er bhitore drag = sorano, kone tene = boro/chhoto.
    Box ta bhru (eyebrow) theke thutni (chin) porjonto, duto kan soho hobe. Enter = OK, Esc = Cancel."""
    HIT = 12
    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 650); self.setMouseTracking(True)
        self.pm = None; self.iw = self.ih = 1; self.box = None      # [x, y, w, h] image pixel
        self._mode = None; self._st = None

    def set_image(self, arr, face):
        arr = np.ascontiguousarray(arr); h, w = arr.shape[:2]
        self.pm = QPixmap.fromImage(QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.iw, self.ih = w, h
        self.box = [float(v) for v in face] if face is not None else None
        self._mode = None; self.update()

    def get_box(self):
        if self.box is None: return None
        x, y, w, h = self.box
        x0, y0 = max(0, x), max(0, y); x1, y1 = min(self.iw, x + w), min(self.ih, y + h)
        if x1 - x0 < 20 or y1 - y0 < 20: return None
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    def _geom(self):
        s = min(self.width() / max(1, self.iw), self.height() / max(1, self.ih))
        return s, (self.width() - self.iw * s) / 2, (self.height() - self.ih * s) / 2

    def _w2i(self, pt):
        s, ox, oy = self._geom(); return (pt.x() - ox) / s, (pt.y() - oy) / s

    def _corner_hit(self, ix, iy):
        if self.box is None: return None
        x, y, w, h = self.box; r = self.HIT / self._geom()[0]
        for cx, cy, ax, ay in ((x, y, x + w, y + h), (x + w, y, x, y + h), (x + w, y + h, x, y), (x, y + h, x + w, y)):
            if abs(ix - cx) < r and abs(iy - cy) < r: return (ax, ay)     # anchor = opposite corner
        return None

    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#cfcfcf"))
        if self.pm is None: p.end(); return
        s, ox, oy = self._geom()
        p.drawPixmap(QRectF(ox, oy, self.iw * s, self.ih * s), self.pm, QRectF(self.pm.rect()))
        if self.box is not None:
            x, y, w, h = self.box
            r = QRectF(ox + x * s, oy + y * s, w * s, h * s)
            path = QPainterPath(); path.addRect(QRectF(self.rect())); path.addRect(r)
            p.fillPath(path, QColor(0, 0, 0, 110))
            p.setPen(QPen(QColor(0, 230, 118), 2)); p.drawRect(r)
            for cx, cy in ((r.left(), r.top()), (r.right(), r.top()), (r.right(), r.bottom()), (r.left(), r.bottom())):
                hr = QRectF(cx - 5, cy - 5, 10, 10); p.fillRect(hr, QColor(255, 255, 255)); p.drawRect(hr)
        p.setPen(QColor(20, 20, 20)); p.fillRect(QRectF(0, 0, self.width(), 24), QColor(255, 255, 255, 200))
        p.drawText(8, 17, "Drag a box over the face (eyebrows to chin, both ears)  |  Enter = OK   Esc = Cancel")
        p.end()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton: return
        ix, iy = self._w2i(e.position())
        anc = self._corner_hit(ix, iy)
        if anc is not None:
            self._mode = "resize"; self._st = dict(ax=anc[0], ay=anc[1])
        elif self.box is not None and self.box[0] <= ix <= self.box[0] + self.box[2] and self.box[1] <= iy <= self.box[1] + self.box[3]:
            self._mode = "move"; self._st = dict(dx=ix - self.box[0], dy=iy - self.box[1])
        else:
            self._mode = "resize"; self._st = dict(ax=ix, ay=iy); self.box = [ix, iy, 0.0, 0.0]
        self.update()

    def mouseMoveEvent(self, e):
        ix, iy = self._w2i(e.position())
        if self._mode is None:
            if self._corner_hit(ix, iy) is not None: self.setCursor(Qt.SizeFDiagCursor)
            elif self.box is not None and self.box[0] <= ix <= self.box[0] + self.box[2] and self.box[1] <= iy <= self.box[1] + self.box[3]:
                self.setCursor(Qt.SizeAllCursor)
            else: self.setCursor(Qt.CrossCursor)
            return
        ix = min(max(ix, 0), self.iw); iy = min(max(iy, 0), self.ih)
        if self._mode == "resize":
            ax, ay = self._st["ax"], self._st["ay"]
            self.box = [min(ax, ix), min(ay, iy), abs(ix - ax), abs(iy - ay)]
        else:
            w, h = self.box[2], self.box[3]
            self.box[0] = min(max(ix - self._st["dx"], 0), self.iw - w)
            self.box[1] = min(max(iy - self._st["dy"], 0), self.ih - h)
        self.update()

    def mouseReleaseEvent(self, e): self._mode = None

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
            notes.append("AI Face skipped: " + (str(e).splitlines() or [""])[0][:90])
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
        notes.append("Face not found, skin/pimple skipped")
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
            notes.append("Face not found, hair skipped")
        else:
            person_np = None if person is None else np.array(person, np.float32) / 255.0
            hair_mask = hair_mask_auto(arr, face, person_np)
            arr = apply_hair_color(arr, hair_mask, hair_rgb, settings.get("hair_strength", 100))
    if settings.get("bg"):
        alpha = person
    return arr, alpha, hair_mask, notes


# =====================================================================
#  DIALOGS: AI Models, Jobs, Accounts, Presets
# =====================================================================
def _table(headers, widths=None):
    t = QTableWidget(0, len(headers)); t.setHorizontalHeaderLabels(headers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows); t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers); t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(True)
    for i, w in enumerate(widths or []): t.setColumnWidth(i, w)
    return t

def _fill(t, rows, align_right=()):
    t.setRowCount(len(rows))
    for r, vals in enumerate(rows):
        for c, v in enumerate(vals):
            it = QTableWidgetItem(str(v))
            if c in align_right: it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(r, c, it)

class ModelsDialog(QDialog):
    def __init__(self, app):
        super().__init__(app); self.app = app
        self.setWindowTitle("AI Models  -  installed / missing / GPU"); self.resize(900, 520)
        self.info = QLabel(); self.info.setWordWrap(True); self.info.setStyleSheet("font-size:13px;padding:6px;background:#eef4fa;border-radius:6px;")
        self.t = _table(["Model", "Status", "Running on", "Size", "What it does"], [300, 120, 90, 80, 400])
        self.c_gpu = QCheckBox("Use GPU (RTX) when available"); self.c_gpu.setChecked(bool(SETTINGS.get("gpu", True)))
        self.c_warm = QCheckBox("Load installed AI models when the program starts (faster first photo)"); self.c_warm.setChecked(bool(SETTINGS.get("warmup", True)))
        self.c_gpu.toggled.connect(self._opts); self.c_warm.toggled.connect(self._opts)
        b = QHBoxLayout()
        for text, fn, tip in (("⬇ Download selected", self.download, "Internet needed once. Saved in the models folder."),
                              ("↻ Reload selected", self.reload, "Unload and load again"),
                              ("Free memory", self.free, "Unload all models from RAM/GPU"),
                              ("Open models folder", lambda: self._open(MODELS_DIR), ""), ("Refresh", self.refresh, "")):
            b.addWidget(btn(text, fn, tip=tip))
        lay = QVBoxLayout(self); lay.addWidget(self.info); lay.addWidget(self.t, 1); lay.addWidget(self.c_gpu); lay.addWidget(self.c_warm)
        lay.addLayout(b); lay.addWidget(QLabel("Models folder: " + MODELS_DIR))
        self.refresh()

    def _open(self, p):
        if hasattr(os, "startfile"): os.startfile(p)

    def _opts(self, *_):
        SETTINGS["gpu"] = self.c_gpu.isChecked(); SETTINGS["warmup"] = self.c_warm.isChecked(); save_settings()
        HUB.unload()          # providers are chosen at load time

    def _sel(self):
        r = self.t.currentRow()
        return list(MODEL_REGISTRY.keys())[r] if r >= 0 else None

    def refresh(self):
        rows, dev = HUB.status_rows()
        _fill(self.t, [(r["label"], r["status"], r["where"], r["size"], (r["note"] + ("   ERROR: " + r["err"] if r["err"] else ""))) for r in rows])
        for i, r in enumerate(rows):
            col = {"Loaded": "#2e7d32", "Installed": "#2e7d32", "Not downloaded": "#ef6c00", "Package missing": "#c62828"}.get(r["status"], "#000")
            self.t.item(i, 1).setForeground(QColor(col))
        lines = []
        if dev["cuda"]: lines.append("✓ GPU ready for face restore / upscale: %s" % dev["gpu"])
        else: lines.append("• Face restore / upscale run on CPU (install PyTorch with CUDA for the RTX 4060)" if not dev["torch"] else "• PyTorch has no CUDA - running on CPU")
        lines.append("✓ GPU ready for background removal / face detect (ONNX CUDA)" if dev["ort_gpu"] else "• Background removal runs on CPU (pip uninstall onnxruntime, pip install onnxruntime-gpu for GPU)")
        self.info.setText("\n".join(lines))

    def download(self):
        k = self._sel()
        if not k: return QMessageBox.information(self, "AI Models", "Select a model row first.")
        if not HUB.lib_available(k):
            return QMessageBox.information(self, "Package missing", "The Python package for this model is not installed.\nSee the install command at the top of the program file / README.")
        self.app.download_model(k, self.refresh)

    def reload(self):
        k = self._sel()
        if k: HUB.unload(k); self.app.warm_model(k, self.refresh)

    def free(self):
        HUB.unload(); self.refresh()

class JobEditDialog(QDialog):
    def __init__(self, parent, job=None):
        super().__init__(parent); self.setWindowTitle("Customer Job"); job = job or {}
        self.name = QLineEdit(job.get("customer", "")); self.phone = QLineEdit(job.get("phone", ""))
        self.service = QComboBox(); self.service.addItems(SERVICES); self.service.setEditable(True)
        self.service.setCurrentText(job.get("service", SERVICES[0]))
        self.copies = QSpinBox(); self.copies.setRange(1, 999); self.copies.setValue(int(job.get("copies", 1) or 1))
        self.price = QDoubleSpinBox(); self.price.setRange(0, 1000000); self.price.setDecimals(0); self.price.setPrefix("৳ "); self.price.setValue(float(job.get("price", 0) or 0))
        self.notes = QLineEdit(job.get("notes", "") or "")
        f = QFormLayout(self); f.addRow("Customer", self.name); f.addRow("Phone", self.phone); f.addRow("Service", self.service)
        f.addRow("Copies", self.copies); f.addRow("Price", self.price); f.addRow("Note", self.notes)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); f.addRow(bb)

    def get(self):
        return dict(customer=self.name.text().strip() or "Walk-in", phone=self.phone.text().strip(), service=self.service.currentText().strip() or "Other",
                    copies=self.copies.value(), price=self.price.value(), notes=self.notes.text().strip())

class JobsDialog(QDialog):
    def __init__(self, app):
        super().__init__(app); self.app = app; self.setWindowTitle("Customer Jobs"); self.resize(980, 560)
        self.q = QLineEdit(); self.q.setPlaceholderText("Search customer / phone / service / job number ...")
        self._t = QTimer(self); self._t.setSingleShot(True); self._t.setInterval(250); self._t.timeout.connect(self.load)
        self.q.textChanged.connect(lambda *_: self._t.start())
        self.t = _table(["#", "Date", "Customer", "Phone", "Service", "Copies", "Price"], [60, 150, 200, 130, 150, 70, 90])
        self.t.doubleClicked.connect(lambda *_: self.open_sel())
        b = QHBoxLayout()
        for text, fn in (("➕ New Job", self.new), ("📂 Open", self.open_sel), ("⧉ Duplicate", self.dup), ("Export Files", self.export),
                         ("Open Folder", self.folder), ("🗑 Delete", self.delete)):
            b.addWidget(btn(text, fn))
        lay = QVBoxLayout(self); lay.addWidget(self.q); lay.addWidget(self.t, 1); lay.addLayout(b)
        self.rows = []; self.load()

    def load(self):
        self.rows = self.app.db.search_jobs(self.q.text())
        _fill(self.t, [(j["id"], j["created"], j["customer"], j["phone"] or "", j["service"], j["copies"], taka(j["price"])) for j in self.rows], (0, 5, 6))

    def _id(self):
        r = self.t.currentRow()
        if r < 0 or r >= len(self.rows):
            QMessageBox.information(self, "Jobs", "Select a job first."); return None
        return self.rows[r]["id"]

    def new(self): self.app.job_new(); self.load()
    def open_sel(self):
        i = self._id()
        if i: self.app.job_open(i); self.accept()
    def dup(self):
        i = self._id()
        if i: self.app.db.duplicate_job(i); self.load()
    def export(self):
        i = self._id()
        if i: self.app.job_export(i)
    def folder(self):
        i = self._id(); j = self.app.db.get_job(i) if i else None
        if j and j.get("folder") and os.path.isdir(j["folder"]) and hasattr(os, "startfile"): os.startfile(j["folder"])
        elif j: QMessageBox.information(self, "Jobs", "This job has no saved files yet.")
    def delete(self):
        i = self._id()
        if i and ask_yes_no(self, "Delete job", "Delete job #%d from the list?\n(Photo files stay on the disk.)" % i):
            self.app.db.delete_job(i); self.load()

class AccountsDialog(QDialog):
    def __init__(self, app):
        super().__init__(app); self.app = app; self.setWindowTitle("Shop Accounts (simple)"); self.resize(860, 600)
        self.day = QLineEdit(datetime.date.today().isoformat()); self.day.setMaximumWidth(120)
        self.day.editingFinished.connect(self.load)
        self.sum_lbl = QLabel(); self.sum_lbl.setStyleSheet("font-size:16px;font-weight:bold;padding:6px;")
        self.t_sales = _table(["Service", "Jobs", "Amount"], [260, 80, 120]); self.t_exp = _table(["#", "Expense", "Amount", "Note"], [50, 240, 100, 200])
        self.e_title = QLineEdit(); self.e_title.setPlaceholderText("Expense (paper, ink, rent ...)")
        self.e_amt = QDoubleSpinBox(); self.e_amt.setRange(0, 10000000); self.e_amt.setDecimals(0); self.e_amt.setPrefix("৳ ")
        self.s_service = QComboBox(); self.s_service.addItems(SERVICES); self.s_amt = QDoubleSpinBox(); self.s_amt.setRange(0, 10000000); self.s_amt.setDecimals(0); self.s_amt.setPrefix("৳ ")
        tab = QTabWidget(); today = QWidget(); tl = QVBoxLayout(today)
        tl.addWidget(row(QLabel("Day (YYYY-MM-DD):"), self.day, btn("Today", self.set_today)))
        tl.addWidget(self.sum_lbl)
        tl.addWidget(QLabel("Sales by service")); tl.addWidget(self.t_sales)
        tl.addWidget(group("Quick sale (no photo job)", [row(self.s_service, self.s_amt, btn("Add sale", self.add_sale))]))
        tl.addWidget(QLabel("Expenses")); tl.addWidget(self.t_exp)
        tl.addWidget(row(self.e_title, self.e_amt, btn("Add expense", self.add_exp), btn("Delete selected", self.del_exp)))
        month = QWidget(); ml = QVBoxLayout(month); self.m_lbl = QLabel(); self.m_lbl.setStyleSheet("font-size:15px;font-weight:bold;")
        self.t_month_srv = _table(["Service (this month)", "Jobs", "Amount"], [260, 80, 120]); self.t_months = _table(["Month", "Income", "Expenses", "Net", "Jobs"], [120, 120, 120, 120, 80])
        ml.addWidget(self.m_lbl); ml.addWidget(self.t_month_srv); ml.addWidget(QLabel("All months")); ml.addWidget(self.t_months)
        tab.addTab(today, "Today"); tab.addTab(month, "Monthly summary")
        lay = QVBoxLayout(self); lay.addWidget(tab); self.exp_rows = []; self.load()

    def set_today(self): self.day.setText(datetime.date.today().isoformat()); self.load()

    def _day(self):
        d = self.day.text().strip()
        try: datetime.date.fromisoformat(d); return d
        except Exception: return datetime.date.today().isoformat()

    def load(self):
        d = self._day(); db = self.app.db
        sales = db.sales_by_service(d, d); inc, exp = db.totals(d, d)
        _fill(self.t_sales, [(s, c, taka(t)) for s, t, c in sales] + [("TOTAL", sum(c for _, _, c in sales), taka(inc))], (1, 2))
        self.sum_lbl.setText("Income %s     Expenses %s     Net %s" % (taka(inc), taka(exp), taka(inc - exp)))
        self.exp_rows = db.expenses_on(d); _fill(self.t_exp, [(e["id"], e["title"], taka(e["amount"]), e["note"] or "") for e in self.exp_rows], (0, 2))
        first = d[:8] + "01"; y, m = int(d[:4]), int(d[5:7]); last = (datetime.date(y + (m == 12), (m % 12) + 1, 1) - datetime.timedelta(days=1)).isoformat()
        ms = db.sales_by_service(first, last); mi, me = db.totals(first, last)
        _fill(self.t_month_srv, [(s, c, taka(t)) for s, t, c in ms] + [("TOTAL", sum(c for _, _, c in ms), taka(mi))], (1, 2))
        self.m_lbl.setText("%s    Income %s   Expenses %s   Net %s" % (d[:7], taka(mi), taka(me), taka(mi - me)))
        _fill(self.t_months, [(mm, taka(i), taka(e), taka(i - e), n) for mm, i, e, n in db.monthly(24)], (1, 2, 3, 4))

    def add_sale(self):
        if self.s_amt.value() <= 0: return
        self.app.db.add_job(customer="Walk-in", service=self.s_service.currentText(), copies=1, price=self.s_amt.value(),
                            created=self._day() + datetime.datetime.now().strftime(" %H:%M:%S")); self.s_amt.setValue(0); self.load()

    def add_exp(self):
        if not self.e_title.text().strip() or self.e_amt.value() <= 0: return
        self.app.db.add_expense(self.e_title.text().strip(), self.e_amt.value(), day=self._day()); self.e_title.clear(); self.e_amt.setValue(0); self.load()

    def del_exp(self):
        r = self.t_exp.currentRow()
        if 0 <= r < len(self.exp_rows): self.app.db.delete_expense(self.exp_rows[r]["id"]); self.load()

class PresetDialog(QDialog):
    def __init__(self, parent, name="", data=None, builtin=False):
        super().__init__(parent); self.setWindowTitle("Photo Preset"); d = dict(data or {}); self._builtin = builtin
        self.bg = tuple(d.get("bg", (255, 255, 255)))
        self.name = QLineEdit(name); self.name.setReadOnly(builtin)
        self.w = QDoubleSpinBox(); self.w.setRange(10, 200); self.w.setValue(d.get("w", 35)); self.w.setSuffix(" mm")
        self.h = QDoubleSpinBox(); self.h.setRange(10, 300); self.h.setValue(d.get("h", 45)); self.h.setSuffix(" mm")
        self.dpi = QSpinBox(); self.dpi.setRange(72, 1200); self.dpi.setValue(d.get("dpi", 300))
        self.copies = QSpinBox(); self.copies.setRange(1, 200); self.copies.setValue(d.get("copies", 6))
        self.paper = QComboBox(); self.paper.addItems(list(PAPERS_MM.keys())); self.paper.setCurrentText(d.get("paper", "4R (4x6 inch)"))
        self.head = QSpinBox(); self.head.setRange(40, 90); self.head.setSuffix(" %"); self.head.setValue(int(round(d.get("head", 0.74) * 100)))
        self.top = QSpinBox(); self.top.setRange(0, 30); self.top.setSuffix(" %"); self.top.setValue(int(round(d.get("top", 0.09) * 100)))
        self.enh = QSpinBox(); self.enh.setRange(0, 100); self.enh.setValue(d.get("enhance", 50))
        self.ret = QSpinBox(); self.ret.setRange(0, 100); self.ret.setValue(d.get("retouch", 40))
        self.bg_btn = QPushButton("Pick background color..."); self.bg_btn.clicked.connect(self._pick); self._paint_bg()
        f = QFormLayout(self)
        f.addRow("Preset name", self.name); f.addRow("Width", self.w); f.addRow("Height", self.h); f.addRow("DPI", self.dpi)
        f.addRow("Background", self.bg_btn); f.addRow("Copies", self.copies); f.addRow("Paper", self.paper)
        f.addRow("Head height (chin to top of hair)", self.head); f.addRow("Space above head", self.top)
        f.addRow("Enhance level (0-100)", self.enh); f.addRow("Skin retouch (0-100)", self.ret)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel); bb.accepted.connect(self._ok); bb.rejected.connect(self.reject); f.addRow(bb)

    def _paint_bg(self): self.bg_btn.setStyleSheet("background: rgb(%d,%d,%d); border:1px solid #444; padding:6px;" % self.bg)
    def _pick(self):
        c = QColorDialog.getColor(QColor(*self.bg), self, "Background")
        if c.isValid(): self.bg = (c.red(), c.green(), c.blue()); self._paint_bg()
    def _ok(self):
        if not self.name.text().strip(): return QMessageBox.warning(self, "Preset", "Give the preset a name.")
        self.accept()
    def get(self):
        return self.name.text().strip(), dict(w=self.w.value(), h=self.h.value(), dpi=self.dpi.value(), bg=list(self.bg), copies=self.copies.value(),
                                              paper=self.paper.currentText(), head=self.head.value() / 100.0, top=self.top.value() / 100.0,
                                              enhance=self.enh.value(), retouch=self.ret.value(), service="Passport Photo")

# =====================================================================
#  DOCUMENT STUDIO  (scan photo -> straight, clean, white page -> PDF)
# =====================================================================
class QuadView(QWidget):
    """shows a page; optional 4 draggable corners for perspective correction"""
    def __init__(self):
        super().__init__(); self.setMinimumSize(520, 600); self.setMouseTracking(True)
        self.pm = None; self.iw = self.ih = 1; self.quad = None; self._drag = -1
    def set_image(self, arr, quad=None):
        arr = np.ascontiguousarray(arr); h, w = arr.shape[:2]; self.iw, self.ih = w, h
        s = min(1.0, 1600.0 / max(w, h)); disp = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else arr
        disp = np.ascontiguousarray(disp)
        self.pm = QPixmap.fromImage(QImage(disp.data, disp.shape[1], disp.shape[0], 3 * disp.shape[1], QImage.Format_RGB888).copy())
        self.quad = None if quad is None else np.array(quad, np.float32).copy(); self.update()
    def _geom(self):
        s = min(self.width() / max(1, self.iw), self.height() / max(1, self.ih)); return s, (self.width() - self.iw * s) / 2, (self.height() - self.ih * s) / 2
    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#cfcfcf"))
        if self.pm is None: p.end(); return
        s, ox, oy = self._geom(); p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawPixmap(QRectF(ox, oy, self.iw * s, self.ih * s), self.pm, QRectF(self.pm.rect()))
        if self.quad is not None:
            pts = [QPointF(ox + x * s, oy + y * s) for x, y in self.quad]
            p.setPen(QPen(QColor(0, 200, 120), 2)); p.setBrush(QColor(0, 200, 120, 40)); p.drawPolygon(QPolygonF(pts))
            p.setPen(QPen(QColor(255, 255, 255), 2)); p.setBrush(QColor(0, 150, 90))
            for q in pts: p.drawEllipse(q, 8, 8)
        p.end()
    def mousePressEvent(self, e):
        if self.quad is None: return
        s, ox, oy = self._geom(); best, bd = -1, 22.0
        for i, (x, y) in enumerate(self.quad):
            d = math.hypot(ox + x * s - e.position().x(), oy + y * s - e.position().y())
            if d < bd: best, bd = i, d
        self._drag = best
    def mouseMoveEvent(self, e):
        if self._drag >= 0 and self.quad is not None:
            s, ox, oy = self._geom()
            self.quad[self._drag] = (min(max((e.position().x() - ox) / s, 0), self.iw - 1), min(max((e.position().y() - oy) / s, 0), self.ih - 1)); self.update()
    def mouseReleaseEvent(self, e): self._drag = -1

class DocStudio(QDialog):
    MAXSIDE = 3600
    def __init__(self, app):
        super().__init__(app); self.app = app; self.pages = []; self.cur = -1
        self.setWindowTitle("Document Studio  -  scan / clean / PDF"); self.resize(1280, 860)
        self.runner = TaskRunner(self); self.runner.progress.connect(self._prog)
        self.banner = ProcessingBanner(); self.banner.cancel_clicked.connect(self.runner.cancel); self.banner.idle("Add document photos to begin")
        self.list = QListWidget(); self.list.setViewMode(QListWidget.IconMode); self.list.setIconSize(QSize(96, 124)); self.list.setFixedWidth(150)
        self.list.setMovement(QListWidget.Static); self.list.setResizeMode(QListWidget.Adjust); self.list.setSpacing(6)
        self.list.currentRowChanged.connect(self._select)
        self.view = QuadView()
        self.s_b = hslider(-60, 60, 0); self.s_c = hslider(-60, 60, 0)
        self.mode = QComboBox(); self.mode.addItems(["Color", "Gray", "Black & White"])
        self.s_white = hslider(10, 100, 70); self.s_sharp = hslider(10, 100, 60)
        self._deb = QTimer(self); self._deb.setSingleShot(True); self._deb.setInterval(80); self._deb.timeout.connect(self._refresh)
        for w in (self.s_b, self.s_c): w.valueChanged.connect(lambda *_: self._on_adjust())
        self.mode.currentIndexChanged.connect(lambda *_: self._on_adjust())
        self.c_a4 = QCheckBox("Fit every page on A4 (white margin)"); self.c_a4.setChecked(True)
        side = QVBoxLayout()
        side.addWidget(group("1. ADD PAGES", [row(btn("Add Images...", self.add_files), btn("From main photo", self.from_main)),
                                              row(btn("Remove page", self.remove), btn("▲", lambda: self.move_page(-1)), btn("▼", lambda: self.move_page(1)))]))
        side.addWidget(group("2. STRAIGHTEN", [btn("Auto Detect Edges", self.detect, tip="Find the paper corners, then drag the green dots if needed"),
                                              btn("Apply Perspective Correction", self.apply_quad, "background:#1565c0;color:white;font-weight:bold;"),
                                              row(btn("⟲ Rotate", lambda: self.rotate(-1)), btn("Rotate ⟳", lambda: self.rotate(1)))]))
        side.addWidget(group("3. CLEAN", [btn("Shadow Remove", self.shadow), QLabel("White background strength"), self.s_white, btn("White Background", self.whiten),
                                          QLabel("Sharpen strength"), self.s_sharp, btn("Sharpen Text", self.sharpen)]))
        side.addWidget(group("4. LOOK", [QLabel("Brightness"), self.s_b, QLabel("Contrast"), self.s_c, self.mode,
                                        row(btn("Undo step", self.undo), btn("Reset page", self.reset))]))
        side.addWidget(group("SIGNATURE", [btn("Signature Crop (replace page)", self.sig_replace), btn("Signature -> transparent PNG...", self.sig_png)]))
        side.addWidget(group("5. OUTPUT (A4)", [self.c_a4, btn("Save PDF (all pages)", self.save_pdf, "background:#2e7d32;color:white;font-weight:bold;"),
                                                 btn("Save pages as JPG...", self.save_jpgs), btn("Print...", self.print_pages)]))
        side.addStretch(); pan = QWidget(); pan.setLayout(side); sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(pan); sc.setFixedWidth(330)
        body = QHBoxLayout(); body.addWidget(self.list); body.addWidget(self.view, 1); body.addWidget(sc)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6); lay.addWidget(self.banner); lay.addLayout(body)

    # ---- background helper ----
    def _prog(self, text, i, n):
        if n < 0: self.banner.set_fraction(i / float(-n), text)
        else: self.banner.progress_update(text, i, n)
    def _run(self, label, fn, apply):
        if self.runner.busy: return
        self.banner.start(label)
        def ok(res):
            try: apply(res); self.banner.done(label.replace("...", "") + " - done")
            except Exception as e: self.banner.error(str(e)); log.exception("doc apply")
        self.runner.submit(fn, ok, lambda m, e: self.banner.error("Cancelled" if m == "Cancelled" else m))

    # ---- pages ----
    def _page(self): return self.pages[self.cur] if 0 <= self.cur < len(self.pages) else None
    def add_pil(self, im):
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > self.MAXSIDE: im.thumbnail((self.MAXSIDE, self.MAXSIDE), Image.LANCZOS)
        a = np.ascontiguousarray(np.array(im))
        self.pages.append(dict(src=a, base=a, quad=None, hist=[], b=0, c=0, mode="Color")); self._rebuild(len(self.pages) - 1)
    def add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Document photos", SETTINGS.get("last_dir", ""), "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)")
        for p in ps:
            try: self.add_pil(Image.open(p))
            except Exception as e: QMessageBox.warning(self, "Open", "%s\n%s" % (p, e))
        if ps: SETTINGS["last_dir"] = os.path.dirname(ps[0])
    def from_main(self):
        if self.app.rgb is None: return QMessageBox.information(self, "Document Studio", "No photo is open in the main window.")
        self.add_pil(self.app.composite())
    def _thumb(self, p):
        im = Image.fromarray(self._render(p)); im.thumbnail((96, 124)); return QIcon(pil_to_pixmap(im))
    def _rebuild(self, select=None):
        self.list.blockSignals(True); self.list.clear()
        for i, p in enumerate(self.pages): self.list.addItem(QListWidgetItem(self._thumb(p), "Page %d" % (i + 1)))
        self.list.blockSignals(False)
        if self.pages:
            self.list.setCurrentRow(select if select is not None else max(0, min(self.cur, len(self.pages) - 1))); self._select(self.list.currentRow())
        else: self.cur = -1; self.view.pm = None; self.view.update()
    def _select(self, r):
        self.cur = r; self._refresh()
    def remove(self):
        if self._page(): del self.pages[self.cur]; self._rebuild(min(self.cur, len(self.pages) - 1))
    def move_page(self, d):
        j = self.cur + d
        if self._page() and 0 <= j < len(self.pages):
            self.pages[self.cur], self.pages[j] = self.pages[j], self.pages[self.cur]; self._rebuild(j)

    # ---- rendering ----
    def _render(self, p):
        return doc_adjust(p["base"], p["b"], p["c"], p["mode"])
    def _refresh(self):
        p = self._page()
        if p is None: return
        self.s_b.blockSignals(True); self.s_c.blockSignals(True); self.mode.blockSignals(True)
        self.s_b.setValue(p["b"]); self.s_c.setValue(p["c"]); self.mode.setCurrentText(p["mode"])
        self.s_b.blockSignals(False); self.s_c.blockSignals(False); self.mode.blockSignals(False)
        self.view.set_image(np.array(self._render(p)), p["quad"])
    def _on_adjust(self):
        p = self._page()
        if p is None: return
        p["b"], p["c"], p["mode"] = self.s_b.value(), self.s_c.value(), self.mode.currentText(); self._deb.start()
    def _commit(self, arr, keep_quad=False):
        p = self._page(); p["hist"].append(p["base"]); p["hist"] = p["hist"][-6:]
        p["base"] = np.ascontiguousarray(arr)
        if not keep_quad: p["quad"] = None
        it = self.list.item(self.cur)
        if it: it.setIcon(self._thumb(p))
        self._refresh()

    # ---- operations (all in the background) ----
    def detect(self):
        p = self._page()
        if p is None: return
        def done(q): p["quad"] = q; self._refresh()
        self._run("Finding paper edges...", lambda ctx: detect_doc_quad(p["base"]), done)
    def apply_quad(self):
        p = self._page()
        if p is None: return
        if p["quad"] is None: return QMessageBox.information(self, "Document", "Press 'Auto Detect Edges' first, then adjust the green dots.")
        q = self.view.quad if self.view.quad is not None else p["quad"]; base = p["base"]
        self._run("Straightening page...", lambda ctx: warp_quad(base, q), lambda r: self._commit(r))
    def rotate(self, d):
        p = self._page()
        if p is not None: self._commit(np.rot90(p["base"], -d))
    def shadow(self):
        p = self._page()
        if p is not None: b = p["base"]; self._run("Removing shadows...", lambda ctx: remove_shadows(b), lambda r: self._commit(r))
    def whiten(self):
        p = self._page(); v = self.s_white.value()
        if p is not None: b = p["base"]; self._run("Whitening background...", lambda ctx: whiten_background(b, v), lambda r: self._commit(r))
    def sharpen(self):
        p = self._page(); v = self.s_sharp.value()
        if p is not None: b = p["base"]; self._run("Sharpening...", lambda ctx: sharpen_doc(b, v), lambda r: self._commit(r))
    def undo(self):
        p = self._page()
        if p is not None and p["hist"]: p["base"] = p["hist"].pop(); p["quad"] = None; self._rebuild(self.cur)
    def reset(self):
        p = self._page()
        if p is not None: p["hist"].append(p["base"]); p["base"] = p["src"]; p["quad"] = None; p["b"] = p["c"] = 0; p["mode"] = "Color"; self._rebuild(self.cur)
    def sig_replace(self):
        p = self._page()
        if p is not None: b = p["base"]; self._run("Cropping signature...", lambda ctx: np.array(signature_extract(b).convert("RGB")), lambda r: self._commit(r))
    def sig_png(self):
        p = self._page()
        if p is None: return
        fn, _ = QFileDialog.getSaveFileName(self, "Signature PNG", os.path.join(self.app.out_dir(), unique_path(out_name(self.app.customer_name(), "Signature", ext="png"))), "PNG (*.png)")
        if not fn: return
        b = p["base"]
        self._run("Making transparent signature...", lambda ctx: signature_extract(b, True), lambda im: im.save(unique_path(fn), "PNG"))

    # ---- output ----
    def _a4_pages(self):
        ims = [Image.fromarray(self._render(p)) for p in self.pages]
        return [a4_page(i) for i in ims] if self.c_a4.isChecked() else ims
    def save_pdf(self):
        if not self.pages: return QMessageBox.information(self, "Document", "Add at least one page.")
        start = os.path.join(self.app.out_dir(), out_name(self.app.customer_name(), "Document", ext="pdf"))
        fn, _ = QFileDialog.getSaveFileName(self, "Save PDF", unique_path(start), "PDF (*.pdf)")
        if not fn: return
        if os.path.abspath(fn) in {os.path.abspath(p) for p in self.app.protected_paths()}: return QMessageBox.warning(self, "Save", "Will not overwrite the original file.")
        fn = fn if fn.lower().endswith(".pdf") else fn + ".pdf"
        a4 = self.c_a4.isChecked()
        pages = [self._render(p) for p in self.pages]
        def work(ctx):
            outs = []
            for i, a in enumerate(pages):
                ctx.step(i, len(pages), "Building PDF page %d / %d ..." % (i + 1, len(pages))); im = Image.fromarray(a); outs.append(a4_page(im) if a4 else im)
            save_pdf_pages(outs, fn, 300); return fn
        self._run("Saving PDF...", work, lambda f: self.app.note_saved(f, "Document"))
    def save_jpgs(self):
        if not self.pages: return
        d = QFileDialog.getExistingDirectory(self, "Folder for page images", self.app.out_dir())
        if not d: return
        for i, p in enumerate(self.pages):
            fn = unique_path(os.path.join(d, out_name(self.app.customer_name(), "Document", "p%d" % (i + 1), "jpg"))); Image.fromarray(self._render(p)).save(fn, quality=95, dpi=(300, 300))
        self.banner.done("%d page(s) saved" % len(self.pages))
    def print_pages(self):
        if self.pages: print_sheets(self, self._a4_pages(), 300 if self.c_a4.isChecked() else 300)


# =====================================================================
#  MAIN WINDOW - part 1: state, undo/redo, open photo, face, jobs, recovery
# =====================================================================
INBOX_DIR = os.path.join(JOBS_DIR, "_inbox")
os.makedirs(INBOX_DIR, exist_ok=True)

def load_image_file(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(im.size) > 4096: im.thumbnail((4096, 4096), Image.Resampling.LANCZOS)      # keeps it fast on big phone photos
    return im

class CoreMixin:
    PREVIEW_MAX = 1100

    # ---------- small helpers ----------
    def status_msg(self, text):
        self.status.setText(text)

    def need_img(self):
        if self.rgb is None:
            QMessageBox.information(self, APP_NAME, "Please open a photo first  (📂 Open Photo).")
            return False
        return True

    def size_mm(self): return (self.preset["w"], self.preset["h"])
    def ratio(self): return self.preset["w"] / float(self.preset["h"])
    def customer_name(self): return self.e_customer.text().strip() or "Customer"
    def protected_paths(self): return [p for p in (self.src_path, self.inbox_path) if p]

    def out_dir(self):
        if self.job_id and self.job_folder: d = os.path.join(self.job_folder, "output")
        else: d = os.path.join(os.path.expanduser("~"), "Pictures", "RiponComputer", datetime.date.today().isoformat())
        try: os.makedirs(d, exist_ok=True)
        except Exception: d = os.path.expanduser("~")
        return d

    def source(self):
        im = self.rgb.convert("RGBA")
        if self.alpha is not None: im.putalpha(self.alpha)
        return im

    def composite(self):
        if self.alpha is None: return self.rgb
        c = Image.new("RGB", self.rgb.size, self.bg_rgb); c.paste(self.rgb, (0, 0), self.alpha); return c

    def alpha_np(self):
        if self.alpha is None: return None
        key = ("alpha", id(self.alpha))
        if key not in self._cache: self._cache[key] = np.array(self.alpha, np.float32) / 255.0
        return self._cache[key]

    # ---------- cache / dirty ----------
    def _invalidate_cache(self):
        self._rev += 1; self._cache.clear(); self._last_display_key = None; self._last_display_pixmap = None

    def set_rgb(self, arr_or_img):
        self.rgb = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(np.ascontiguousarray(arr_or_img))
        self._invalidate_cache()

    # ---------- preview ----------
    def _display_size(self):
        return min(self.PREVIEW_MAX, max(320, self.preview.width() - 12)), min(self.PREVIEW_MAX, max(320, self.preview.height() - 12))

    def show_img(self, im):
        if im is None: return
        self.stack.setCurrentIndex(0)
        mw, mh = self._display_size(); key = (id(im), im.size, mw, mh)
        if key == self._last_display_key and self._last_display_pixmap is not None:
            self.preview.setPixmap(self._last_display_pixmap); return
        view = im
        if im.width > mw or im.height > mh:
            view = im.copy(); view.thumbnail((mw, mh), Image.Resampling.LANCZOS)
        pm = pil_to_pixmap(view); self._last_display_key = key; self._last_display_pixmap = pm; self.preview.setPixmap(pm)

    def show_current(self):
        """what the big preview shows: Photo / Print Sheet / Original"""
        if self.stack.currentIndex() != 0 or self.rgb is None: return
        m = self.view_mode
        if m == "orig" and self.orig is not None: self.show_img(self.orig)
        elif m == "sheet" and self.sheets: self.show_img(self.sheets[0])
        elif self.photo is not None: self.show_img(self.photo)
        else: self.show_img(self.composite())

    def set_view(self, mode):
        self.view_mode = mode
        for k, b in self.view_btns.items(): b.setChecked(k == mode)
        if mode == "sheet" and not self.sheets and self.photo is not None: self.make_sheet()
        self.show_current()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._resize_timer.start(90)

    # ---------- undo / redo ----------
    def push(self):
        self.history.append(self._snap()); self.history = self.history[-12:]; self.redo_stack = []

    def _snap(self):
        return dict(rgb=self.rgb, alpha=self.alpha, orig=self.orig, face=self.face, finfo=self.finfo, fi_orig=self.fi_orig,
                    box=list(self.box) if self.box else None, base_h=self.base_h, cropped=self.photo_base is not None)

    def _restore(self, s):
        self.rgb, self.alpha, self.orig, self.face, self.finfo, self.fi_orig = s["rgb"], s["alpha"], s["orig"], s["face"], s["finfo"], s["fi_orig"]
        self.box = list(s["box"]) if s["box"] else None; self.base_h = s["base_h"]
        if not s["cropped"]: self.photo_base = self.photo = None; self.sheets = []
        if self.hair_mask is not None and self.hair_mask.shape != (self.rgb.height, self.rgb.width): self.hair_mask = None
        self._invalidate_cache()
        if self.box: self.on_box_changed()
        if s["cropped"] and self.photo_base is None and self.stack.currentIndex() == 0: self.apply_crop()
        else: self.after_change()

    def undo(self):
        if self.stack.currentIndex() == 2: return self.rt_undo()
        if not self.history: return
        self.redo_stack.append(self._snap()); self._restore(self.history.pop()); self.status_msg("Undo")

    def redo(self):
        if self.stack.currentIndex() == 2 or not self.redo_stack: return
        self.history.append(self._snap()); self._restore(self.redo_stack.pop()); self.status_msg("Redo")

    def reset_orig(self):
        if not self.need_img(): return
        self.push(); self.rgb = self.orig; self.alpha = None; self.fi_orig = self.fi_orig
        self.finfo = self.fi_orig; self.face = fi_face_tuple(self.finfo); self.hair_mask = None; self.box = None
        self.photo_base = self.photo = None; self.sheets = []
        self._invalidate_cache(); self.after_change(); self.status_msg("Back to the original photo")

    def after_change(self):
        if self.rgb is None: return
        self._invalidate_cache()
        if self.stack.currentIndex() == 1: self.refresh_preview()
        elif self.photo_base is not None: self.apply_crop()
        else: self.show_img(self.composite())

    # ---------- open photo (original is copied, never modified) ----------
    def adopt_original(self, path):
        """keep our own read-only copy of the customer's photo; the source file is never written to"""
        self.src_path = path
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = os.path.join(INBOX_DIR, "%s_%s" % (ts, safe_name(os.path.basename(path), "photo")))
            shutil.copy2(path, dst); self.inbox_path = dst
        except Exception as e:
            log.warning("could not copy original: %s", e); self.inbox_path = path

    def open_photo(self, path=None):
        if self.runner.busy: return self.status_msg("Please wait - a job is running.")
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Photo", SETTINGS.get("last_dir", ""), "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)")
        if not path: return
        try: im = load_image_file(path)
        except Exception as e:
            return QMessageBox.critical(self, "Open Photo", "Cannot open this photo.\n%s" % e)
        SETTINGS["last_dir"] = os.path.dirname(path); self.job_id = None; self.job_folder = None
        self.adopt_original(path)
        self.orig = im; self.rgb = im; self.alpha = None; self.history = []; self.redo_stack = []; self.hair_mask = None
        self.finfo = self.fi_orig = None; self.face = None; self.box = None; self.photo_base = self.photo = None; self.sheets = []
        self.view_mode = "photo"; self._invalidate_cache(); self._update_face_label(); self.set_view("photo")
        self.update_quality(); self.status_msg("Photo loaded: %s (%dx%d)" % (os.path.basename(path), im.width, im.height))
        self._mark_dirty()
        if SETTINGS.get("auto_ai") and not self.runner.busy: QTimer.singleShot(50, self.ai_passport)
        else: self.detect_face_async(silent=True)

    # ---------- face (async) ----------
    @property
    def face(self): return getattr(self, "_face", None)
    @face.setter
    def face(self, v):
        self._face = v; self._update_face_label()

    def _update_face_label(self):
        lb = getattr(self, "face_lbl", None)
        if lb is None: return
        if getattr(self, "rgb", None) is None: lb.setText("Face: open a photo"); lb.setStyleSheet("color:#555;")
        elif self.face is not None:
            how = (self.finfo or {}).get("src", "")
            lb.setText("Face: ✔ found" + (" (%s)" % how if how else "")); lb.setStyleSheet("color:#2e7d32;font-weight:bold;")
        else: lb.setText("Face: ✘ not found - use 'Face Manual'"); lb.setStyleSheet("color:#c62828;font-weight:bold;")

    def detect_face_async(self, silent=False, after=None, banner=True):
        if self.rgb is None or self.runner.busy: return False
        arr = np.array(self.rgb); is_orig = self.rgb is self.orig
        def work(ctx):
            ctx.msg("Detecting face..."); return detect_face_info(arr, HUB)
        def ok(fi):
            if self.rgb is None: return
            self.finfo = fi; self.face = fi_face_tuple(fi); self.hair_mask = None
            if is_orig or self.fi_orig is None: self.fi_orig = fi if is_orig else self.fi_orig
            self._invalidate_cache(); self.update_quality()
            if fi is not None:
                if banner: self.banner.done("Face found" + (" - eyes and chin located" if not fi.get("est") else " (basic detector)"))
                if after: after()
            else:
                if banner: self.banner.warn("Face not found - draw a box with Face Manual")
                if not silent: self.open_face_picker("Face not found automatically. Drag a box over the face, then press Enter.", retry=after)
        if banner: self.banner.start("Detecting face...", cancellable=False, label="LOOKING")
        return self.runner.submit(work, ok, lambda m, e: self.banner.error("Face detection failed: %s" % m))

    def m_face_detect(self):
        if self.need_img(): self.detect_face_async(silent=False)

    def m_face_manual(self): self.open_face_picker()

    def open_face_picker(self, msg=None, retry=None):
        if not self.need_img(): return
        if self.stack.currentIndex() in (1, 2): return self.status_msg("Press Enter in Crop / Retouch first, then Face Manual.")
        self._pending_face_action = retry
        self.face_view.set_image(np.array(self.rgb), self.face); self.stack.setCurrentIndex(3)
        self.status_msg(msg or "Drag a box over the face (eyebrows to chin, both ears). Enter = OK, Esc = Cancel")

    def close_face_picker(self, commit=True):
        if self.stack.currentIndex() != 3: return
        box = self.face_view.get_box() if commit else None
        self.stack.setCurrentIndex(0); self.show_current()
        retry, self._pending_face_action = self._pending_face_action, None
        if not commit: return self.status_msg("Face selection cancelled")
        if box is None: return self.status_msg("Box too small. Press Face Manual again.")
        fi = fi_from_box(box)
        # if the better detector can refine eyes inside this box, keep manual box but real eye points where possible
        self.finfo = fi; self.face = fi_face_tuple(fi); self.hair_mask = None
        if self.rgb is self.orig: self.fi_orig = fi
        self._invalidate_cache(); self.update_quality(); self.status_msg("Face set. Tools can use it now.")
        if retry: QTimer.singleShot(0, retry)

    def _need_face(self, retry=None):
        """True when a face is known. Otherwise detect in the background, then run `retry` automatically (or ask for a manual box)."""
        if not self.need_img(): return False
        if self.face is not None: return True
        if self.runner.busy: self.status_msg("Please wait - a job is running."); return False
        self.detect_face_async(silent=False, after=retry); return False

    # ---------- background colour ----------
    def set_bg(self, rgb):
        self.bg_rgb = tuple(int(v) for v in rgb)
        self.swatch.setStyleSheet("background: rgb(%d,%d,%d); border:1px solid #444;" % self.bg_rgb)
        self.preset["bg"] = list(self.bg_rgb)
        if self.rgb is not None: self.after_change()

    def pick_bg(self):
        c = QColorDialog.getColor(QColor(*self.bg_rgb), self, "Background Color")
        if c.isValid(): self.set_bg((c.red(), c.green(), c.blue()))

    # ---------- presets ----------
    def fill_preset_box(self):
        self.preset_box.blockSignals(True); self.preset_box.clear()
        self.preset_box.addItems(list(self.presets.keys())); self.preset_box.addItem("➕ New size / preset...")
        self.preset_box.setCurrentText(self.preset_name); self.preset_box.blockSignals(False)

    def on_preset_pick(self, i):
        name = self.preset_box.itemText(i)
        if name.startswith("➕"):
            self.preset_box.blockSignals(True); self.preset_box.setCurrentText(self.preset_name); self.preset_box.blockSignals(False)
            return self.preset_new()
        self.apply_preset(name)

    def apply_preset(self, name, reframe=True):
        if name not in self.presets: return
        self.preset_name = name; self.preset = dict(self.presets[name]); p = self.preset
        set_dpi(p.get("dpi", 300))
        self.preset_box.blockSignals(True); self.preset_box.setCurrentText(name); self.preset_box.blockSignals(False)
        self.copies.blockSignals(True); self.copies.setValue(int(p.get("copies", 6))); self.copies.blockSignals(False)
        if p.get("paper") in PAPERS_MM: self.paper_box.blockSignals(True); self.paper_box.setCurrentText(p["paper"]); self.paper_box.blockSignals(False)
        self.crop_view.ratio = self.ratio(); self.crop_view.guide = (p.get("top", 0.09), p.get("head", 0.74)); self.crop_view.update()
        self.s_enh.setValue(int(p.get("enhance", 50)))
        if p.get("service") in SERVICES: self.e_service.setCurrentText(p["service"])
        self.set_bg_silent(p.get("bg", (255, 255, 255)))
        self.size_lbl.setText("%g x %g mm   %d DPI" % (p["w"], p["h"], p.get("dpi", 300)))
        if self.rgb is not None and self.stack.currentIndex() == 0 and (self.photo_base is not None):
            if self.finfo is not None and reframe: self.box = ideal_box(self.finfo, self.preset, self.alpha_np()); self.base_h = self.box[2]
            self.apply_crop()
        self.update_layout_info(); self._mark_dirty()

    def set_bg_silent(self, rgb):
        self.bg_rgb = tuple(int(v) for v in rgb)
        self.swatch.setStyleSheet("background: rgb(%d,%d,%d); border:1px solid #444;" % self.bg_rgb)

    def preset_new(self):
        d = PresetDialog(self, "My Preset", dict(self.preset))
        if d.exec():
            name, data = d.get()
            if name in BUILTIN_PRESETS and name in self.presets: name += " (mine)"
            self.presets[name] = data; store_presets(self.presets); self.fill_preset_box(); self.apply_preset(name)

    def preset_edit(self):
        builtin = self.preset_name in BUILTIN_PRESETS
        d = PresetDialog(self, self.preset_name, dict(self.presets[self.preset_name]), builtin)
        if d.exec():
            name, data = d.get(); self.presets[self.preset_name if builtin else name] = data
            if not builtin and name != self.preset_name: self.presets.pop(self.preset_name, None)
            store_presets(self.presets); self.preset_name = self.preset_name if builtin else name
            self.fill_preset_box(); self.apply_preset(self.preset_name)

    def preset_delete(self):
        n = self.preset_name
        if n in BUILTIN_PRESETS:
            if n in self._saved_builtin_overrides():
                if ask_yes_no(self, "Preset", "Reset '%s' to the original values?" % n):
                    self.presets[n] = dict(BUILTIN_PRESETS[n]); store_presets(self.presets); self.apply_preset(n)
            else: QMessageBox.information(self, "Preset", "Built-in presets cannot be deleted. You can edit them with ✎.")
            return
        if len(self.presets) > 1 and ask_yes_no(self, "Delete preset", "Delete preset '%s'?" % n):
            self.presets.pop(n, None); store_presets(self.presets); self.preset_name = next(iter(self.presets))
            self.fill_preset_box(); self.apply_preset(self.preset_name)

    def _saved_builtin_overrides(self):
        try:
            with open(PRESETS_PATH, "r", encoding="utf-8") as f: return set(json.load(f).keys())
        except Exception: return set()

    # ---------- customer jobs ----------
    def _job_state(self):
        return dict(preset=self.preset_name, bg=list(self.bg_rgb), copies=self.copies.value(), paper=self.paper_box.currentText(),
                    caption=self.cap_edit.text(), date=self.c_date.isChecked(), bw=self.c_bw.isChecked(), cut=self.cut_box.currentText(),
                    box=self.box, base_h=self.base_h, face=self.face, finfo=self.finfo, dress=self.dress_path,
                    dress_s=[self.s_dress.value(), self.s_dress_y.value(), self.s_dress_x.value()],
                    bright=self.s_bright.value(), contrast=self.s_contrast.value(), cropped=self.photo_base is not None)

    def _job_fields(self):
        return dict(customer=self.customer_name() if self.e_customer.text().strip() else "Walk-in", phone=self.e_phone.text().strip(),
                    service=self.e_service.currentText(), copies=self.copies.value(), price=float(self.e_price.value()),
                    preset=self.preset_name, size_txt="%gx%g" % self.size_mm(), paper=self.paper_box.currentText())

    def save_job(self, quiet=False):
        """create or update the job record, keep the original + working files in the job's own folder"""
        if self.rgb is None and not self.e_customer.text().strip():
            return QMessageBox.information(self, "Job", "Type the customer name, or open a photo first.")
        f = self._job_fields()
        if self.job_id is None:
            self.job_id = self.db.add_job(**f)
        else:
            self.db.update_job(self.job_id, **f)
        self.job_folder = os.path.join(JOBS_DIR, "%04d_%s" % (self.job_id, safe_name(f["customer"])))
        os.makedirs(os.path.join(self.job_folder, "output"), exist_ok=True)
        src = self.inbox_path
        if src and os.path.exists(src):
            dst = os.path.join(self.job_folder, "original" + os.path.splitext(src)[1].lower())
            if not os.path.exists(dst): shutil.copy2(src, dst)
            self.inbox_path = dst; self.db.update_job(self.job_id, source=dst)
        if self.rgb is not None:
            self.rgb.save(os.path.join(self.job_folder, "work_rgb.png"), "PNG", compress_level=1)
            wa = os.path.join(self.job_folder, "work_alpha.png")
            if self.alpha is not None: self.alpha.save(wa, "PNG", compress_level=1)
            elif os.path.exists(wa): os.remove(wa)
        self.db.update_job(self.job_id, folder=self.job_folder, state=json.dumps(self._job_state()))
        self.job_lbl.setText("Job #%d" % self.job_id)
        if not quiet: self.banner.done("Job #%d saved (%s)" % (self.job_id, taka(f["price"])))

    def job_new(self):
        d = JobEditDialog(self, dict(customer=self.e_customer.text(), phone=self.e_phone.text(), service=self.e_service.currentText(),
                                     copies=self.copies.value(), price=self.e_price.value()))
        if not d.exec(): return
        v = d.get(); self.job_id = None
        self.e_customer.setText(v["customer"]); self.e_phone.setText(v["phone"]); self.e_service.setCurrentText(v["service"])
        self.copies.setValue(v["copies"]); self.e_price.setValue(v["price"]); self.save_job()

    def job_open(self, jid):
        j = self.db.get_job(jid)
        if not j: return
        if self.runner.busy: return self.status_msg("Please wait - a job is running.")
        self.e_customer.setText(j["customer"] if j["customer"] != "Walk-in" else ""); self.e_phone.setText(j["phone"] or "")
        self.e_service.setCurrentText(j["service"] or SERVICES[0]); self.e_price.setValue(float(j["price"] or 0)); self.copies.setValue(int(j["copies"] or 1))
        self.job_id = jid; self.job_folder = j.get("folder"); self.job_lbl.setText("Job #%d" % jid)
        try: st = json.loads(j.get("state") or "{}")
        except Exception: st = {}
        folder = self.job_folder or ""
        orig_p = j.get("source") if j.get("source") and os.path.exists(j["source"]) else None
        wr = os.path.join(folder, "work_rgb.png"); wa = os.path.join(folder, "work_alpha.png")
        try:
            if orig_p: self.orig = load_image_file(orig_p); self.inbox_path = orig_p; self.src_path = orig_p
            if os.path.exists(wr):
                self.rgb = Image.open(wr).convert("RGB"); self.rgb.load()
                self.alpha = Image.open(wa).convert("L") if os.path.exists(wa) else None
                if self.orig is None: self.orig = self.rgb
            elif self.orig is not None: self.rgb = self.orig; self.alpha = None
            else: return QMessageBox.information(self, "Job", "This job has no saved photo yet.")
        except Exception as e:
            return QMessageBox.warning(self, "Job", "Could not load the job photo.\n%s" % e)
        self.history = []; self.redo_stack = []; self.hair_mask = None
        if st.get("preset") in self.presets: self.apply_preset(st["preset"], reframe=False)
        if st.get("bg"): self.set_bg_silent(st["bg"])
        self.cap_edit.setText(st.get("caption", "")); self.c_date.setChecked(bool(st.get("date"))); self.c_bw.setChecked(bool(st.get("bw")))
        if st.get("cut") in ("None", "Light border", "Cut marks"): self.cut_box.setCurrentText(st["cut"])
        self.finfo = st.get("finfo"); self.face = tuple(st["face"]) if st.get("face") else None
        if self.finfo:
            for k in ("le", "re", "nose", "ml", "mr", "chin", "crown"):
                if self.finfo.get(k) is not None: self.finfo[k] = tuple(self.finfo[k])
            self.finfo["bbox"] = tuple(self.finfo["bbox"])
        self.fi_orig = self.finfo if self.rgb is self.orig else None
        self.box = st.get("box"); self.base_h = st.get("base_h", self.box[2] if self.box else 1.0)
        for sl, v in ((self.s_bright, st.get("bright", 0)), (self.s_contrast, st.get("contrast", 0))):
            sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)
        ds = st.get("dress_s")
        if ds: self.s_dress.setValue(ds[0]); self.s_dress_y.setValue(ds[1]); self.s_dress_x.setValue(ds[2] if len(ds) > 2 else 0)
        if st.get("dress") and os.path.exists(st["dress"]): self.load_dress(st["dress"])
        self.photo_base = self.photo = None; self.sheets = []; self.view_mode = "photo"; self._invalidate_cache(); self._update_face_label()
        if st.get("cropped") and self.box: self.apply_crop()
        else: self.show_img(self.composite())
        self.set_view("photo"); self.banner.done("Job #%d opened" % jid)

    def job_export(self, jid):
        j = self.db.get_job(jid)
        if not j or not j.get("folder"): return QMessageBox.information(self, "Export", "This job has no saved files.")
        out = os.path.join(j["folder"], "output"); files = [f for f in os.listdir(out)] if os.path.isdir(out) else []
        if not files: return QMessageBox.information(self, "Export", "No final files yet. Open the job, press Save, then export.")
        d = QFileDialog.getExistingDirectory(self, "Copy final files to...", SETTINGS.get("last_dir", ""))
        if not d: return
        for f in files: shutil.copy2(os.path.join(out, f), unique_path(os.path.join(d, f)))
        self.banner.done("%d file(s) copied to %s" % (len(files), d))

    def note_saved(self, path, kind=""):
        self._saved_rev = self._rev; self.status_msg("Saved: %s" % path)
        self.banner.done("Saved  %s" % os.path.basename(path))
        self._last_saved_dir = os.path.dirname(path)

    # ---------- auto-save / recovery ----------
    def _mark_dirty(self): self._dirty = True

    def _autosave(self):
        if self.rgb is None or self.runner.busy or not self._dirty: return
        self._dirty = False
        st = self._job_state(); st["orig_path"] = self.inbox_path; st["job_id"] = self.job_id
        st["customer"] = self.e_customer.text(); st["phone"] = self.e_phone.text(); st["service"] = self.e_service.currentText(); st["price"] = self.e_price.value()
        try: json.dumps(st)
        except Exception: st["finfo"] = None
        Recovery.save_async(st, self.rgb, self.alpha)

    def try_recover(self):
        if not Recovery.crashed(): Recovery.begin(); return
        if ask_yes_no(self, "Recover previous work?", "The program did not close normally last time.\n\nRecover the previous work?"):
            r = Recovery.load()
            if r:
                st, rgb, alpha = r
                try:
                    op = st.get("orig_path"); self.inbox_path = op if op and os.path.exists(op) else None; self.src_path = self.inbox_path
                    self.orig = load_image_file(op) if self.inbox_path else rgb
                    self.rgb, self.alpha = rgb, alpha
                    if st.get("preset") in self.presets: self.apply_preset(st["preset"], reframe=False)
                    if st.get("bg"): self.set_bg_silent(st["bg"])
                    self.copies.setValue(int(st.get("copies", self.copies.value()))); self.e_customer.setText(st.get("customer", "")); self.e_phone.setText(st.get("phone", ""))
                    self.e_price.setValue(float(st.get("price", 0) or 0))
                    self.finfo = st.get("finfo"); self.face = tuple(st["face"]) if st.get("face") else None
                    if self.finfo:
                        for k in ("le", "re", "nose", "ml", "mr", "chin", "crown"):
                            if self.finfo.get(k) is not None: self.finfo[k] = tuple(self.finfo[k])
                        self.finfo["bbox"] = tuple(self.finfo["bbox"])
                    self.fi_orig = self.finfo if self.rgb is self.orig else None
                    self.box = st.get("box"); self.base_h = st.get("base_h", self.box[2] if self.box else 1.0)
                    self.cap_edit.setText(st.get("caption", "")); self.c_bw.setChecked(bool(st.get("bw")))
                    self.job_id = st.get("job_id")
                    self._invalidate_cache()
                    if st.get("cropped") and self.box: self.apply_crop()
                    else: self.show_img(self.composite())
                    self._update_face_label(); self.banner.done("Previous work recovered")
                except Exception as e:
                    log.exception("recover"); QMessageBox.warning(self, "Recover", "Could not restore everything.\n%s" % e)
            else:
                QMessageBox.information(self, "Recover", "The recovery files could not be read.")
        Recovery.end(); Recovery.begin()


# =====================================================================
#  MAIN WINDOW - part 2: editing tools (all heavy work in the background)
# =====================================================================
class EditMixin:
    def _bg_op(self, label, fn, apply, history=True, cancellable=True):
        """run fn(ctx) off the GUI thread; apply(result) on the GUI thread. Duplicate clicks are ignored."""
        if self.runner.busy:
            self.status_msg("Please wait - another job is running."); return False
        if history: self.push()
        self.banner.start(label, cancellable)
        def ok(res):
            apply(res); self._mark_dirty()
            if self._fix_queue: QTimer.singleShot(40, self._next_fix)
        def fail(msg, exc):
            if history and self.history: self.history.pop()
            self._fix_queue = []
            if msg == "Cancelled": return self.banner.idle("Cancelled")
            self.banner.error("%s failed" % label.rstrip("."))
            QMessageBox.warning(self, APP_NAME, "%s\n\n%s" % (label.rstrip("."), msg[:800]))
        return self.runner.submit(fn, ok, fail)

    def _done_rgb(self, arr, msg):
        self.set_rgb(arr); self.after_change(); self.banner.done(msg); self.status_msg(msg)

    # ---------- background ----------
    def _segment_job(self, arr, model, edge, face_pt):
        def fn(ctx):
            ctx.msg("Removing background...")
            chain = [model] + (["u2net_human_seg"] if model != "u2net_human_seg" else []); last = None
            for m in chain:
                if not HUB.lib_available(m): last = RuntimeError("Background AI is not installed (pip install rembg onnxruntime)"); continue
                try: a = HUB.segment(arr, m); break
                except Exception as e: last = e; a = None
            if a is None: raise RuntimeError(str(last))
            ctx.msg("Refining hair edges..."); ctx.check()
            a = refine_alpha(arr, a, edge, face_pt)
            return decontaminate(arr, a), a
        return fn

    def _face_pt(self):
        f = self.finfo
        if f is None: return None
        return ((f["le"][0] + f["re"][0]) / 2.0, (f["le"][1] + f["re"][1]) / 2.0 + 0.2 * f["bbox"][3])

    def m_bg_remove(self):
        if not self.need_img(): return
        arr = np.array(self.rgb); model = MODELS[self.model_box.currentText()]
        def apply(res):
            a_rgb, a = res; self.set_rgb(a_rgb); self.alpha = Image.fromarray((a * 255).astype(np.uint8)); self._invalidate_cache()
            self.after_change(); self.banner.done("Background removed - pick any color")
        self._bg_op("Removing background...", self._segment_job(arr, model, self.s_edge.value(), self._face_pt()), apply)

    def m_bg_restore(self):
        if self.need_img(): self.push(); self.alpha = None; self.after_change(); self.status_msg("Original background restored")

    def m_edge_refine(self):
        if not self.need_img(): return
        if self.alpha is None: return QMessageBox.information(self, "Hair Edge", "Remove the background first.")
        arr = np.array(self.rgb); a0 = np.array(self.alpha, np.float32) / 255.0; v = self.s_edge.value(); fp = self._face_pt()
        def fn(ctx): a = refine_alpha(arr, a0, v, fp); return decontaminate(arr, a), a
        def apply(res):
            self.set_rgb(res[0]); self.alpha = Image.fromarray((res[1] * 255).astype(np.uint8)); self._invalidate_cache(); self.after_change(); self.banner.done("Hair edges refined")
        self._bg_op("Refining hair edges...", fn, apply)

    # ---------- enhance / skin ----------
    def m_enhance(self):
        if not self.need_img(): return
        arr, s = np.array(self.rgb), self.s_enh.value()
        self._bg_op("Enhancing photo...", lambda ctx: enhance_pro(arr, s), lambda r: self._done_rgb(r, "Photo enhanced"))

    def m_auto_enhance(self):
        if not self.need_img(): return
        arr, fi = np.array(self.rgb), self.finfo; lvl = int(self.preset.get("enhance", 50))
        self._bg_op("Auto enhance...", lambda ctx: exposure_fix(gentle_enhance(arr, lvl), fi), lambda r: self._done_rgb(r, "Auto enhance done"))

    def _skin_op(self, label, fn_skin, msg, retry):
        if not self._need_face(retry): return
        arr, face, fi = np.array(self.rgb), self.face, self.finfo
        def fn(ctx):
            m = feature_guard(skin_mask(arr, face), fi); return fn_skin(arr, m, face)
        self._bg_op(label, fn, lambda r: self._done_rgb(r, msg))

    def m_smooth(self):  s = self.s_smooth.value(); self._skin_op("Skin smooth...", lambda a, m, f: skin_smooth(a, m, f, s), "Skin smoothed", self.m_smooth)
    def m_tone(self):    s = self.s_tone.value(); self._skin_op("Skin tone...", lambda a, m, f: skin_tone_up(a, m, s), "Skin tone adjusted", self.m_tone)
    def m_retouch(self): s = self.s_smooth.value(); self._skin_op("Skin retouch...", lambda a, m, f: skin_retouch(a, m, f, s), "Skin retouched (natural)", self.m_retouch)
    def m_shine(self):   self._skin_op("Reducing oily shine...", lambda a, m, f: skin_shine_fix(a, m, 70), "Oily shine reduced", self.m_shine)

    def m_pimple(self):
        if not self._need_face(self.m_pimple): return
        arr, face, s = np.array(self.rgb), self.face, self.s_pimple.value()
        def apply(res):
            a, n = res; self._done_rgb(a, "%d spot(s) removed. Use Retouch Studio for the rest." % n)
        self._bg_op("Looking for pimples / spots...", lambda ctx: remove_blemishes(arr, face, s), apply)

    def m_face_ai(self):
        if not self.need_img(): return
        if not (HUB.is_installed("gfpgan") and HUB.lib_available("gfpgan")):
            if ask_yes_no(self, "AI Face Restore", "AI Face Restore is not available (not installed / not downloaded).\n\nContinue with normal enhancement instead?\n(Open 'AI Models' to install it.)"):
                self.m_auto_enhance()
            return
        arr, w = np.array(self.rgb), self.s_aiface.value() / 100.0
        def fn(ctx): r = HUB.restore_face(arr, w); return r
        def apply(r):
            self._done_rgb(r, "AI Face Restore done")
            if HUB.notes: self.banner.warn(HUB.notes.pop()); HUB.notes.clear()
        self._bg_op("AI Face Restore...", fn, apply)

    def m_upscale(self):
        if not self.need_img(): return
        if max(self.rgb.size) >= 2400: return self.status_msg("Photo is already large (2400px+), no upscale needed")
        im = self.rgb.copy(); old = self.rgb.size
        def apply(res):
            big, how = res; k = big.width / float(old[0])
            if self.orig is not None and self.orig.size == old:
                self.orig = self.orig.resize(big.size, Image.LANCZOS)
                if self.fi_orig is not None: self.fi_orig = _scale_fi(self.fi_orig, k)
            if self.alpha is not None: self.alpha = self.alpha.resize(big.size, Image.LANCZOS)
            self.set_rgb(big)
            if self.finfo is not None: self.finfo = _scale_fi(self.finfo, k); self.face = fi_face_tuple(self.finfo)
            if self.box is not None: self.box = [self.box[0] * k, self.box[1] * k, self.box[2] * k, self.box[3]]; self.base_h *= k
            self.hair_mask = None; self.after_change(); self.banner.done("Upscaled x2: " + how)
        self._bg_op("Upscaling x2...", lambda ctx: HUB.upscale(im, 2), apply)

    # ---------- hair colour ----------
    def hair_detect(self, then=None):
        if not self._need_face(then or self.m_hair): return False
        arr, face = np.array(self.rgb), self.face; al = self.alpha_np()
        def fn(ctx):
            person = al
            if person is None:
                try: person = HUB.segment(arr, "u2net_human_seg")
                except Exception: person = None
            return hair_mask_auto(arr, face, person)
        def apply(m):
            self.hair_mask = m
            if self.stack.currentIndex() == 2: self.paint_view.set_overlay(m)
            self.banner.done("Hair area ready")
            if then: QTimer.singleShot(0, then)
        return self._bg_op("Finding hair area...", fn, apply, history=False)

    def hair_custom(self):
        c = QColorDialog.getColor(QColor(*self.hair_rgb), self, "Hair Color")
        if c.isValid(): self.hair_rgb = (c.red(), c.green(), c.blue()); self.status_msg("Custom hair color set. Press 'Apply Hair Color'.")

    def on_hair_pick(self, t):
        if t in HAIR_COLORS: self.hair_rgb = HAIR_COLORS[t]

    def m_hair(self, color=None):
        if not self.need_img(): return
        if self.hair_mask is None or self.hair_mask.shape != (self.rgb.height, self.rgb.width):
            self.hair_detect(lambda: self.m_hair(color)); return
        if (self.hair_mask > 0.5).sum() < 200:
            return QMessageBox.information(self, "Hair", "Hair area was not found automatically. Use 'Hair Area (brush)' to paint it, then apply again.")
        arr, hm, col, st = np.array(self.rgb), self.hair_mask.copy(), color or self.hair_rgb, self.s_hair.value()
        self._bg_op("Changing hair color...", lambda ctx: apply_hair_color(arr, hm, col, st), lambda r: self._done_rgb(r, "Hair color changed"))

    def hair_edit(self): self.open_retouch("hair_add")

    # ---------- Retouch Studio ----------
    def open_retouch(self, tool="heal"):
        if not self.need_img() or self.stack.currentIndex() == 1: return
        if self.stack.currentIndex() != 2:
            self.rt_work = np.ascontiguousarray(np.array(self.rgb)); self.rt_stack = []
            self.paint_view.set_image(self.rt_work); self.paint_view.dirty_hair = False
            ok = self.hair_mask is not None and self.hair_mask.shape == self.rt_work.shape[:2]
            self.paint_view.set_overlay(self.hair_mask if ok else None); self.stack.setCurrentIndex(2)
        i = RT_TOOLS.index(tool)
        self.rt_tool_box.blockSignals(True); self.rt_tool_box.setCurrentIndex(i); self.rt_tool_box.blockSignals(False); self.on_rt_tool(i)

    def on_rt_tool(self, i):
        if self.stack.currentIndex() != 2: return
        tool = RT_TOOLS[i]; self.paint_view.tool = tool
        if tool.startswith("hair") and self.paint_view.ov is None and self.face is not None:
            if self.hair_mask is not None and self.hair_mask.shape == (self.rgb.height, self.rgb.width): self.paint_view.set_overlay(self.hair_mask)
            else: self.hair_mask = hair_mask_auto(np.array(self.rgb), self.face, self.alpha_np()); self.paint_view.set_overlay(self.hair_mask)
        self.paint_view.update()
        self.status_msg("Click on a pimple (brush a little bigger than it). Scroll = zoom, Right-drag = move. Enter = Done, Esc = Cancel"
                        if tool == "heal" else "Red = hair. Brush adds, eraser removes. Enter = Done, then 'Apply Hair Color'")

    def on_brush(self, v): self.paint_view.radius = v; self.paint_view.update()

    def on_paint_action(self, kind, x, y):
        if kind != "heal" or self.rt_work is None: return
        res = heal_spot(self.rt_work, x, y, self.paint_view.radius)
        if res: self.rt_stack.append(res); self.paint_view.refresh(self.rt_work)
        else: self.status_msg("Too close to the photo edge - click a little more inside")

    def rt_undo(self):
        if not self.rt_stack: return
        x0, y0, old = self.rt_stack.pop(); h, w = old.shape[:2]
        self.rt_work[y0:y0 + h, x0:x0 + w] = old; self.paint_view.refresh(self.rt_work)

    def close_retouch(self, commit=True):
        if self.stack.currentIndex() != 2: return
        pv = self.paint_view; changed = commit and bool(self.rt_stack)
        if commit and pv.dirty_hair:
            m = pv.get_overlay_mask()
            if m is not None and m.shape == (self.rgb.height, self.rgb.width): self.hair_mask = cv2.GaussianBlur(m, (0, 0), 1.5)
        if changed: self.push(); self.set_rgb(self.rt_work)
        self.rt_work = None; self.rt_stack = []; pv.clear(); self.stack.setCurrentIndex(0)
        if changed: self.after_change(); self._mark_dirty()
        else: self.show_current()
        self.status_msg("Retouch applied" if changed else "Retouch closed")

    # ---------- dress ----------
    def pick_dress(self):
        p, _ = QFileDialog.getOpenFileName(self, "Dress PNG", DRESS_DIR if os.path.isdir(DRESS_DIR) else "", "PNG (*.png)")
        if p: self.load_dress(p)

    def load_dress(self, p):
        try: self.dress = Image.open(p).convert("RGBA") if p else None
        except Exception as e: return QMessageBox.critical(self, "Dress", str(e))
        self.dress_path = p
        self.apply_adjust(); self.status_msg("Dress applied" if p else "Dress removed")

    def dress_files(self):
        """{category: [paths]} - sub-folders Shirt/Suit/Tie/Coat/... become categories, loose PNGs go to 'Other'"""
        out = {}
        if not os.path.isdir(DRESS_DIR): return out
        for name in sorted(os.listdir(DRESS_DIR)):
            full = os.path.join(DRESS_DIR, name)
            if os.path.isdir(full):
                fs = sorted(os.path.join(full, f) for f in os.listdir(full) if f.lower().endswith(".png"))
                if fs: out[name] = fs
            elif name.lower().endswith(".png"): out.setdefault("Other", []).append(full)
        return out

    def fill_dress_list(self, *_):
        cat = self.dress_cat.currentText(); self.dress_list.clear()
        for p in self._dress_cache.get(cat, []):
            it = QListWidgetItem(os.path.splitext(os.path.basename(p))[0]); it.setData(Qt.UserRole, p)
            try:
                im = Image.open(p).convert("RGBA"); im.thumbnail((84, 84)); bg = Image.new("RGBA", im.size, (235, 235, 235, 255)); bg.alpha_composite(im)
                it.setIcon(QIcon(pil_to_pixmap(bg.convert("RGB"))))
            except Exception: pass
            self.dress_list.addItem(it)

    def refresh_dress_tab(self):
        self._dress_cache = self.dress_files(); self.dress_cat.blockSignals(True); self.dress_cat.clear()
        self.dress_cat.addItems(list(self._dress_cache.keys()) or ["(empty - put PNGs in the 'dress' folder)"]); self.dress_cat.blockSignals(False); self.fill_dress_list()

    def fill_dress_menu(self):
        mn = self.dress_menu; mn.clear()
        for cat, files in self.dress_files().items():
            sub = mn.addMenu(cat)
            for f in files: sub.addAction(os.path.splitext(os.path.basename(f))[0]).triggered.connect(lambda _=False, p=f: self.load_dress(p))
        mn.addSeparator(); mn.addAction("Other PNG...").triggered.connect(lambda _=False: self.pick_dress())
        mn.addAction("Remove dress").triggered.connect(lambda _=False: self.load_dress(None))

    # ---------- crop editor ----------
    def ensure_box(self):
        if self.box is None:
            self.box = ideal_box(self.finfo, self.preset, self.alpha_np()) if self.finfo is not None else auto_box(self.rgb, self.face, self.ratio(), True)
            self.base_h = self.box[2]; self._reset_zoom_tilt()

    def _reset_zoom_tilt(self):
        for sl, v in ((self.s_zoom, 100), (self.s_tilt, 0)): sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)

    def open_editor(self):
        if not self.need_img(): return
        self.ensure_box(); self._pre = self._snap()
        self.crop_view.set_data(self.rgb.size, self.ratio(), self.box); self.refresh_preview(); self.stack.setCurrentIndex(1)
        self.status_msg("Drag corners to resize, inside = move, outside = rotate, scroll = zoom. Enter = Done, Esc = Cancel")

    def refresh_preview(self):
        if self.rgb is None: return
        small = self.source(); small.thumbnail((self.PREVIEW_MAX, self.PREVIEW_MAX), Image.Resampling.LANCZOS)
        self.crop_view.set_preview(render_preview(small, self.bg_rgb, 0))
        self.crop_view.set_data(self.rgb.size, self.ratio(), self.box) if self.box else None
        self.update_quality()

    def on_box_changed(self):
        v = int(round(self.base_h * 100 / self.box[2]))
        self.s_zoom.blockSignals(True); self.s_zoom.setValue(max(40, min(300, v))); self.s_zoom.blockSignals(False)
        self.s_tilt.blockSignals(True); self.s_tilt.setValue(max(-450, min(450, int(round(self.box[3] * 10))))); self.s_tilt.blockSignals(False)
        if self.stack.currentIndex() == 1: self._q_timer.start()

    def on_zoom(self, v):
        if self.box: self.box[2] = self.base_h * 100 / v; self.crop_view.update(); self._q_timer.start()

    def on_tilt(self, v):
        if self.box: self.box[3] = v / 10.0; self.crop_view.update(); self._q_timer.start()

    def rot90(self, d):
        if self.box: self.box[3] = (self.box[3] + d + 180) % 360 - 180; self.crop_view.update(); self.on_box_changed()

    def nudge(self, dx, dy):
        if self.box: self.box[0] += dx * self.box[2] * 0.01; self.box[1] += dy * self.box[2] * 0.01; self.crop_view.update(); self._q_timer.start()

    def on_enter(self, *_):
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit)): return
        i = self.stack.currentIndex()
        if i == 2: self.close_retouch()
        elif i == 3: self.close_face_picker()
        elif i == 1:
            pre = self._pre
            if pre is not None and pre["box"] != self.box: self.history.append(pre); self.history = self.history[-12:]; self.redo_stack = []
            self.apply_crop()

    def on_esc(self, *_):
        i = self.stack.currentIndex()
        if i == 2: return self.close_retouch(False)
        if i == 3: return self.close_face_picker(False)
        if i == 1:
            pre = self._pre
            if pre is not None and pre["box"] is not None and self.box is not None: self.box[:] = pre["box"]; self.base_h = pre["base_h"]; self.on_box_changed()
            self.stack.setCurrentIndex(0); self.show_current(); self.status_msg("Crop cancelled")

    def apply_crop(self):
        if self.rgb is None: return
        self.ensure_box()
        self.photo_base = crop_to(self.source(), self.bg_rgb, self.size_mm(), self.box)
        self.sheets = []; self.stack.setCurrentIndex(0); self._apply_adjust_now()

    # ---------- final adjust (debounced) ----------
    def adjust(self, im):
        b, c = self.s_bright.value(), self.s_contrast.value()
        if b: im = ImageEnhance.Brightness(im).enhance(1 + b / 100.0)
        if c: im = ImageEnhance.Contrast(im).enhance(1 + c / 100.0)
        if self.c_bw.isChecked(): im = ImageOps.grayscale(im).convert("RGB")
        if self.dress is not None: im = add_dress(im, self.dress, self.s_dress.value(), self.s_dress_y.value(), self.s_dress_x.value())
        txt = self.cap_edit.text().strip()
        if self.c_date.isChecked(): txt = (txt + "   " if txt else "") + datetime.date.today().strftime("%d-%m-%Y")
        return add_caption(im, txt)

    def apply_adjust(self, *_):
        if self.photo_base is None: return
        self._adjust_timer.start()

    def _apply_adjust_now(self):
        if self.photo_base is None: return
        self.photo = self.adjust(self.photo_base); self.sheets = []
        if self.stack.currentIndex() == 0:
            if self.view_mode == "sheet": self.make_sheet()
            self.show_current()
        self.update_quality(); self._mark_dirty()

    def toggle(self):
        if self.orig is None or self.stack.currentIndex() == 1: return
        self.set_view("photo" if self.view_mode == "orig" else "orig")


# =====================================================================
#  MAIN WINDOW - part 3: print studio, save, quality check
# =====================================================================
class OutputMixin:
    def custom_paper_mm(self): return (self.sp_pw.value(), self.sp_ph.value())

    def paper_px_now(self):
        return paper_px(self.paper_box.currentText(), self.custom_paper_mm(), DPI)

    def paper_short(self):
        n = self.paper_box.currentText()
        return "Custom" if n == CUSTOM_PAPER else n.split(" ")[0]

    def update_layout_info(self):
        try:
            self.layout_lbl.setText(layout_text(self.size_mm(), self.paper_box.currentText(), self.paper_px_now(), self.sp_gap.value(),
                                                self.sp_margin.value(), self.copies.value(), self.orient_box.currentText()))
        except Exception as e:
            self.layout_lbl.setText("")
        self.sheets = [] if self.view_mode != "sheet" else self.sheets
        self._mark_dirty()

    def on_print_option(self, *_):
        self.sp_pw.setVisible(self.paper_box.currentText() == CUSTOM_PAPER); self.sp_ph.setVisible(self.paper_box.currentText() == CUSTOM_PAPER)
        self.sheets = []; self.update_layout_info()
        if self.view_mode == "sheet" and self.photo is not None: self.make_sheet(); self.show_current()

    def make_sheet(self):
        """build every sheet needed for the chosen copies (best layout is automatic)"""
        if self.photo is None: return False
        paper = self.paper_px_now(); gap, mg = self.sp_gap.value(), self.sp_margin.value(); style = self.cut_box.currentText(); orient = self.orient_box.currentText()
        if self.c_mix.isChecked() and self.rgb is not None and self.box is not None:
            sz = SIZES[self.size2_box.currentText()]
            b2 = crop_to(self.source(), self.bg_rgb, sz, self.box)
            s, placed, total = pack_sheet([(self.photo, self.copies.value()), (self.adjust(b2), self.copies2.value())], paper, gap, mg, style=style, orient=orient)
            self.sheets = [s]
            if placed < total: QMessageBox.information(self, "Not enough space", "%d of %d photos fit. Reduce the copies or use bigger paper." % (placed, total))
        else:
            self.sheets = make_sheets(self.photo, self.copies.value(), paper, gap, mg, style, orient)
        self.sheet = self.sheets[0]; return True

    def do_layout(self):
        if self.photo is None: return QMessageBox.information(self, APP_NAME, "Make the photo first (✨ AI PASSPORT PHOTO or Crop).")
        self.make_sheet(); self.set_view("sheet")
        self.status_msg("%d sheet(s) ready - %s" % (len(self.sheets), self.layout_lbl.text()))

    # ---------- quality ----------
    def current_report(self):
        if self.photo_base is None or self.finfo is None and self.photo_base is None:
            return []
        box = self.box
        if box is None: return []
        return quality_report(self.photo_base, self.finfo, box, self.ratio(), self.preset, self.alpha is not None)

    def update_quality(self):
        """live 'Passport Position' chip under the preview"""
        if not hasattr(self, "chip"): return
        if self.stack.currentIndex() == 1 and self.rgb is not None and self.box is not None and self.finfo is not None:
            items = passport_check(self.finfo, self.box, self.ratio(), self.preset, self.alpha_np())
            worst = "warn" if any(r[0] != "ok" for r in items) else "ok"
            self.chip.setText("   ".join(r[2] for r in items[:3])); self.chip.setToolTip("\n".join(r[2] for r in items))
            self.chip.setStyleSheet("color:%s;font-weight:bold;font-size:14px;padding:4px;" % ("#b26a00" if worst == "warn" else "#1b7f2a"))
            self.fix_btn.setVisible(worst == "warn"); return
        if self.rgb is None or self.box is None or self.photo_base is None:
            self.chip.setText("Open a photo, then press ✨ AI PASSPORT PHOTO"); self.chip.setStyleSheet("color:#555;font-size:14px;padding:4px;")
            self.fix_btn.setVisible(False); self.chip.setToolTip(""); return
        rep = self.current_report()
        worst = "bad" if any(r[0] == "bad" for r in rep) else ("warn" if any(r[0] == "warn" for r in rep) else "ok")
        col = {"ok": "#1b7f2a", "warn": "#b26a00", "bad": "#c62828"}[worst]
        txt = "   ".join(r[2] for r in rep[:3]) + ("   (+%d more)" % (len(rep) - 3) if len(rep) > 3 else "")
        self.chip.setText(txt); self.chip.setStyleSheet("color:%s;font-weight:bold;font-size:14px;padding:4px;" % col)
        self.chip.setToolTip("\n".join(r[2] for r in rep))
        self.fix_btn.setVisible(any(r[3] for r in rep)); self._report = rep

    def confirm_quality(self):
        """before Save / Print. returns True to continue"""
        if self.photo_base is None: QMessageBox.information(self, APP_NAME, "Make the photo first."); return False
        rep = [r for r in self.current_report() if r[0] != "ok"]
        if not rep: return True
        m = QMessageBox(self); m.setIcon(QMessageBox.Warning); m.setWindowTitle("Check before printing")
        m.setText("This photo is not perfect yet:\n\n" + "\n".join(r[2] for r in rep))
        fx = m.addButton("⚡ AUTO FIX", QMessageBox.AcceptRole) if any(r[3] for r in rep) else None
        go = m.addButton("Continue anyway", QMessageBox.DestructiveRole); m.addButton("Cancel", QMessageBox.RejectRole); m.exec()
        if fx is not None and m.clickedButton() is fx: self.auto_fix(); return False
        return m.clickedButton() is go

    # ---------- save ----------
    def _guard_path(self, fn):
        if os.path.abspath(fn) in {os.path.abspath(p) for p in self.protected_paths()}:
            QMessageBox.warning(self, "Save", "This would overwrite the ORIGINAL photo. Choose another name."); return False
        return True

    def _need_sheet(self):
        if not self.sheets: self.make_sheet()
        return bool(self.sheets)

    def save_all(self):
        """one click: photo + sheet saved with automatic names (never overwrites). The job is saved too."""
        if not self.confirm_quality(): return
        if self.e_customer.text().strip() or self.job_id: self.save_job(quiet=True)
        d = self.out_dir(); name = self.customer_name(); kind = "Passport" if "Passport" in self.e_service.currentText() or self.preset_name.endswith("Passport") else safe_name(self.e_service.currentText(), "Photo")
        size_txt = "%gx%g" % self.size_mm(); saved = []
        p1 = unique_path(os.path.join(d, out_name(name, kind, size_txt))); save_image_file(self.photo, p1); saved.append(p1)
        if self._need_sheet():
            for i, s in enumerate(self.sheets):
                nm = out_name(name, kind, ext="jpg", sheet=True, paper=self.paper_short() + ("_%d" % (i + 1) if len(self.sheets) > 1 else ""))
                p = unique_path(os.path.join(d, nm)); save_image_file(s, p); saved.append(p)
        extra = SETTINGS.get("extra_save_dir")
        if extra and os.path.isdir(extra):
            for p in saved: shutil.copy2(p, unique_path(os.path.join(extra, os.path.basename(p))))
        self.note_saved(saved[0]); self.banner.done("Saved %d file(s) in %s" % (len(saved), d))
        if hasattr(self, "open_folder_btn"): self.open_folder_btn.setVisible(True)

    def open_output_folder(self):
        d = getattr(self, "_last_saved_dir", None) or self.out_dir()
        if hasattr(os, "startfile"): os.startfile(d)
        else: QMessageBox.information(self, "Folder", d)

    def _dialog_save(self, title, default, filt):
        fn, _ = QFileDialog.getSaveFileName(self, title, unique_path(os.path.join(self.out_dir(), default)), filt)
        if not fn or not self._guard_path(fn): return None
        return fn

    def save_single(self):
        if not self.confirm_quality(): return
        fn = self._dialog_save("Save Photo", out_name(self.customer_name(), "Passport", "%gx%g" % self.size_mm()), "JPG (*.jpg);;PNG (*.png);;PDF (*.pdf)")
        if fn: save_image_file(self.photo, fn); self.note_saved(fn, "photo")

    def save_sheet(self):
        if not self._need_sheet(): return QMessageBox.information(self, APP_NAME, "Make the photo first.")
        fn = self._dialog_save("Save Sheet", out_name(self.customer_name(), "Passport", sheet=True, paper=self.paper_short()), "JPG (*.jpg);;PNG (*.png)")
        if not fn: return
        base, ext = os.path.splitext(fn)
        for i, s in enumerate(self.sheets): save_image_file(s, fn if i == 0 else "%s_%d%s" % (base, i + 1, ext))
        self.note_saved(fn, "sheet")

    def save_pdf(self):
        if not self._need_sheet(): return QMessageBox.information(self, APP_NAME, "Make the photo first.")
        fn = self._dialog_save("Save PDF", out_name(self.customer_name(), "Passport", ext="pdf", sheet=True, paper=self.paper_short()), "PDF (*.pdf)")
        if fn: save_pdf_pages(self.sheets, fn, DPI); self.note_saved(fn, "pdf")

    def save_online_cb(self):
        if self.photo is None: return QMessageBox.information(self, APP_NAME, "Make the photo first.")
        fn = self._dialog_save("Save for Online Form", out_name(self.customer_name(), "Online", "%gx%g" % self.size_mm()), "JPG (*.jpg)")
        if not fn: return
        sz = (self.sp_ow.value(), self.sp_oh.value())
        r = save_online(self.photo, fn, self.sp_kb.value(), sz if sz[0] and sz[1] else None)
        if r is None: QMessageBox.warning(self, "Too big", "Cannot reach this size. Allow more KB or use smaller pixels.")
        else: self.note_saved(fn, "online"); self.status_msg("Saved: %.0f KB, %dx%d px" % (r[0] / 1024.0, r[1][0], r[1][1]))

    # ---------- print ----------
    def print_now(self):
        if not self.confirm_quality() or not self._need_sheet(): return
        if print_sheets(self, self.sheets, DPI): self.banner.done("Sent to printer")

    def print_preview(self):
        if self.photo is None: return QMessageBox.information(self, APP_NAME, "Make the photo first.")
        if self._need_sheet(): print_sheets(self, self.sheets, DPI, preview=True)

    def quick_print(self):
        """send straight to the Windows default printer (v2 behaviour)"""
        if not self.confirm_quality() or not self._need_sheet(): return
        p = unique_path(os.path.join(os.environ.get("TEMP", "."), "ripon_print.jpg")); save_image_file(self.sheets[0], p)
        if hasattr(os, "startfile"): os.startfile(p, "print")
        else: QMessageBox.information(self, "Print", "Saved: " + p)

# =====================================================================
#  MAIN WINDOW - part 4: AI passport (one click), auto fix, model jobs
# =====================================================================
class AIMixin:
    def ai_passport(self, preset_name=None, basic=False):
        """✨ AI PASSPORT PHOTO: the whole pipeline in one click, in the background"""
        if not self.need_img(): return
        if self.runner.busy: return self.status_msg("Please wait - a job is running.")
        if self.stack.currentIndex() in (1, 2, 3): self.on_esc()
        if preset_name and preset_name in self.presets and preset_name != self.preset_name: self.apply_preset(preset_name, reframe=False)
        preset = dict(self.preset); preset["dpi"] = DPI
        bg_on = self.c_ai_bg.isChecked() and not basic
        opts = dict(model=MODELS[self.model_box.currentText()], bg=bg_on, enhance=self.c_ai_enh.isChecked(), retouch=self.c_ai_skin.isChecked(),
                    restore=("auto" if self.c_ai_restore.isChecked() and not basic else False), upscale=("auto" if self.c_ai_up.isChecked() else False),
                    bg_rgb=self.bg_rgb, edge=self.s_edge.value(), face=self.fi_orig)
        img = self.orig
        self.banner.start("Starting...", True)
        def fn(ctx): return run_ai_pipeline(img, preset, opts, ctx, HUB)
        def ok(res): self._apply_pipeline_result(res)
        def fail(msg, exc): self._ai_failed(msg, exc)
        if not self.runner.submit(fn, ok, fail): self.banner.idle()

    def _apply_pipeline_result(self, res):
        self.push()
        self.rgb, self.alpha, self.finfo, self.face = res["rgb"], res["alpha"], res["fi"], res["face"]
        if self.fi_orig is None: pass
        self.hair_mask = None; self.box = res["box"]; self.base_h = self.box[2]; self._reset_zoom_tilt()
        self.photo_base = res["photo_base"]; self._invalidate_cache(); self._update_face_label()
        self.banner.progress_update("Preparing print sheet...", 7, 7)
        self._apply_adjust_now(); self.make_sheet(); self.set_view("photo")
        self._mark_dirty(); self.update_quality()
        notes = list(res["notes"]) + [("warn", n) for n in HUB.notes]; HUB.notes.clear()
        self.notes_lbl.setText("<br>".join(("⚠ " if lv == "warn" else "ℹ ") + t for lv, t in notes))
        warns = [t for lv, t in notes if lv == "warn"]
        tsecs = time.time() - min(res.get("t0", time.time()), time.time())
        rep = getattr(self, "_report", [])
        if warns: self.banner.warn("%s   (%d note%s below the photo)" % (warns[0], len(warns), "" if len(warns) == 1 else "s"))
        else: self.banner.done("Passport photo ready - " + (rep[0][2] if rep else "check the preview"))
        self.status_msg("AI Passport finished. Check the preview, then Print or Save.")
        if any("Background could not be removed" in t for t in warns):
            QMessageBox.information(self, "Background", "The background could not be removed automatically (AI model missing or failed).\nThe photo was still framed and cleaned. You can open 'AI Models' to install it, or continue.")

    def _ai_failed(self, msg, exc):
        if isinstance(exc, NoFace):
            self.banner.warn("Face not found - draw a box on the face, then it continues")
            return self.open_face_picker("Face not found automatically. Drag a box over the face, then press Enter - AI Passport will continue.",
                                         retry=lambda: self.ai_passport())
        if msg == "Cancelled": return self.banner.idle("Cancelled")
        self.banner.error("AI Passport failed")
        if ask_yes_no(self, "AI Passport", "The AI step failed:\n\n%s\n\nContinue in BASIC mode (no AI background / face restore)?" % msg[:500]):
            QTimer.singleShot(0, lambda: self.ai_passport(basic=True))

    def ai_visa(self): self.ai_passport("Visa Photo")
    def ai_id(self): self.ai_passport("NID Photo")

    # ---------- one-click fixes for the quality chip ----------
    def auto_fix(self):
        if self.photo_base is None: return self.status_msg("Make the photo first.")
        fixes = []
        for lv, code, txt, fix in self.current_report():
            if fix and fix not in fixes: fixes.append(fix)
        if not fixes: return self.banner.done("Nothing to fix - ready")
        order = ["framing", "bg", "upscale", "exposure", "restore"]
        self._fix_queue = sorted(fixes, key=order.index); self._next_fix()

    def _next_fix(self):
        if not self._fix_queue: self.update_quality(); self.banner.done("Auto fix finished"); return
        if self.runner.busy: return
        f = self._fix_queue.pop(0)
        if f == "framing":
            if self.finfo is not None:
                self.box = ideal_box(self.finfo, self.preset, self.alpha_np()); self.base_h = self.box[2]; self._reset_zoom_tilt(); self.apply_crop()
                self.status_msg("Face position corrected")
        elif f == "bg": self.m_bg_remove()
        elif f == "upscale": self.m_upscale()
        elif f == "exposure":
            arr, fi = np.array(self.rgb), self.finfo
            self._bg_op("Fixing brightness...", lambda ctx: exposure_fix(arr, fi), lambda r: self._done_rgb(r, "Brightness corrected"))
        elif f == "restore":
            if HUB.is_installed("gfpgan") and HUB.lib_available("gfpgan"): self.m_face_ai()
            else: self.m_auto_enhance()
        if not self.runner.busy and self._fix_queue: QTimer.singleShot(0, self._next_fix)
        elif not self.runner.busy: self.update_quality(); self.banner.done("Auto fix finished")

    # ---------- legacy "AUTO CHALAO" (v2 checklist) ----------
    def run_auto(self):
        if not self.need_img() or self.runner.busy: return
        if self.face is None: return self.detect_face_async(silent=False, after=self.run_auto)
        settings = {"enh": self.c_enh.isChecked(), "enh_strength": self.s_enh.value(), "face_ai": self.c_face.isChecked(), "face_strength": self.s_aiface.value(),
                    "pimple": self.c_pimple.isChecked(), "pimple_strength": self.s_pimple.value(), "smooth": self.c_smooth.isChecked(),
                    "smooth_strength": self.s_smooth.value(), "tone": self.c_tone.isChecked(), "tone_strength": self.s_tone.value(),
                    "shine": self.c_shine.isChecked(), "bg": self.c_bg.isChecked(), "hair": self.c_hair.isChecked(), "hair_strength": self.s_hair.value()}
        arr = np.array(self.orig if self.orig is not None else self.rgb); face = self.face; model = MODELS[self.model_box.currentText()]; hrgb = self.hair_rgb
        def apply(res):
            a, alpha, hm, notes = res
            self.set_rgb(a); self.alpha = alpha; self.hair_mask = hm
            self.box = ideal_box(self.finfo, self.preset, self.alpha_np()) if (self.finfo is not None and self.c_crop.isChecked()) else auto_box(self.rgb, self.face, self.ratio(), self.c_crop.isChecked())
            self.base_h = self.box[2]; self._reset_zoom_tilt()
            if self.c_manual.isChecked() or (self.face is None and self.c_crop.isChecked()): self.open_editor()
            else: self.apply_crop()
            self.banner.done("Auto finished" + ("  [" + " | ".join(notes) + "]" if notes else ""))
        self._bg_op("Auto processing...", lambda ctx: run_auto_pipeline(arr, face, settings, model, hrgb), apply)

    # ---------- AI model jobs ----------
    def _load_key(self, key):
        r = MODEL_REGISTRY[key]
        if r["kind"] == "insightface": HUB._insight()
        elif r["kind"] == "rembg": HUB._rembg_session(key)
        elif key == "gfpgan": HUB._gfpgan()
        elif key == "realesrgan": HUB._esrgan()

    def download_model(self, key, cb=None):
        if self.runner.busy: return QMessageBox.information(self, "AI Models", "Please wait - a job is running.")
        def fn(ctx):
            HUB.download(key, lambda m, f: ctx.frac(m, f) if f is not None else ctx.msg(m))
            ctx.msg("Loading model..."); self._load_key(key); return key
        def ok(_):
            self.banner.done("Model ready: " + MODEL_REGISTRY[key]["label"])
            if cb: cb()
        def fail(m, e):
            self.banner.error("Download failed"); HUB.errors[key] = m[:200]
            QMessageBox.warning(self, "AI Models", "Download / load failed:\n%s\n\nCheck the internet connection and the packages listed in the install notes." % m[:600])
            if cb: cb()
        self.banner.start("Downloading model...", True, "DOWNLOADING"); self.runner.submit(fn, ok, fail)

    def warm_model(self, key, cb=None):
        if self.runner.busy: return
        def fn(ctx): ctx.msg("Loading model..."); self._load_key(key); return key
        self.banner.start("Loading model...", False, "LOADING")
        self.runner.submit(fn, lambda _: (self.banner.done("Model loaded"), cb() if cb else None),
                           lambda m, e: (self.banner.error("Could not load: " + m[:80]), cb() if cb else None))

    def startup_warmup(self):
        if not SETTINGS.get("warmup", True) or self.runner.busy: return
        model = MODELS[self.model_box.currentText()]
        def fn(ctx): HUB.warmup(model, lambda m, f: ctx.msg(m)); return True
        self.banner.start("Starting AI (only installed models)...", False, "STARTING")
        self.runner.submit(fn, lambda _: self.banner.idle("Ready - open a photo"), lambda m, e: self.banner.idle("Ready"))


# =====================================================================
#  MAIN WINDOW: layout (dashboard) + entry point
# =====================================================================
APP_STYLE = """
QWidget{font-size:13px;}
QPushButton{padding:5px 10px;border:1px solid #b9c3cc;border-radius:6px;background:#f4f7fa;}
QPushButton:hover{background:#e6eef6;} QPushButton:disabled{color:#999;}
QPushButton:checked{background:#1565c0;color:white;border-color:#0d47a1;}
QGroupBox{font-weight:bold;border:1px solid #cfd8e0;border-radius:8px;margin-top:10px;padding-top:8px;}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;color:#37474f;}
QTabBar::tab{padding:8px 12px;font-weight:bold;} QTabBar::tab:selected{color:#1565c0;border-bottom:3px solid #1565c0;}
QFrame#Bar{background:#f0f4f8;border:1px solid #d5dde5;border-radius:8px;}
"""
BTN_GREEN = "QPushButton{background:#2e7d32;color:white;font-weight:bold;border:1px solid #1b5e20;} QPushButton:hover{background:#388e3c;}"
BTN_BLUE = "QPushButton{background:#1565c0;color:white;font-weight:bold;border:1px solid #0d47a1;} QPushButton:hover{background:#1e73d0;}"
BTN_ORANGE = "QPushButton{background:#ef6c00;color:white;font-weight:bold;border:1px solid #bf5600;} QPushButton:hover{background:#f57c00;}"
BTN_PURPLE = "QPushButton{background:#6a1b9a;color:white;font-weight:bold;border:1px solid #4a148c;}"

def _scroll(widget_layout):
    w = QWidget(); w.setLayout(widget_layout); s = QScrollArea(); s.setWidgetResizable(True); s.setWidget(w); s.setFrameShape(QFrame.NoFrame); return s

def _lbl(text, wrap=True, style=""):
    l = QLabel(text); l.setWordWrap(wrap)
    if style: l.setStyleSheet(style)
    return l

class AppV3(CoreMixin, EditMixin, OutputMixin, AIMixin, QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE); self.resize(1420, 920)
        # ---- state ----
        self.orig = self.rgb = self.alpha = None; self._face = None; self.finfo = self.fi_orig = None
        self.history = []; self.redo_stack = []; self.hair_mask = None; self.hair_rgb = tuple(HAIR_COLORS["Black (Kalo)"])
        self.dress = None; self.dress_path = None; self._pre = None; self.rt_work = None; self.rt_stack = []
        self.box = None; self.base_h = 1.0; self.bg_rgb = (255, 255, 255)
        self.photo_base = self.photo = self.sheet = None; self.sheets = []; self.view_mode = "photo"
        self._rev = 0; self._cache = {}; self._last_display_key = None; self._last_display_pixmap = None
        self._fix_queue = []; self._dirty = False; self._saved_rev = 0; self._pending_face_action = None; self._report = []
        self.job_id = None; self.job_folder = None; self.src_path = self.inbox_path = None; self._last_saved_dir = None
        self.doc_studio = None; self._dress_cache = {}
        self.db = ShopDB(); self.presets = load_presets()
        self.preset_name = next(iter(self.presets)); self.preset = dict(self.presets[self.preset_name])
        self.runner = TaskRunner(self); self.runner.progress.connect(self._on_progress)
        # ---- timers ----
        self._adjust_timer = QTimer(self); self._adjust_timer.setSingleShot(True); self._adjust_timer.setInterval(70); self._adjust_timer.timeout.connect(self._apply_adjust_now)
        self._resize_timer = QTimer(self); self._resize_timer.setSingleShot(True); self._resize_timer.timeout.connect(self.show_current)
        self._q_timer = QTimer(self); self._q_timer.setSingleShot(True); self._q_timer.setInterval(120); self._q_timer.timeout.connect(self.update_quality)
        self._auto_timer = QTimer(self); self._auto_timer.setInterval(max(10, int(SETTINGS.get("autosave_sec", 30))) * 1000); self._auto_timer.timeout.connect(self._autosave); self._auto_timer.start()
        self.setStyleSheet(APP_STYLE)
        self.build_widgets(); self.build_layout()
        self.fill_preset_box(); self.apply_preset(self.preset_name, reframe=False); self.refresh_dress_tab(); self._update_face_label(); self.update_quality()
        for key in (Qt.Key_Return, Qt.Key_Enter): QShortcut(QKeySequence(key), self).activated.connect(self.on_enter)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.on_esc)

    def _on_progress(self, text, i, n):
        if n < 0: self.banner.set_fraction(i / float(-n), text)
        else: self.banner.progress_update(text, i, n)

    # ================= widgets =================
    def build_widgets(self):
        self.banner = ProcessingBanner(); self.banner.cancel_clicked.connect(self.runner.cancel)
        self.preview = QLabel("Open a photo to begin"); self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(560, 600)
        self.preview.setStyleSheet("border:1px solid #888; background:#e9eef2; color:#667;font-size:18px;"); self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.crop_view = CropView(); self.crop_view.changed = self.on_box_changed; self.crop_view.on_confirm = self.on_enter
        self.paint_view = PaintView(); self.paint_view.on_action = self.on_paint_action; self.face_view = FacePickView()
        self.stack = QStackedWidget()
        for w in (self.preview, self.crop_view, self.paint_view, self.face_view): self.stack.addWidget(w)
        self.status = QLabel(""); self.status.setWordWrap(True); self.status.setStyleSheet("color:#455a64;")
        self.chip = QLabel(""); self.chip.setWordWrap(True)
        self.fix_btn = btn("⚡ AUTO FIX", self.fix_clicked, BTN_ORANGE, "Fix face position, background, brightness automatically", 34); self.fix_btn.setVisible(False)
        self.notes_lbl = QLabel(""); self.notes_lbl.setWordWrap(True); self.notes_lbl.setTextFormat(Qt.RichText); self.notes_lbl.setStyleSheet("color:#8a5a00;")
        self.view_btns = {}
        for k, t in (("photo", "Photo"), ("sheet", "Print Sheet"), ("orig", "Original")):
            b = QPushButton(t); b.setCheckable(True); b.setMinimumHeight(30); b.clicked.connect(lambda _=False, m=k: self.set_view(m)); self.view_btns[k] = b
        self.view_btns["photo"].setChecked(True)
        # header
        self.preset_box = QComboBox(); self.preset_box.setMinimumWidth(190); self.preset_box.activated.connect(self.on_preset_pick)
        self.size_lbl = QLabel(""); self.size_lbl.setStyleSheet("color:#555;")
        self.copies = QSpinBox(); self.copies.setRange(1, 200); self.copies.setValue(6); self.copies.setMinimumHeight(30); self.copies.valueChanged.connect(self.update_layout_info)
        self.swatch = QLabel(""); self.swatch.setFixedSize(34, 28)
        self.e_customer = QLineEdit(); self.e_customer.setPlaceholderText("Customer name"); self.e_phone = QLineEdit(); self.e_phone.setPlaceholderText("Phone")
        self.e_service = QComboBox(); self.e_service.addItems(SERVICES); self.e_service.setEditable(True)
        self.e_price = QDoubleSpinBox(); self.e_price.setRange(0, 1000000); self.e_price.setDecimals(0); self.e_price.setPrefix("৳ "); self.e_price.setMinimumWidth(95)
        self.job_lbl = QLabel("New job"); self.job_lbl.setStyleSheet("color:#1565c0;font-weight:bold;")
        self.model_box = QComboBox(); self.model_box.addItems(MODELS.keys())
        for i, (k, v) in enumerate(MODELS.items()):
            if v == SETTINGS.get("model"): self.model_box.setCurrentIndex(i)
        self.model_box.currentIndexChanged.connect(lambda *_: (SETTINGS.__setitem__("model", MODELS[self.model_box.currentText()]), save_settings()))
        self.face_lbl = QLabel("")
        # AI options
        def chk(t, on=True, tip=None):
            c = QCheckBox(t); c.setChecked(on)
            if tip: c.setToolTip(tip)
            return c
        self.c_ai_bg = chk("Remove background"); self.c_ai_enh = chk("Clean and enhance photo"); self.c_ai_skin = chk("Natural skin retouch + remove spots")
        self.c_ai_restore = chk("AI Face Restore - only when the face is blurry"); self.c_ai_up = chk("Upscale - only when the photo is small")
        self.c_auto_open = chk("Run AI Passport automatically after opening a photo", bool(SETTINGS.get("auto_ai", False)))
        self.c_auto_open.toggled.connect(lambda v: (SETTINGS.__setitem__("auto_ai", bool(v)), save_settings()))
        self.s_edge = hslider(0, 100, 60)
        # legacy auto checklist (v2)
        self.c_crop = chk("Auto Crop (follow face)"); self.c_manual = chk("Adjust crop by hand after auto", False); self.c_bg = chk("Auto Background Remove")
        self.c_enh = chk("Auto Enhance"); self.c_smooth = chk("Auto Skin Retouch (natural)"); self.c_tone = chk("Auto Skin Tone Up", False)
        self.c_pimple = chk("Auto Pimple/Spot Remove"); self.c_shine = chk("Reduce oily shine", False); self.c_face = chk("AI Face Restore (GFPGAN)", False)
        self.c_hair = chk("Auto Hair Color (select color in Photo Edit)", False)
        # sliders
        self.s_smooth = hslider(0, 100, 50); self.s_tone = hslider(0, 100, 50); self.s_enh = hslider(0, 100, 60); self.s_pimple = hslider(0, 100, 55)
        self.s_aiface = hslider(10, 100, 50); self.s_hair = hslider(20, 100, 100)
        self.s_zoom = hslider(40, 300, 100); self.s_zoom.valueChanged.connect(self.on_zoom)
        self.s_tilt = hslider(-450, 450, 0); self.s_tilt.valueChanged.connect(self.on_tilt)
        self.s_bright = hslider(-50, 50, 0); self.s_bright.valueChanged.connect(self.apply_adjust)
        self.s_contrast = hslider(-50, 50, 0); self.s_contrast.valueChanged.connect(self.apply_adjust)
        self.s_brush = hslider(2, 120, 12); self.s_brush.valueChanged.connect(self.on_brush)
        self.s_dress = hslider(30, 220, 100); self.s_dress.valueChanged.connect(self.apply_adjust)
        self.s_dress_y = hslider(-40, 60, 0); self.s_dress_y.valueChanged.connect(self.apply_adjust)
        self.s_dress_x = hslider(-40, 40, 0); self.s_dress_x.valueChanged.connect(self.apply_adjust)
        self.hair_box = QComboBox(); self.hair_box.addItems(list(HAIR_COLORS.keys())); self.hair_box.currentTextChanged.connect(self.on_hair_pick)
        self.rt_tool_box = QComboBox(); self.rt_tool_box.addItems(["Pimple Heal (click)", "Hair area + (brush)", "Hair area - (eraser)"]); self.rt_tool_box.currentIndexChanged.connect(self.on_rt_tool)
        self.c_bw = QCheckBox("Black & White photo"); self.c_bw.stateChanged.connect(self.apply_adjust)
        self.cap_edit = QLineEdit(); self.cap_edit.setPlaceholderText("Name under the photo (optional, English)"); self.cap_edit.textChanged.connect(self.apply_adjust)
        self.c_date = QCheckBox("Add date"); self.c_date.stateChanged.connect(self.apply_adjust)
        # print
        self.paper_box = QComboBox(); self.paper_box.addItems(list(PAPERS_MM.keys()) + [CUSTOM_PAPER]); self.paper_box.setMinimumHeight(30); self.paper_box.currentIndexChanged.connect(self.on_print_option)
        self.sp_pw = QDoubleSpinBox(); self.sp_pw.setRange(30, 1000); self.sp_pw.setValue(210); self.sp_pw.setSuffix(" mm wide"); self.sp_pw.valueChanged.connect(self.on_print_option)
        self.sp_ph = QDoubleSpinBox(); self.sp_ph.setRange(30, 1500); self.sp_ph.setValue(297); self.sp_ph.setSuffix(" mm high"); self.sp_ph.valueChanged.connect(self.on_print_option)
        self.sp_pw.setVisible(False); self.sp_ph.setVisible(False)
        self.orient_box = QComboBox(); self.orient_box.addItems(["Auto", "Portrait", "Landscape"]); self.orient_box.currentIndexChanged.connect(self.on_print_option)
        self.cut_box = QComboBox(); self.cut_box.addItems(["Light border", "Cut marks", "None"]); self.cut_box.currentIndexChanged.connect(self.on_print_option)
        self.sp_gap = QDoubleSpinBox(); self.sp_gap.setRange(0, 30); self.sp_gap.setValue(3); self.sp_gap.setSuffix(" mm gap"); self.sp_gap.valueChanged.connect(self.on_print_option)
        self.sp_margin = QDoubleSpinBox(); self.sp_margin.setRange(0, 30); self.sp_margin.setValue(4); self.sp_margin.setSuffix(" mm margin"); self.sp_margin.valueChanged.connect(self.on_print_option)
        self.c_mix = QCheckBox("Mixed sheet: add a second size"); self.c_mix.stateChanged.connect(self.on_print_option)
        self.size2_box = QComboBox(); self.size2_box.addItems(list(SIZES.keys())); self.size2_box.setCurrentIndex(2)
        self.copies2 = QSpinBox(); self.copies2.setRange(1, 60); self.copies2.setValue(8); self.copies2.setSuffix(" copies (2nd size)")
        self.layout_lbl = _lbl("", True, "color:#1565c0;font-weight:bold;")
        self.sp_kb = QSpinBox(); self.sp_kb.setRange(10, 2000); self.sp_kb.setValue(100); self.sp_kb.setSuffix(" KB max")
        self.sp_ow = QSpinBox(); self.sp_ow.setRange(0, 4000); self.sp_ow.setSuffix(" px width (0 = same)")
        self.sp_oh = QSpinBox(); self.sp_oh.setRange(0, 4000); self.sp_oh.setSuffix(" px height (0 = same)")
        self.dress_cat = QComboBox(); self.dress_cat.currentIndexChanged.connect(self.fill_dress_list)
        self.dress_list = QListWidget(); self.dress_list.setViewMode(QListWidget.IconMode); self.dress_list.setIconSize(QSize(84, 84)); self.dress_list.setResizeMode(QListWidget.Adjust)
        self.dress_list.setMovement(QListWidget.Static); self.dress_list.setMinimumHeight(190); self.dress_list.itemDoubleClicked.connect(lambda it: self.load_dress(it.data(Qt.UserRole)))
        self.open_folder_btn = btn("📁 Open folder", self.open_output_folder, "", "Show saved files", 40); self.open_folder_btn.setVisible(False)

    def fix_clicked(self):
        if self.stack.currentIndex() == 1 and self.finfo is not None and self.box is not None:
            self.box[:] = ideal_box(self.finfo, self.preset, self.alpha_np()); self.base_h = self.box[2]; self.on_box_changed(); self.crop_view.update(); self._q_timer.start()
        else: self.auto_fix()

    # ================= layout =================
    def build_layout(self):
        # --- row A: the main workflow ---
        self.btn_ai = btn("✨  AI PASSPORT PHOTO", self.ai_passport, BTN_GREEN + "QPushButton{font-size:19px;}", "One click: face, background, hair edges, enhance, retouch, crop.   এক ক্লিকে পাসপোর্ট ছবি", 56)
        self.btn_ai.setMinimumWidth(330)
        rowA = QHBoxLayout(); rowA.setSpacing(8)
        rowA.addWidget(btn("📂  1  Open Photo", self.open_photo, BTN_BLUE, "ছবি খুলুন  (Ctrl+O)", 56)); rowA.addWidget(self.btn_ai, 2)
        rowA.addWidget(btn("AI Visa Photo", self.ai_visa, "", "Visa 40x50 mm", 56)); rowA.addWidget(btn("AI ID Photo", self.ai_id, "", "NID / ID photo", 56))
        rowA.addWidget(btn("Auto Enhance", self.m_auto_enhance, "", "Light, brightness and colour clean-up", 56))
        # --- row B: size / background / copies / output ---
        bgrow = QHBoxLayout(); bgrow.setSpacing(4)
        for n, c in BG_COLORS.items(): bgrow.addWidget(btn(n, lambda c=c: self.set_bg(c), "", "", 30))
        bgrow.addWidget(btn("Color...", self.pick_bg, "", "Any colour", 30)); bgrow.addWidget(self.swatch)
        rowB = QHBoxLayout(); rowB.setSpacing(6)
        rowB.addWidget(QLabel("<b>3 Size</b>")); rowB.addWidget(self.preset_box)
        rowB.addWidget(btn("✎", self.preset_edit, "", "Edit this preset", 30)); rowB.addWidget(btn("🗑", self.preset_delete, "", "Delete / reset this preset", 30)); rowB.addWidget(self.size_lbl)
        rowB.addSpacing(10); rowB.addWidget(QLabel("<b>4 Background</b>")); rowB.addLayout(bgrow)
        rowB.addSpacing(10); rowB.addWidget(QLabel("<b>5 Copies</b>")); rowB.addWidget(self.copies); rowB.addWidget(self.paper_box)
        rowB.addStretch(1)
        rowB.addWidget(btn("🖨  6  Print", self.print_now, BTN_BLUE, "Print (Ctrl+P)   প্রিন্ট", 38)); rowB.addWidget(btn("💾  Save", self.save_all, BTN_BLUE, "Save photo + sheet with automatic names   সেভ", 38)); rowB.addWidget(self.open_folder_btn)
        # --- row C: customer ---
        rowC = QHBoxLayout(); rowC.setSpacing(6)
        for w in (QLabel("Customer"), self.e_customer, self.e_phone, self.e_service, self.e_price): rowC.addWidget(w)
        rowC.addWidget(btn("💾 Save Job", self.save_job, "", "Record this customer and keep the files", 30)); rowC.addWidget(btn("📋 Jobs", self.open_jobs, "", "Customer jobs (Ctrl+J)", 30))
        rowC.addWidget(btn("৳ Accounts", self.open_accounts, "", "Today's income and expenses", 30)); rowC.addWidget(self.job_lbl); rowC.addStretch(1)
        rowC.addWidget(btn("AI Models", self.open_models, "", "Installed / missing / GPU", 30))
        bar = QFrame(); bar.setObjectName("Bar"); bl = QVBoxLayout(bar); bl.setContentsMargins(8, 8, 8, 8); bl.setSpacing(6); bl.addLayout(rowA); bl.addLayout(rowB); bl.addLayout(rowC)
        # --- left: preview ---
        vb = QHBoxLayout()
        for b in self.view_btns.values(): vb.addWidget(b)
        vb.addStretch(1); vb.addWidget(btn("Undo", self.undo, "", "Ctrl+Z", 30)); vb.addWidget(btn("Redo", self.redo, "", "Ctrl+Y", 30))
        chip_row = QHBoxLayout(); chip_row.addWidget(self.chip, 1); chip_row.addWidget(self.fix_btn)
        left = QVBoxLayout(); left.addWidget(self.stack, 1); left.addLayout(vb); left.addLayout(chip_row); left.addWidget(self.notes_lbl); left.addWidget(self.status)
        # --- right: tabs ---
        tabs = QTabWidget(); tabs.setFixedWidth(430)
        tabs.addTab(self._tab_ai(), "AI PHOTO"); tabs.addTab(self._tab_edit(), "PHOTO EDIT"); tabs.addTab(self._tab_bg(), "BACKGROUND")
        tabs.addTab(self._tab_dress(), "DRESS"); tabs.addTab(self._tab_doc(), "DOCUMENT"); tabs.addTab(self._tab_print(), "PRINT")
        self.tabs = tabs
        body = QHBoxLayout(); body.addLayout(left, 1); body.addWidget(tabs)
        outer = QVBoxLayout(self); outer.setMenuBar(self.build_menu()); outer.setContentsMargins(8, 4, 8, 8); outer.setSpacing(6)
        outer.addWidget(self.banner); outer.addWidget(bar); outer.addLayout(body, 1)

    def _tab_ai(self):
        l = QVBoxLayout()
        l.addWidget(group("ONE CLICK", [btn("✨ AI PASSPORT PHOTO  (F2)", self.ai_passport, BTN_GREEN + "QPushButton{font-size:15px;}", "", 46),
                                         row(btn("AI Visa Photo", self.ai_visa), btn("AI ID Photo", self.ai_id)), btn("Auto Enhance", self.m_auto_enhance),
                                         _lbl("The AI always starts from the ORIGINAL photo, so you can run it again with another size or background.")]))
        l.addWidget(group("AI OPTIONS  (normally leave all ON)", [self.c_ai_bg, self.c_ai_enh, self.c_ai_skin, self.c_ai_restore, self.c_ai_up, self.c_auto_open]))
        l.addWidget(group("FACE", [self.face_lbl, row(btn("Face Auto Detect (F7)", self.m_face_detect), btn("Face Manual (F8)", self.m_face_manual, BTN_BLUE)),
                                   _lbl("Eyes, nose, mouth and chin are found automatically. If the face is not found, draw a box with 'Face Manual'.")]))
        l.addWidget(group("OLD AUTO LIST  (v2 - choose your own steps)", [self.c_crop, self.c_manual, self.c_bg, self.c_enh, self.c_pimple, self.c_smooth, self.c_tone, self.c_shine, self.c_face, self.c_hair,
                                                                       row(btn("Select All", lambda: self.set_all(True)), btn("Clear", lambda: self.set_all(False))),
                                                                       btn("AUTO CHALAO  (F5)", self.run_auto, BTN_GREEN)]))
        l.addStretch(1); return _scroll(l)

    def set_all(self, v):
        for c in (self.c_crop, self.c_bg, self.c_enh, self.c_pimple, self.c_smooth, self.c_tone, self.c_shine): c.setChecked(v)
        if not v: self.c_face.setChecked(False); self.c_hair.setChecked(False)

    def _tab_edit(self):
        l = QVBoxLayout()
        l.addWidget(group("CROP  /  ROTATE", [btn("Open Crop (Photoshop style)", self.open_editor, BTN_BLUE, "Drag corners; yellow line = eye line", 38),
                                              _lbl("Corners = resize, inside = move, outside = rotate, scroll = zoom, Enter = done.", True, "color:#666;"),
                                              QLabel("Zoom"), self.s_zoom, QLabel("Rotate (degree)"), self.s_tilt,
                                              row(btn("⟲ 90", lambda: self.rot90(-90)), btn("⟳ 90", lambda: self.rot90(90))),
                                              row(btn("←", lambda: self.nudge(-1, 0)), btn("→", lambda: self.nudge(1, 0)), btn("↑", lambda: self.nudge(0, -1)), btn("↓", lambda: self.nudge(0, 1))),
                                              btn("Crop Apply (Enter)", self.apply_crop, BTN_ORANGE)]))
        l.addWidget(group("BRIGHTNESS / CONTRAST / COLOR  (final photo)", [QLabel("Brightness"), self.s_bright, QLabel("Contrast"), self.s_contrast, self.c_bw,
                                                                         QLabel("Enhance strength"), self.s_enh, btn("Photo Enhance PRO", self.m_enhance)]))
        l.addWidget(group("SKIN RETOUCH", [QLabel("Retouch / smooth strength"), self.s_smooth, row(btn("Skin Retouch (natural)", self.m_retouch), btn("Skin Smooth (soft)", self.m_smooth)),
                                           QLabel("Skin tone strength"), self.s_tone, btn("Skin Tone Up", self.m_tone), btn("Reduce oily shine", self.m_shine),
                                           QLabel("Pimple / spot sensitivity"), self.s_pimple, btn("Pimple / Spot Auto Remove", self.m_pimple)]))
        l.addWidget(group("RETOUCH STUDIO  (zoom + click)", [btn("Open Retouch Studio (F6)", lambda: self.open_retouch("heal"), BTN_PURPLE, "", 38), self.rt_tool_box, QLabel("Brush size"), self.s_brush,
                                                             _lbl("Zoom in, then click on the pimple. Scroll = zoom, right-drag = move, Enter = done, Esc = cancel.", True, "color:#666;"),
                                                             btn("Done (Enter)", lambda: self.close_retouch())]))
        l.addWidget(group("FACE RESTORE / UPSCALE", [QLabel("Face restore strength (low = closer to the real face)"), self.s_aiface,
                                                     row(btn("AI Face Restore", self.m_face_ai), btn("AI Upscale 2x", self.m_upscale))]))
        l.addWidget(group("HAIR COLOR", [self.hair_box, btn("Custom color...", self.hair_custom), QLabel("Strength"), self.s_hair,
                                         row(btn("Apply Hair Color", lambda: self.m_hair()), btn("Hair Area (brush)", self.hair_edit))]))
        l.addWidget(group("NAME / DATE under photo", [self.cap_edit, self.c_date]))
        l.addStretch(1); return _scroll(l)

    def _tab_bg(self):
        l = QVBoxLayout()
        l.addWidget(group("REMOVE BACKGROUND   ব্যাকগ্রাউন্ড মুছুন", [QLabel("Quality (Best hair = BiRefNet, needs the model download)"), self.model_box,
                                                                  row(btn("Remove Background", self.m_bg_remove, BTN_BLUE, "", 40), btn("Restore Background", self.m_bg_restore))]))
        l.addWidget(group("BACKGROUND COLOR", [row(*[btn(n, lambda c=c: self.set_bg(c)) for n, c in BG_COLORS.items()]), btn("Custom color...", self.pick_bg)]))
        l.addWidget(group("HAIR EDGE REFINEMENT", [QLabel("Edge cleaning strength"), self.s_edge, btn("Refine Hair Edges", self.m_edge_refine),
                                                   _lbl("Cleans halos and rough edges around hair. Also used automatically after Remove Background.", True, "color:#666;")]))
        l.addStretch(1); return _scroll(l)

    def _tab_dress(self):
        l = QVBoxLayout()
        l.addWidget(group("SHIRT / SUIT / TIE / COAT", [self.dress_cat, self.dress_list, _lbl("Double-click a dress. Put transparent PNGs in the 'dress' folder (sub-folders Shirt, Suit, Tie, Coat).", True, "color:#666;"),
                                                        row(btn("Custom PNG...", self.pick_dress), btn("Remove dress", lambda: self.load_dress(None))), btn("Reload list", self.refresh_dress_tab)]))
        l.addWidget(group("DRESS POSITION", [QLabel("Size"), self.s_dress, QLabel("Up / Down"), self.s_dress_y, QLabel("Left / Right"), self.s_dress_x]))
        l.addStretch(1); return _scroll(l)

    def _tab_doc(self):
        l = QVBoxLayout()
        l.addWidget(group("DOCUMENT STUDIO", [btn("📄 Open Document Studio  (Ctrl+D)", self.open_doc_studio, BTN_BLUE, "", 46),
                                              btn("Send current photo to Document Studio", self.send_to_doc),
                                              _lbl("• Photo of a paper → straight, clean A4 page\n• Auto edge detection + perspective correction\n• Shadow removal, white background, sharpen\n"
                                                   "• Signature crop (white or transparent PNG)\n• Many pages → one PDF, image to PDF, print on A4")]))
        l.addStretch(1); return _scroll(l)

    def _tab_print(self):
        l = QVBoxLayout()
        l.addWidget(group("PAPER", [QLabel("Paper and copies are chosen at the top bar."), self.sp_pw, self.sp_ph, row(QLabel("Orientation"), self.orient_box),
                                    row(QLabel("Cut lines"), self.cut_box), self.sp_gap, self.sp_margin, self.layout_lbl]))
        l.addWidget(group("MIXED SHEET", [self.c_mix, self.size2_box, self.copies2]))
        l.addWidget(group("PRINT", [btn("Sheet Layout / Preview (Ctrl+L)", self.do_layout, "", "", 38), btn("🔍 Print Preview", self.print_preview), btn("🖨 Print...  (printer, copies)", self.print_now, BTN_BLUE, "", 40),
                                    btn("Quick Print (default printer)", self.quick_print), row(btn("Save Sheet JPG", self.save_sheet), btn("Save Sheet PDF", self.save_pdf)),
                                    btn("Save Single Photo...", self.save_single)]))
        l.addWidget(group("ONLINE FORM SAVE (KB limit)", [self.sp_kb, self.sp_ow, self.sp_oh, btn("Save for Online", self.save_online_cb)]))
        l.addStretch(1); return _scroll(l)

    # ================= menu =================
    def build_menu(self):
        mb = QMenuBar()
        def add(menu, text, fn, key=None):
            a = QAction(text, self); a.triggered.connect(lambda _=False, f=fn: f())
            if key: a.setShortcut(QKeySequence(key))
            menu.addAction(a)
        m = mb.addMenu("&File")
        add(m, "Open Photo...", self.open_photo, "Ctrl+O"); add(m, "Save (photo + sheet, automatic names)", self.save_all, "Ctrl+Shift+A")
        add(m, "Save Single Photo...", self.save_single, "Ctrl+S"); add(m, "Save Sheet (JPG)...", self.save_sheet, "Ctrl+Shift+S"); add(m, "Save Sheet (PDF)...", self.save_pdf)
        add(m, "Save for Online Form (KB limit)...", self.save_online_cb); m.addSeparator()
        add(m, "Print Preview", self.print_preview); add(m, "Print...", self.print_now, "Ctrl+P"); m.addSeparator(); add(m, "Exit", self.close)
        m = mb.addMenu("&Edit")
        add(m, "Undo", self.undo, "Ctrl+Z"); add(m, "Redo", self.redo, "Ctrl+Y"); add(m, "Reset to Original", self.reset_orig); add(m, "Original / Result", self.toggle)
        m = mb.addMenu("&AI Tools")
        add(m, "✨ AI PASSPORT PHOTO", self.ai_passport, "F2"); add(m, "AI Visa Photo", self.ai_visa); add(m, "AI ID Photo", self.ai_id); add(m, "⚡ Auto Fix", self.fix_clicked, "F9"); m.addSeparator()
        add(m, "AUTO CHALAO (old checklist)", self.run_auto, "F5"); add(m, "Face Auto Detect", self.m_face_detect, "F7"); add(m, "Face Manual (box)", self.m_face_manual, "F8"); m.addSeparator()
        add(m, "Photo Enhance PRO", self.m_enhance); add(m, "Auto Enhance", self.m_auto_enhance); add(m, "AI Face Restore", self.m_face_ai); add(m, "AI Upscale 2x", self.m_upscale); m.addSeparator()
        add(m, "Skin Retouch (natural)", self.m_retouch); add(m, "Skin Smooth (soft)", self.m_smooth); add(m, "Skin Tone Up", self.m_tone); add(m, "Reduce oily shine", self.m_shine)
        add(m, "Pimple / Spot Auto Remove", self.m_pimple); add(m, "Retouch Studio (click to heal)", lambda: self.open_retouch("heal"), "F6"); m.addSeparator()
        hm = m.addMenu("Hair Color")
        for n, c in HAIR_COLORS.items(): add(hm, n, lambda c=c: self.m_hair(c))
        hm.addSeparator(); add(hm, "Custom color...", self.hair_custom); add(hm, "Apply selected color", lambda: self.m_hair()); add(hm, "Hair Area (brush)", self.hair_edit); m.addSeparator()
        add(m, "Remove Background", self.m_bg_remove); add(m, "Restore Background", self.m_bg_restore); add(m, "Refine Hair Edges", self.m_edge_refine)
        m = mb.addMenu("&Studio")
        add(m, "Crop (Photoshop style)", self.open_editor); self.dress_menu = m.addMenu("Dress / Suit (PNG)"); self.dress_menu.aboutToShow.connect(self.fill_dress_menu)
        add(m, "Black && White on/off", lambda: self.c_bw.setChecked(not self.c_bw.isChecked())); add(m, "Sheet Layout", self.do_layout, "Ctrl+L"); add(m, "Document Studio", self.open_doc_studio, "Ctrl+D")
        m = mb.addMenu("&Shop")
        add(m, "New Job...", self.job_new); add(m, "Customer Jobs", self.open_jobs, "Ctrl+J"); add(m, "Accounts (income / expense)", self.open_accounts); m.addSeparator()
        add(m, "New size / preset...", self.preset_new); add(m, "Edit this preset", self.preset_edit); add(m, "Delete / reset preset", self.preset_delete)
        m = mb.addMenu("&Help")
        add(m, "AI Models (install / GPU)", self.open_models); add(m, "Open data folder", lambda: os.startfile(APP_DIR) if hasattr(os, "startfile") else None); add(m, "About", self.about)
        return mb

    def about(self):
        QMessageBox.about(self, APP_NAME, "<h3>%s</h3>One-click AI passport photo, print studio, document studio, customer jobs and shop accounts.<br>Data folder: %s" % (APP_TITLE, APP_DIR))

    # ================= dialogs =================
    def open_jobs(self): JobsDialog(self).exec()
    def open_accounts(self): AccountsDialog(self).exec()
    def open_models(self):
        self.models_dlg = ModelsDialog(self); self.models_dlg.exec()
    def open_doc_studio(self):
        if self.doc_studio is None: self.doc_studio = DocStudio(self)
        self.doc_studio.show(); self.doc_studio.raise_()
    def send_to_doc(self):
        if not self.need_img(): return
        self.open_doc_studio(); self.doc_studio.from_main()

    # ================= close =================
    def closeEvent(self, e):
        if self.photo_base is not None and self._rev != self._saved_rev and self.job_id is None and not self.runner.busy:
            r = QMessageBox.question(self, "Close", "You have a photo that was not saved.\nClose anyway?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes: return e.ignore()
        self.runner.cancel(); self._auto_timer.stop(); self._adjust_timer.stop()
        save_settings(); Recovery.end()                       # clean exit: nothing to recover next time
        threading.Timer(4.0, lambda: os._exit(0)).start()     # never hang on a stuck model
        e.accept()

def _excepthook(t, v, tb):
    log.error("UNCAUGHT: %s", "".join(traceback.format_exception(t, v, tb)))
    try: QMessageBox.warning(None, APP_NAME, "Something went wrong, but your work is safe.\n\n%s: %s\n\nDetails: %s" % (t.__name__, v, LOG_PATH))
    except Exception: pass

def main():
    app = QApplication(sys.argv); app.setApplicationName(APP_NAME); sys.excepthook = _excepthook
    w = AppV3(); w.show()
    QTimer.singleShot(150, w.try_recover); QTimer.singleShot(900, w.startup_warmup)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
