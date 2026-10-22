# -*- coding: utf-8 -*-
"""Shared helpers for the Layout Studio files (paths, image cache, Qt<->PIL, background tasks, presets)."""
import os, sys, json, logging, collections, traceback
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QImage, QImageReader
from PySide6.QtWidgets import QProgressDialog, QMessageBox

SC = 4.0                    # scene units per millimetre (zoom is a view transform, so print size never changes)

def _app_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "RiponComputer"); os.makedirs(d, exist_ok=True); return d

APP_DIR = _app_dir()
MODELS_DIR = os.path.join(APP_DIR, "models")
CACHE_DIR = os.path.join(APP_DIR, "layout_cache")          # derived files (crops, restored scans); originals are never touched
RECOVERY_DIR = os.path.join(APP_DIR, "recovery")
LAYOUT_PRESETS_PATH = os.path.join(APP_DIR, "layout_presets.json")
for _d in (MODELS_DIR, CACHE_DIR, RECOVERY_DIR): os.makedirs(_d, exist_ok=True)
log = logging.getLogger("ripon")

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All files (*.*)"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def unique_path(p):
    if not os.path.exists(p): return p
    root, ext = os.path.splitext(p); i = 2
    while os.path.exists("%s_%d%s" % (root, i, ext)): i += 1
    return "%s_%d%s" % (root, i, ext)

def pil_open(path):
    """open any image for editing/export: EXIF-rotated (phone photos), RGB/RGBA only. Read-only use of the original."""
    im = Image.open(path); im = ImageOps.exif_transpose(im)
    return im.convert("RGBA" if "A" in im.getbands() else "RGB")

def qimage_to_pil(img):
    img = img.convertToFormat(QImage.Format_RGBA8888); w, h = img.width(), img.height()
    return Image.frombuffer("RGBA", (w, h), bytes(img.constBits())[:img.bytesPerLine() * h], "raw", "RGBA", img.bytesPerLine(), 1).convert("RGB")

def pil_to_qimage(im):
    im = im.convert("RGBA"); w, h = im.size
    return QImage(im.tobytes("raw", "RGBA"), w, h, w * 4, QImage.Format_RGBA8888).copy()

class ImageCache:
    """Display copies of images, capped in size and count (memory control). Export always re-reads the full-resolution original."""
    def __init__(self, max_items=24, max_side=2400):
        self.max_items, self.max_side = max_items, max_side; self.d = collections.OrderedDict()
    def clear(self): self.d.clear()
    def qimage(self, path):
        if not path or not os.path.isfile(path): return None
        key = (path, os.path.getmtime(path))
        if key in self.d: self.d.move_to_end(key); return self.d[key]
        img = None
        try:
            rd = QImageReader(path); rd.setAutoTransform(True); sz = rd.size()
            if sz.isValid() and max(sz.width(), sz.height()) > self.max_side:
                k = self.max_side / float(max(sz.width(), sz.height())); rd.setScaledSize(QSize(max(1, int(sz.width() * k)), max(1, int(sz.height() * k))))
            img = rd.read()
        except Exception: img = None
        if img is None or img.isNull():                       # formats Qt cannot read -> Pillow
            try:
                im = pil_open(path); im.thumbnail((self.max_side, self.max_side)); img = pil_to_qimage(im)
            except Exception as e:
                log.warning("cannot open %s: %s", path, e); return None
        img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        for k in [k for k in self.d if k[0] == path]: del self.d[k]
        self.d[key] = img
        while len(self.d) > self.max_items: self.d.popitem(last=False)
        return img
    def pil(self, path):
        q = self.qimage(path); return qimage_to_pil(q) if q is not None else None

class TaskThread(QThread):
    """run fn(task) off the UI thread. fn may call task.progress.emit(i, n, text) and should check task.cancelled()."""
    progress = Signal(int, int, str); done = Signal(object); failed = Signal(str)
    def __init__(self, fn, parent=None):
        super().__init__(parent); self.fn = fn; self._cancel = False
    def cancel(self): self._cancel = True
    def cancelled(self): return self._cancel
    def run(self):
        try: res = self.fn(self)
        except Exception as e:
            log.error("task failed: %s\n%s", e, traceback.format_exc()); self.failed.emit("%s: %s" % (e.__class__.__name__, e)); return
        if self._cancel: self.failed.emit("Cancelled"); return
        self.done.emit(res)

def run_task(parent, title, fn, on_done, on_fail=None, show_dialog=True):
    """start a TaskThread with a progress dialog + Cancel. Keeps a reference so the thread is not garbage-collected mid-run."""
    th = TaskThread(fn, parent); dlg = QProgressDialog(title, "Cancel", 0, 0, parent) if show_dialog else None
    if not hasattr(parent, "_tasks"): parent._tasks = []
    parent._tasks.append(th)
    def finish():
        if dlg is not None: dlg.close()
        if th in parent._tasks: parent._tasks.remove(th)
    def _prog(i, n, text):
        if dlg is not None:
            dlg.setMaximum(max(0, n)); dlg.setValue(min(i, n) if n > 0 else 0); dlg.setLabelText("%s\n%s" % (title, text))
    def _done(res): finish(); on_done(res)
    def _fail(msg):
        finish()
        if on_fail: on_fail(msg)
        elif msg != "Cancelled": QMessageBox.warning(parent, "Ripon Studio", "%s\n\n%s\n\nDetails are in the log file." % (title, msg))
    th.progress.connect(_prog); th.done.connect(_done); th.failed.connect(_fail)
    if dlg is not None:
        dlg.setWindowModality(Qt.WindowModal); dlg.setMinimumDuration(300); dlg.canceled.connect(th.cancel)
    th.start(); return th

# ---- layout presets (Duet / ID card / passport) : one JSON file, never touches presets.json of the main program ----
def load_layout_presets():
    try:
        with open(LAYOUT_PRESETS_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def store_layout_presets(d):
    tmp = LAYOUT_PRESETS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(tmp, LAYOUT_PRESETS_PATH)

def cover_crop(img_w, img_h, target_ratio):
    """non-destructive centre crop (fractions l,t,r,b) so the image fills a box of target_ratio (w/h) without distortion"""
    r = img_w / float(img_h)
    if abs(r - target_ratio) < 1e-4: return None
    if r > target_ratio: k = target_ratio / r; return ((1 - k) / 2.0, 0.0, (1 + k) / 2.0, 1.0)
    k = r / target_ratio; return (0.0, (1 - k) / 2.0, 1.0, (1 + k) / 2.0)
