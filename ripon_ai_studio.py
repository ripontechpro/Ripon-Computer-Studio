# -*- coding: utf-8 -*-
"""
=====================================================================
 RIPON AI STUDIO  -  extra AI tools for Ripon Computer (Photostudio.py)
=====================================================================
 Put this file NEXT TO Photostudio.py.  The main program imports it automatically;
 if this file is missing or broken the main program still starts and works as before.

 Tools (AI menu -> "AI Studio", Ctrl+Shift+I, or the purple "AI Studio" button):
   1  AI Upscale 2x / 4x          Real-ESRGAN (ncnn-Vulkan, runs on the RTX 4060, CPU retry)
   2  Face Restore                CodeFormer (ONNX) or GFPGAN 1.4 (ONNX), all faces
   3  Skin Enhance                optional AI face restore + the program's own natural skin retouch
   4  Object Removal              LaMa big-lama (ONNX): paint the object, or auto dust / spots / scratches
   5  Background Remove/Replace   BiRefNet (portrait / lite) , MODNet, u2net ...  -> transparent / colour / blur / picture
   6  Portrait Cutout             MODNet soft portrait matte (transparent PNG)
   7  Old Photo Restoration       colour fix -> scratches (LaMa) -> denoise -> faces -> upscale   (real steps, real progress)
   8  Face Detection              InsightFace / YuNet / OpenCV, boxes + 5 landmarks, many faces

 Every tool: Auto / Manual, Quality (Fast / Balanced / Best), progress bar, Cancel, Before/After split view.
 Models are downloaded only when you press Download (AI Models window or the button in AI Studio).

 NOT included (cannot be done honestly in this program):
   SUPIR   needs SDXL + SUPIR weights (>10 GB), a CUDA PyTorch stack and normally 24 GB VRAM; it is not a pip package.
           On an 8 GB card it is impractical. Use it outside (ComfyUI) and bring the picture back with "Re-import".
   Automatic "pick any object to remove" needs a segmentation model such as SAM - not in your list, so Auto removal
           finds dust / spots / scratches only; for real objects you paint the mask.
   Colourising old black & white photos - no model for it in your list.

 Install (Windows, Python 3.14):
   pip install --upgrade PySide6 opencv-python numpy pillow rembg
   pip uninstall -y onnxruntime
   pip install onnxruntime-gpu        (cp314 Windows wheel exists; needs a recent NVIDIA driver)
=====================================================================
"""
import os, sys, re, math, time, shutil, hashlib, threading, subprocess, tempfile, collections, queue, json, traceback
import numpy as np
import cv2
from PIL import Image

G = {}        # the main program's globals (HUB, MODELS_DIR, SETTINGS, TaskRunner ...). Filled by bind().

def _log(): return G.get("log")

def _short(e, n=90):
    return (str(e).strip().splitlines() or [e.__class__.__name__])[0][:n]

class NeedModel(Exception):
    """a model that is required is not downloaded yet; .keys = registry keys"""
    def __init__(self, keys, msg=""):
        super().__init__(msg or "Model not downloaded: " + ", ".join(keys))
        self.keys = list(keys)

def _cancelled():
    return G["Cancelled"]()

# =====================================================================
#  MODEL REGISTRY ADDITIONS  (show up automatically in the main "AI Models" window)
# =====================================================================
NEW_MODELS = collections.OrderedDict([
    ("codeformer", dict(label="Face Restore: CodeFormer (ONNX)", size="~375 MB", kind="file",
                        file=os.path.join("codeformer", "codeformer.onnx"),
                        url="https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx",
                        note="Best for old / very blurry faces. Fidelity slider = real face vs. cleaner face. Needs onnxruntime (GPU: onnxruntime-gpu).")),
    ("lama", dict(label="Object Removal: LaMa big-lama (ONNX)", size="~208 MB", kind="file",
                  file=os.path.join("lama", "lama_fp32.onnx"),
                  url="https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
                  note="Removes objects / dust / scratches. Apache-2.0 licence. Works on 512 px crops around the marked area.")),
    ("modnet", dict(label="Portrait Cutout: MODNet (ONNX)", size="~26 MB", kind="file",
                    file=os.path.join("modnet", "modnet.onnx"),
                    url="https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx",
                    note="Small and fast soft portrait matte (hair). Apache-2.0 licence. Portraits only.")),
])
ONNX_ONLY = ("codeformer", "lama", "modnet")

# rough GPU memory each model needs while running (MB) - used to decide what to unload first
MODEL_VRAM = {"codeformer": 1700, "lama": 1600, "modnet": 900, "gfpgan_onnx": 1300, "face_detect": 700,
              "u2net_human_seg": 1100, "silueta": 500, "isnet-general-use": 1400,
              "birefnet-general-lite": 2800, "birefnet-portrait": 5000, "bria-rmbg": 5000}

BG_ENGINES = ["birefnet-portrait", "birefnet-general-lite", "modnet", "u2net_human_seg", "isnet-general-use", "silueta", "bria-rmbg"]
BG_ENGINE_LABEL = {"birefnet-portrait": "BiRefNet portrait (best hair, heavy)", "birefnet-general-lite": "BiRefNet lite (good, medium)",
                   "modnet": "MODNet (portrait matte, fast)", "u2net_human_seg": "u2net human (fast)",
                   "isnet-general-use": "ISNet general (sharp)", "silueta": "Silueta (tiny, very fast)", "bria-rmbg": "RMBG-2.0 (non-commercial licence)"}

def bind(host_globals):
    """called once by Photostudio.py after all its helpers exist"""
    G.clear(); G.update(host_globals)
    reg = G["MODEL_REGISTRY"]
    for k, v in NEW_MODELS.items():
        if k not in reg: reg[k] = v
    hub = G["HUB"]
    if getattr(hub, "_ais_patched", False): return
    import importlib.util as iu
    orig_lib = hub.lib_available
    def lib_available(key, _o=orig_lib):
        if key in ONNX_ONLY: return iu.find_spec("onnxruntime") is not None
        return _o(key)
    hub.lib_available = lib_available
    # lighter CUDA settings for an 8 GB card: no giant cuDNN workspace search, arena grows only as much as asked
    orig_prov = hub.ort_providers
    def ort_providers(_o=orig_prov):
        out = []
        for p in _o():
            if p == "CUDAExecutionProvider":
                out.append(("CUDAExecutionProvider", {"device_id": "0", "arena_extend_strategy": "kSameAsRequested",
                                                      "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0"}))
            else: out.append(p)
        return out
    hub.ort_providers = ort_providers
    try:                                  # lets onnxruntime-gpu find CUDA / cuDNN DLLs shipped as pip packages (ORT >= 1.21)
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"): ort.preload_dlls()
    except Exception:
        pass
    hub._ais_patched = True

# =====================================================================
#  GPU MEMORY  (nvidia-smi, no torch needed)  +  MODEL MANAGER
# =====================================================================
_VR = {"t": 0.0, "v": None}

def _nvsmi():
    for c in (shutil.which("nvidia-smi"), r"C:\Windows\System32\nvidia-smi.exe",
              r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if c and os.path.isfile(c): return c
    return None

def vram_info(max_age=1.0):
    """dict(name, total, free) in MB for the first NVIDIA GPU, or None when unknown"""
    now = time.time()
    if _VR["v"] is not None and now - _VR["t"] < max_age: return _VR["v"]
    info = None; exe = _nvsmi()
    if exe:
        try:
            p = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            parts = [x.strip() for x in (p.stdout or "").strip().splitlines()[0].split(",")]
            info = dict(name=parts[0], total=int(float(parts[1])), free=int(float(parts[2])))
        except Exception:
            info = None
    if info is None and "torch" in sys.modules:
        try:
            t = sys.modules["torch"]
            if t.cuda.is_available():
                f, tot = t.cuda.mem_get_info(); info = dict(name=t.cuda.get_device_name(0), total=int(tot / 2 ** 20), free=int(f / 2 ** 20))
        except Exception:
            pass
    _VR["t"], _VR["v"] = now, info
    return info

class ModelManager:
    """One place that loads every ONNX model ONCE (kept in the main program's HUB cache), unloads the least recently
    used GPU model when the 8 GB card is getting full, and retries on the CPU when the GPU fails."""
    def __init__(self):
        self.gpu_failed = set(); self.notes = []

    @property
    def hub(self): return G["HUB"]

    def note(self, text):
        if text not in self.notes: self.notes.append(text)
        if _log(): _log().info("AIS: %s", text)

    def take_notes(self):
        n, self.notes = self.notes, []
        return n

    def reset(self):
        self.gpu_failed.clear()

    def gpu_wanted(self, device="auto"):
        return device != "cpu" and bool(self.hub.ort_gpu_any())

    def ensure_vram(self, key, device="auto"):
        """free GPU memory for `key` by unloading the least recently used GPU models"""
        if not self.gpu_wanted(device): return None
        hub = self.hub
        with hub.lock:
            if key in hub.cache and getattr(hub.cache[key], "_ripon_gpu", False): return vram_info()
        need = MODEL_VRAM.get(key, 1500) + 300
        vi = vram_info(0.0)
        def victims():
            with hub.lock:
                return [k for k, v in hub.cache.items() if getattr(v, "_ripon_gpu", False) and k != key]
        if vi is None:                       # unknown free memory: keep at most one other GPU model
            vs = victims()
            while len(vs) > 1: hub.unload(vs.pop(0))
            return None
        guard = 0
        while vi["free"] < need and guard < 10:
            vs = victims()
            if not vs: break
            hub.unload(vs[0]); guard += 1; time.sleep(0.05); vi = vram_info(0.0)
        if vi["free"] < need:
            self.note("GPU memory is low (%d MB free, about %d MB needed for %s) - it may be slow or fall back to CPU" % (vi["free"], need, key))
        return vi

    def session(self, key, device="auto"):
        hub = self.hub
        want_gpu = self.gpu_wanted(device) and key not in self.gpu_failed
        with hub.lock:
            cur = hub.cache.get(key)
            if cur is not None and bool(getattr(cur, "_ripon_gpu", False)) == want_gpu:
                hub.cache.move_to_end(key); return cur
        if cur is not None: hub.unload(key)
        if want_gpu: self.ensure_vram(key, device)
        try:
            sess = hub._onnx_sess(key, force_cpu=not want_gpu)
        except Exception as e:
            if not want_gpu: raise
            self.note("GPU could not load %s (%s) - using CPU" % (key, _short(e, 60))); self.gpu_failed.add(key)
            hub.unload(key); sess = hub._onnx_sess(key, force_cpu=True); want_gpu = False
        try:                                  # onnxruntime silently falls back to CPU when CUDA DLLs are missing: tell the truth
            real = bool(sess.get_providers()) and sess.get_providers()[0] != "CPUExecutionProvider"
        except Exception:
            real = want_gpu
        try: sess._ripon_gpu = real
        except Exception: pass
        if want_gpu and not real:
            self.note("GPU provider not available for %s - running on CPU (check onnxruntime-gpu / CUDA / cuDNN)" % key); self.gpu_failed.add(key)
        return sess

    def run(self, key, build_feed, device="auto"):
        """build_feed(sess) -> dict. Returns the list of outputs."""
        sess = self.session(key, device)
        try:
            return sess.run(None, build_feed(sess))
        except Exception as e:
            if getattr(sess, "_ripon_gpu", False):
                self.note("GPU run failed (%s) - used CPU" % _short(e, 60)); self.gpu_failed.add(key)
                self.hub.unload(key); sess = self.session(key, "cpu")
                return sess.run(None, build_feed(sess))
            raise

    def where(self, key):
        c = self.hub.cache.get(key)
        return "-" if c is None else ("GPU" if getattr(c, "_ripon_gpu", False) else "CPU")

MGR = ModelManager()

def installed(key):
    hub = G["HUB"]
    try: return bool(hub.is_installed(key) and hub.lib_available(key))
    except Exception: return False

def _dt(inp):
    t = (getattr(inp, "type", "") or "").lower()
    return np.float64 if "double" in t else (np.float16 if "float16" in t else np.float32)

# =====================================================================
#  small shared helpers
# =====================================================================
def fingerprint(arr):
    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=12); h.update(str(a.shape).encode()); h.update(a.data)
    return h.hexdigest()

class ResultCache:
    """same picture + same tool + same settings = same result: never compute it twice (preview -> Apply, repeated clicks)"""
    def __init__(self, n=3): self.n = n; self.d = collections.OrderedDict()
    def get(self, k):
        if k in self.d: self.d.move_to_end(k); return self.d[k]
        return None
    def put(self, k, v):
        self.d[k] = v; self.d.move_to_end(k)
        while len(self.d) > self.n: self.d.popitem(last=False)
    def clear(self): self.d.clear()

RC = ResultCache(3)
_FACE_CACHE = {}

def shift_alpha(a, px):
    """grow (px>0) or shrink (px<0) a float 0..1 alpha"""
    if not px: return a
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(int(px)) * 2 + 1,) * 2)
    return cv2.dilate(a, k) if px > 0 else cv2.erode(a, k)

def checker_composite(rgb, alpha, size=14):
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    c = (((yy // size) + (xx // size)) % 2).astype(np.float32)
    bg = (200 + 40 * c)[..., None]
    a = alpha[..., None]
    return np.clip(rgb.astype(np.float32) * a + bg * (1 - a), 0, 255).astype(np.uint8)

def compose_bg(rgb, alpha, mode, color=(255, 255, 255), image_path="", blur=18):
    H, W = rgb.shape[:2]; a = alpha[..., None].astype(np.float32)
    if mode == "blur":
        bg = cv2.GaussianBlur(rgb, (0, 0), max(1.0, float(blur))).astype(np.float32)
    elif mode == "image" and image_path and os.path.isfile(image_path):
        im = Image.open(image_path).convert("RGB"); k = max(W / im.width, H / im.height)
        im = im.resize((max(W, int(round(im.width * k))), max(H, int(round(im.height * k)))), Image.LANCZOS)
        l, t = (im.width - W) // 2, (im.height - H) // 2
        bg = np.array(im.crop((l, t, l + W, t + H)), np.float32)
    else:
        bg = np.empty((H, W, 3), np.float32); bg[:] = np.array(color, np.float32)
    return np.clip(rgb.astype(np.float32) * a + bg * (1 - a), 0, 255).astype(np.uint8)

# =====================================================================
#  FACE DETECTION (many faces)  +  FACE ALIGN / RESTORE / PASTE BACK
# =====================================================================
def detect_faces(rgb, detector="auto", min_conf=0.5, min_px=0, max_faces=8):
    """returns (list of FaceInfo dicts big->small, detector name). Same dict layout as the main program (le, re, nose, ml, mr, chin, bbox ...)"""
    hub = G["HUB"]; mk = G["_fi"]; scale_fi = G["_scale_fi"]; log = _log()
    H, W = rgb.shape[:2]; s = min(1.0, 1600.0 / max(H, W))
    small = cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    order = {"auto": ["insightface", "yunet", "haar"]}.get(detector, [detector])
    faces, used = [], ""
    for det in order:
        try:
            if det == "insightface" and hub.is_installed("face_detect") and hub.lib_available("face_detect"):
                for f in hub._insight().get(cv2.cvtColor(small, cv2.COLOR_RGB2BGR)):
                    x0, y0, x1, y1 = [float(v) for v in f.bbox]; k = f.kps
                    chin = ((x0 + x1) / 2.0, y1 + 0.02 * (y1 - y0))
                    lm = getattr(f, "landmark_2d_106", None)
                    if lm is not None and len(lm) >= 33:
                        j = lm[0:33]; i = int(np.argmax(j[:, 1])); chin = (float(j[i, 0]), float(j[i, 1]))
                    faces.append(mk((x0, y0, x1 - x0, y1 - y0), k[0], k[1], k[2], k[3], k[4], chin, "insightface", float(f.det_score)))
                used = "InsightFace"
            elif det == "yunet" and hub.is_installed("yunet") and hasattr(cv2, "FaceDetectorYN"):
                h2, w2 = small.shape[:2]; d = hub._yunet(w2, h2); d.setInputSize((w2, h2))
                _, res = d.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
                for r in (res if res is not None else []):
                    x, y, w, h = [float(v) for v in r[:4]]
                    faces.append(mk((x, y, w, h), (r[4], r[5]), (r[6], r[7]), (r[8], r[9]), (r[10], r[11]), (r[12], r[13]),
                                    (x + w / 2.0, y + h * 1.03), "yunet", float(r[14])))
                used = "OpenCV YuNet"
            elif det == "haar":
                fi = G["_fi_from_haar"](small)
                if fi is not None: faces.append(fi)
                used = "OpenCV Haar (finds one face, eyes estimated)"
        except Exception as e:
            if log: log.warning("AIS face detect %s failed: %s", det, e)
            faces = []
        if faces: break
    faces = [f for f in faces if f["conf"] >= min_conf]
    faces.sort(key=lambda f: -f["bbox"][2] * f["bbox"][3])
    out = []
    for f in faces:
        f = scale_fi(f, 1.0 / s) if s < 1 else f
        if f["bbox"][2] >= min_px: out.append(f)
    return out[:max_faces], used

def face_sharpness(rgb, fi):
    x, y, w, h = [int(v) for v in fi["bbox"]]; H, W = rgb.shape[:2]
    c = rgb[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
    if c.size == 0: return 0.0
    g = cv2.cvtColor(cv2.resize(c, (160, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

_FACE_MASK = None
def _face_mask512():
    global _FACE_MASK
    if _FACE_MASK is None:
        m = np.zeros((512, 512), np.float32)
        cv2.ellipse(m, (256, 262), (178, 214), 0, 0, 360, 1.0, -1)
        _FACE_MASK = cv2.GaussianBlur(m, (0, 0), 26)
    return _FACE_MASK

def _align_matrix(fi):
    src = np.array([fi["le"], fi["re"], fi["nose"], fi["ml"], fi["mr"]], np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, G["HUB"]._FFHQ5, method=cv2.LMEDS)
    return M

def _match_mean(out, ref, mask, amount=0.8):
    """move the colour average of the restored face to the original face (restorers shift colour a little)"""
    if amount <= 0: return out
    a = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32); b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    w = (mask > 0.5)
    if int(w.sum()) < 500: return out
    d = (b[w].mean(0) - a[w].mean(0)) * amount
    a += d
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def restore_faces(rgb, faces, run512, strength=1.0, color_fix=0.8, ctx=None, label="Restoring face"):
    """run512(crop 512x512 RGB uint8) -> restored 512x512 RGB uint8.  Works on a small region per face (fast on big photos)."""
    out = rgb.copy(); H, W = rgb.shape[:2]; n = len(faces); mask512 = _face_mask512(); did = 0
    for i, fi in enumerate(faces):
        if ctx: ctx.check(); ctx.step(i, n, "%s %d / %d" % (label, i + 1, n))
        M = _align_matrix(fi)
        if M is None: continue
        Mi = cv2.invertAffineTransform(M)
        pts = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], np.float32)
        q = pts @ Mi[:, :2].T + Mi[:, 2]
        x0, y0 = max(0, int(q[:, 0].min()) - 6), max(0, int(q[:, 1].min()) - 6)
        x1, y1 = min(W, int(q[:, 0].max()) + 7), min(H, int(q[:, 1].max()) + 7)
        if x1 - x0 < 16 or y1 - y0 < 16: continue
        roi = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        Mr = M.copy(); Mr[:, 2] += M[:, :2] @ np.array([x0, y0], np.float64)
        sc = float(math.hypot(Mr[0, 0], Mr[1, 0]))
        if sc < 0.7: roi = cv2.GaussianBlur(roi, (0, 0), max(0.3, 0.5 / sc))      # big face squeezed into 512: avoid aliasing
        crop = cv2.warpAffine(roi, Mr, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fixed = run512(crop)
        if fixed.shape[:2] != (512, 512): fixed = cv2.resize(fixed, (512, 512), interpolation=cv2.INTER_CUBIC)
        fixed = _match_mean(fixed, crop, mask512, color_fix)
        Mir = cv2.invertAffineTransform(Mr); rw, rh = x1 - x0, y1 - y0
        back = cv2.warpAffine(fixed, Mir, (rw, rh), flags=cv2.INTER_LINEAR)
        m = cv2.warpAffine(mask512, Mir, (rw, rh), flags=cv2.INTER_LINEAR)[..., None] * float(np.clip(strength, 0, 1))
        cur = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(cur * (1 - m) + back.astype(np.float32) * m, 0, 255).astype(np.uint8)
        did += 1
    if did == 0: raise RuntimeError("Could not align any face for restoring")
    return out, did

def _face_feed(x, sess, weight=None):
    ins = sess.get_inputs(); feed = {ins[0].name: x.astype(_dt(ins[0]))}
    if len(ins) > 1 and weight is not None:
        shp = getattr(ins[1], "shape", None); dt = _dt(ins[1])
        feed[ins[1].name] = np.array(weight, dtype=dt) if (isinstance(shp, (list, tuple)) and len(shp) == 0) else np.array([weight], dtype=dt)
    return feed

def make_face_runner(engine, fidelity, device):
    """-> function(crop512 RGB uint8) -> restored 512x512 RGB uint8   (engine: 'codeformer' | 'gfpgan_onnx')"""
    key = "codeformer" if engine == "codeformer" else "gfpgan_onnx"
    def run(crop):
        x = ((crop.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        o = MGR.run(key, lambda s: _face_feed(x, s, fidelity if key == "codeformer" else None), device)[0]
        o = np.clip(np.asarray(o, np.float32)[0].transpose(1, 2, 0), -1, 1)
        return ((o + 1.0) * 127.5).astype(np.uint8)
    return run

# =====================================================================
#  REAL-ESRGAN UPSCALE  (ncnn-Vulkan program: GPU on RTX, real progress %, cancel, CPU retry)
# =====================================================================
_VK = {"devs": None}

def _run_proc(ctx, cmd, cwd=None, label="Working", timeout=1800):
    """run a program, report its 'NN.NN%' lines as real progress, honour Cancel (process is killed)"""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    q = queue.Queue()
    def reader():
        buf = b""
        try:
            while True:
                ch = p.stdout.read(1)
                if not ch: break
                if ch in (b"\n", b"\r"):
                    if buf: q.put(buf.decode("utf-8", "ignore")); buf = b""
                else: buf += ch
            if buf: q.put(buf.decode("utf-8", "ignore"))
        except Exception:
            pass
    th = threading.Thread(target=reader, daemon=True); th.start()
    lines = []; t0 = time.time()
    def drain():
        while True:
            try: ln = q.get_nowait()
            except queue.Empty: return
            lines.append(ln); m = re.match(r"^\s*(\d+(?:\.\d+)?)%", ln)
            if m and ctx: ctx.frac("%s  %d%%" % (label, int(float(m.group(1)))), float(m.group(1)) / 100.0)
    try:
        while p.poll() is None:
            if ctx: ctx.check()
            drain(); time.sleep(0.08)
            if time.time() - t0 > timeout: raise RuntimeError("The program took too long and was stopped")
    except BaseException:
        try: p.kill()
        except Exception: pass
        raise
    th.join(1.0); drain()
    return p.returncode, lines

def vulkan_devices(exe):
    """[(id, name)] as printed by the ncnn program (cached). Lets 'Auto' prefer the NVIDIA card over an AMD iGPU."""
    if _VK["devs"] is not None: return _VK["devs"]
    devs = []
    try:
        mdir = os.path.join(os.path.dirname(exe), "models"); model = None
        for m, s in (("realesrgan-x4plus", 4), ("realesr-animevideov3", 2)):
            if os.path.isfile(os.path.join(mdir, (m if s == 4 else "%s-x%d" % (m, s)) + ".bin")): model = (m, s); break
        if model:
            tmp = tempfile.mkdtemp(prefix="ripon_vk_")
            try:
                src, dst = os.path.join(tmp, "i.png"), os.path.join(tmp, "o.png")
                Image.fromarray(np.full((16, 16, 3), 127, np.uint8)).save(src)
                rc, lines = _run_proc(None, [exe, "-i", src, "-o", dst, "-n", model[0], "-s", str(model[1]), "-m", mdir, "-g", "-1"],
                                      cwd=os.path.dirname(exe), timeout=60)
                for ln in lines:
                    m = re.match(r"^\[(\d+) (.+?)\]\s+queueC", ln)
                    if m and (int(m.group(1)), m.group(2)) not in devs: devs.append((int(m.group(1)), m.group(2)))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        if _log(): _log().info("AIS vulkan probe failed: %s", e)
    _VK["devs"] = devs
    return devs

def _auto_vk_gpu(exe):
    for i, name in vulkan_devices(exe):
        if re.search(r"nvidia|geforce|rtx|gtx", name, re.I): return i, name
    return None, ""

def upscale_ncnn(ctx, rgb, scale=2, model_pref="photo", tile=0, tta=False, gpu="auto", device="auto"):
    """-> (rgb_out uint8, info text).  model_pref: fast | photo | anime | net"""
    hub = G["HUB"]
    if not installed("esrgan_ncnn"): raise NeedModel(["esrgan_ncnn"])
    exe = hub.path_of("esrgan_ncnn"); mdir = os.path.join(os.path.dirname(exe), "models")
    H, W = rgb.shape[:2]; scale = 4 if int(scale) >= 4 else 2; notes = []
    def has(m, s=None): return os.path.isfile(os.path.join(mdir, (m if s is None else "%s-x%d" % (m, s)) + ".bin"))
    fast_ok = has("realesr-animevideov3", scale)
    cand = {"fast": ("realesr-animevideov3", scale) if fast_ok else None,
            "photo": ("realesrgan-x4plus", 4) if has("realesrgan-x4plus") else None,
            "anime": ("realesrgan-x4plus-anime", 4) if has("realesrgan-x4plus-anime") else None,
            "net": ("realesrnet-x4plus", 4) if has("realesrnet-x4plus") else None}
    pick = cand.get(model_pref) or cand["photo"] or cand["fast"] or cand["anime"] or cand["net"]
    if pick is None: raise RuntimeError("Real-ESRGAN model files were not found in:\n" + mdir)
    model, s = pick
    if cand.get(model_pref) is None and model_pref != "photo": notes.append("model '%s' not found, used %s" % (model_pref, model))
    if max(H, W) * s > 12000 and fast_ok and model != "realesr-animevideov3":
        model, s = "realesr-animevideov3", scale; notes.append("big photo: used the fast model to keep memory low")
    if max(H, W) * scale > 16000: raise RuntimeError("The result would be larger than 16000 px. Choose 2x or a smaller photo.")
    # which processor
    if device == "cpu" or gpu == "cpu": gid = "-1"
    elif str(gpu).isdigit(): gid = str(gpu)
    else:
        i, name = _auto_vk_gpu(exe); gid = str(i) if i is not None else None
    if tile <= 0:                                           # pick a tile that fits the free GPU memory
        vi = vram_info(0.0)
        if gid == "-1": tile = 200
        elif vi is None: tile = 0
        elif vi["free"] >= 5000: tile = 0
        elif vi["free"] >= 3000: tile = 400
        elif vi["free"] >= 1800: tile = 256
        else: tile = 128
    tmp = tempfile.mkdtemp(prefix="ripon_up_")
    try:
        src, dst = os.path.join(tmp, "in.png"), os.path.join(tmp, "out.png")
        Image.fromarray(rgb).save(src, "PNG", compress_level=1)
        def cmd_for(g):
            c = [exe, "-i", src, "-o", dst, "-n", model, "-s", str(s), "-m", mdir, "-f", "png"]
            if tile > 0: c += ["-t", str(tile)]
            if g is not None: c += ["-g", g]
            if tta: c += ["-x"]
            return c
        label = "Upscaling x%d (%s)" % (scale, model)
        rc, lines = _run_proc(ctx, cmd_for(gid), cwd=os.path.dirname(exe), label=label)
        if (rc != 0 or not os.path.isfile(dst)) and gid != "-1":
            MGR.note("GPU upscale failed - retried on the CPU (slow)")
            if os.path.isfile(dst):
                try: os.remove(dst)
                except OSError: pass
            if ctx: ctx.msg("GPU failed, trying CPU (slow) ...")
            rc, lines = _run_proc(ctx, cmd_for("-1"), cwd=os.path.dirname(exe), label=label + " [CPU]"); gid = "-1"
        if rc != 0 or not os.path.isfile(dst):
            raise RuntimeError("Real-ESRGAN failed: " + " | ".join(lines[-3:])[-200:])
        im = Image.open(dst).convert("RGB"); im.load()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tw, th = W * scale, H * scale
    if im.size != (tw, th):
        a = np.array(im)
        a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA if a.shape[1] > tw else cv2.INTER_LANCZOS4)
    else: a = np.array(im)
    if gid == "-1": used = "CPU"
    else:
        used = "GPU"
        for i, name in vulkan_devices(exe):
            if gid is None or str(i) == gid:
                if gid is not None or re.search(r"nvidia|geforce|rtx|gtx", name, re.I): used = "GPU " + name; break
    return a, "Real-ESRGAN %s x%d on %s, tile %s%s%s" % (model, scale, used, tile or "auto", ", TTA" if tta else "", ("; " + "; ".join(notes)) if notes else "")

# =====================================================================
#  LaMa OBJECT REMOVAL  (ONNX, fixed 512x512 -> crop around the marked area, fill, paste back only the marked area)
# =====================================================================
def _lama512(img512, mask512, device):
    x = (img512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    m = (mask512 > 127).astype(np.float32)[None, None]
    def build(sess):
        ins = sess.get_inputs(); names = [i.name for i in ins]
        ni = "image" if "image" in names else names[0]; nm = "mask" if "mask" in names else names[1]
        return {ni: x.astype(_dt(ins[names.index(ni)])), nm: m.astype(_dt(ins[names.index(nm)]))}
    o = np.asarray(MGR.run("lama", build, device)[0], np.float32)[0].transpose(1, 2, 0)
    if float(o.max()) <= 1.5: o = o * 255.0                  # some exports return 0..1
    return np.clip(o, 0, 255).astype(np.uint8)

def _mask_groups(mb, max_groups=24):
    H, W = mb.shape; m8 = mb.astype(np.uint8); k = max(9, int(0.015 * max(H, W)))
    n = lab = st = None
    for _ in range(6):
        kk = int(k) | 1
        closed = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        n, lab, st, _c = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if n - 1 <= max_groups: break
        k *= 2
    out = []
    for i in range(1, n):
        x, y, w, h = [int(v) for v in st[i, :4]]
        ys, xs = np.where((lab[y:y + h, x:x + w] == i) & mb[y:y + h, x:x + w])
        if len(xs): out.append((x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1))
    return out

def _crop_box(bb, H, W, ctx_scale, min_side=320):
    x0, y0, x1, y1 = bb; bw, bh = x1 - x0, y1 - y0
    side = min(max(max(bw, bh) * ctx_scale, min_side), max(H, W)); cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if side <= min(H, W):
        s = int(round(side)); l = int(min(max(0, round(cx - s / 2.0)), W - s)); t = int(min(max(0, round(cy - s / 2.0)), H - s))
        if l <= x0 and t <= y0 and l + s >= x1 and t + s >= y1: return l, t, l + s, t + s
    mx, my = int(bw * (ctx_scale - 1) / 2.0) + 8, int(bh * (ctx_scale - 1) / 2.0) + 8
    return max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)

def lama_inpaint(rgb, mask, ctx=None, device="auto", ctx_scale=2.2, dilate=6, feather=3, label="Removing"):
    """rgb HxWx3 uint8, mask HxW uint8 (>127 = remove). Returns (new_rgb, number_of_areas)"""
    H, W = rgb.shape[:2]; mb = mask > 127
    if not mb.any(): return rgb, 0
    if dilate > 0:
        mb = cv2.dilate(mb.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate) * 2 + 1,) * 2)) > 0
    groups = _mask_groups(mb); out = rgb.copy(); n = len(groups)
    for i, bb in enumerate(groups):
        if ctx: ctx.check(); ctx.step(i, n, "%s object %d / %d" % (label, i + 1, n))
        l, t, r, b = _crop_box(bb, H, W, ctx_scale)
        ci = np.ascontiguousarray(out[t:b, l:r]); cm = mb[t:b, l:r].astype(np.uint8) * 255
        h0, w0 = ci.shape[:2]
        i512 = cv2.resize(ci, (512, 512), interpolation=cv2.INTER_AREA if max(h0, w0) > 512 else cv2.INTER_CUBIC)
        m512 = (cv2.resize(cm, (512, 512), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
        o512 = _lama512(i512, m512, device)
        fill = cv2.resize(o512, (w0, h0), interpolation=cv2.INTER_AREA if max(h0, w0) > 512 else cv2.INTER_CUBIC)
        a = cv2.GaussianBlur(cm.astype(np.float32) / 255.0, (0, 0), max(0.6, float(feather)))
        a = np.clip((a - 0.12) / 0.7, 0, 1)[..., None] * (cv2.dilate(cm, np.ones((3, 3), np.uint8)) > 0)[..., None]
        out[t:b, l:r] = np.clip(ci.astype(np.float32) * (1 - a) + fill.astype(np.float32) * a, 0, 255).astype(np.uint8)
    return out, n

def find_defects(rgb, faces=None, sens=50, max_cover=0.03):
    """dust / spots / thin scratches (classical detector, not AI). Returns uint8 mask HxW (255 = defect)."""
    H, W = rgb.shape[:2]; s = min(1.0, 1600.0 / max(H, W))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if s < 1: g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = g.shape; k = max(3, int(round(7 * max(h, w) / 1600.0)) | 1)
    ker = np.ones((k, k), np.uint8)
    d = np.maximum(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker), cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker)).astype(np.float32)
    t = 40.0 - 0.3 * float(sens)
    for _ in range(8):
        m = (d > t).astype(np.uint8)
        if m.mean() <= max_cover: break
        t *= 1.4
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    amax = 0.002 * h * w; keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] <= amax: keep[lab == i] = 1
    for f in (faces or []):                                   # never touch eyes, brows, nose, mouth
        x, y, fw, fh = [v * s for v in f["bbox"]]
        for a0, b0, c0, e0 in ((0.0, 0.20, 1.0, 0.50), (0.30, 0.50, 0.70, 0.90)):
            keep[max(0, int(y + b0 * fh)):max(0, int(y + e0 * fh)), max(0, int(x + a0 * fw)):max(0, int(x + c0 * fw))] = 0
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    if s < 1: keep = cv2.resize(keep, (W, H), interpolation=cv2.INTER_NEAREST)
    return keep * 255

# =====================================================================
#  MODNet portrait matte  /  background segmentation front door
# =====================================================================
def modnet_matte(rgb, ref=512, device="auto"):
    """float alpha HxW 0..1.  Official MODNet resize rule, sizes are multiples of 32."""
    H, W = rgb.shape[:2]; ref = int(ref)
    if max(H, W) < ref or min(H, W) > ref:          # official MODNet rule: the shorter side becomes `ref`
        rh, rw = (ref, int(W / H * ref)) if W >= H else (int(H / W * ref), ref)
    else: rh, rw = H, W
    rw, rh = max(32, rw - rw % 32), max(32, rh - rh % 32)
    im = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA if (rw < W) else cv2.INTER_CUBIC)
    x = ((im.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    o = MGR.run("modnet", lambda s: {s.get_inputs()[0].name: x.astype(_dt(s.get_inputs()[0]))}, device)[0]
    m = np.asarray(o, np.float32).reshape(rh, rw)
    return np.clip(cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)

def segment_alpha(rgb, engine, quality="balanced", device="auto"):
    hub = G["HUB"]
    if engine == "modnet":
        if not installed("modnet"): raise NeedModel(["modnet"])
        return modnet_matte(rgb, {"fast": 512, "balanced": 640, "best": 768}.get(quality, 640), device)
    if not installed(engine): raise NeedModel([engine])
    ms = {"fast": 1024, "balanced": 1280, "best": 1600}.get(quality, 1280)
    MGR.ensure_vram(engine, device)
    S = G["SETTINGS"]; old = S.get("gpu", True)
    try:
        if device == "cpu": hub.unload(engine); S["gpu"] = False
        a = hub.segment(rgb, engine, ms)                         # the main program's loader: GPU first, CPU retry
    finally:
        S["gpu"] = old
        if device == "cpu": hub.unload(engine)                   # do not leave a CPU session in the cache
    while hub.notes: MGR.note(hub.notes.pop(0))
    return a

# =====================================================================
#  TOOLS  (settings, Auto logic, run).  No Qt code in here.
# =====================================================================
QUALITIES = [("fast", "Fast"), ("balanced", "Balanced"), ("best", "Best")]

class P:
    """one setting. kind: combo | int | float | bool | color | file.  always=True -> shown in Auto mode too"""
    def __init__(self, key, label, kind, default, options=None, lo=0, hi=100, tip="", always=False, step=1):
        self.key, self.label, self.kind, self.default, self.options = key, label, kind, default, options
        self.lo, self.hi, self.tip, self.always, self.step = lo, hi, tip, always, step

class Env:
    def __init__(self, device="auto", quality="balanced", mask=None):
        self.device, self.quality, self.mask = device, quality, mask

def faces_for(src, ctx=None, detector="auto", min_conf=0.5, min_px=0, max_faces=8):
    key = (fingerprint(src), detector, round(float(min_conf), 2), int(min_px), int(max_faces))
    if key in _FACE_CACHE: return _FACE_CACHE[key]
    if ctx: ctx.msg("Looking for faces...")
    r = detect_faces(src, detector, min_conf, min_px, max_faces)
    if len(_FACE_CACHE) > 6: _FACE_CACHE.pop(next(iter(_FACE_CACHE)))
    _FACE_CACHE[key] = r
    return r

def first_installed(keys):
    for k in keys:
        if installed(k): return k
    return None

class Tool:
    key = ""; title = ""; blurb = ""; params = []; kind = "rgb"
    def defaults(self, q):
        d = {p.key: p.default for p in self.params}; d.update(self.tier(q)); return d
    def tier(self, q): return {}
    def needs(self, st, auto, q): return []                  # groups: at least one key of each group must be installed
    def wants(self, st, auto, q): return []                  # optional models
    def auto(self, ctx, env, src, st): return dict(st), ""
    def run(self, ctx, env, src, st): raise NotImplementedError

# ---------------------------------------------------------------- 1. upscale
class UpscaleTool(Tool):
    key, title = "upscale", "AI Upscale 2x / 4x  (Real-ESRGAN)"
    blurb = "Makes a small or soft photo bigger with real detail. Runs on the GPU through Vulkan (works on your RTX 4060)."
    params = [P("scale", "Scale", "combo", 2, [("2x", 2), ("4x", 4)], always=True),
              P("model", "Model", "combo", "photo", [("Photo (best, x4plus)", "photo"), ("Fast (animevideov3)", "fast"), ("Anime / drawing", "anime"), ("Smoother (x4 net)", "net")]),
              P("tile", "Tile size (0 = auto, smaller = less GPU memory)", "combo", 0, [("Auto", 0), ("128", 128), ("200", 200), ("256", 256), ("400", 400), ("512", 512)]),
              P("tta", "TTA (8x slower, a little cleaner)", "bool", False),
              P("gpu", "Processor", "combo", "auto", [("Auto (prefer NVIDIA)", "auto"), ("GPU 0", "0"), ("GPU 1", "1"), ("GPU 2", "2"), ("CPU (slow)", "cpu")])]
    def tier(self, q): return {"model": {"fast": "fast", "balanced": "photo", "best": "photo"}[q], "tta": q == "best"}
    def needs(self, st, auto, q): return [["esrgan_ncnn"]]
    def auto(self, ctx, env, src, st):
        d = self.defaults(env.quality); d["scale"] = st.get("scale", 2); why = []
        H, W = src.shape[:2]
        if max(H, W) * d["scale"] > 16000: d["scale"] = 2; why.append("too big for 4x, using 2x")
        if d["tta"] and H * W > 2e6: d["tta"] = False; why.append("TTA skipped (large photo)")
        if max(H, W) >= 3500: why.append("photo is already large - upscale may not be needed")
        return d, "; ".join(why)
    def run(self, ctx, env, src, st):
        out, info = upscale_ncnn(ctx, src, st["scale"], st["model"], int(st["tile"]), bool(st["tta"]), st["gpu"], env.device)
        return dict(rgb=out, kind="upscale", info=info)

# ---------------------------------------------------------------- 2. face restore
class RestoreTool(Tool):
    key, title = "restore", "Face Restore  (CodeFormer / GFPGAN)"
    blurb = "Sharpens and rebuilds blurry, low-quality faces. Works on every face it finds, then blends them back softly."
    params = [P("faces", "Faces", "combo", "all", [("All faces", "all"), ("Biggest face only", "main")], always=True),
              P("engine", "Restorer", "combo", "codeformer", [("CodeFormer", "codeformer"), ("GFPGAN 1.4", "gfpgan_onnx")]),
              P("fidelity", "CodeFormer fidelity  (low = cleaner, high = more like the real face)", "int", 70, lo=0, hi=100),
              P("strength", "Blend strength", "int", 90, lo=10, hi=100),
              P("color", "Keep original colour", "bool", True)]
    def tier(self, q):
        return {"fast": dict(engine="gfpgan_onnx", fidelity=70, strength=85), "balanced": dict(engine="codeformer", fidelity=70, strength=90),
                "best": dict(engine="codeformer", fidelity=60, strength=100)}[q]
    def needs(self, st, auto, q):
        return [["codeformer", "gfpgan_onnx"]] if auto else [[st["engine"]]]
    def auto(self, ctx, env, src, st):
        d = self.defaults(env.quality); d["faces"] = st.get("faces", "all"); why = []
        order = ["gfpgan_onnx", "codeformer"] if env.quality == "fast" else ["codeformer", "gfpgan_onnx"]
        d["engine"] = first_installed(order) or order[0]
        faces, used = faces_for(src, ctx)
        if faces:
            sh = face_sharpness(src, faces[0]); size = faces[0]["bbox"][2]
            if sh < 25 or size < 110: d["fidelity"] = max(35, d["fidelity"] - 15); d["strength"] = 100; why.append("face is very soft/small -> stronger restore")
            elif sh > 120: d["strength"] = int(d["strength"] * 0.7); d["fidelity"] = min(90, d["fidelity"] + 10); why.append("face is already sharp -> light touch")
            else: why.append("face is a little soft -> normal restore")
        return d, "; ".join(why)
    def run(self, ctx, env, src, st):
        key = st["engine"]
        if not installed(key): raise NeedModel([key])
        faces, used = faces_for(src, ctx)
        if not faces: raise RuntimeError("No face found. Try the Face Detection tool, or use 'Face Manual' in the main window.")
        if st["faces"] == "main": faces = faces[:1]
        run = make_face_runner(key, st["fidelity"] / 100.0, env.device)
        out, n = restore_faces(src, faces, run, st["strength"] / 100.0, 0.8 if st["color"] else 0.0, ctx)
        return dict(rgb=out, kind="rgb", info="%s restored %d face(s) via %s on %s" % ("CodeFormer" if key == "codeformer" else "GFPGAN", n, used, MGR.where(key)))

# ---------------------------------------------------------------- 3. skin enhance
class SkinTool(Tool):
    key, title = "skin", "Skin Enhancement"
    blurb = ("Natural skin retouch (keeps pores) from the program's own skin tools, plus optional AI texture restore (CodeFormer / GFPGAN) "
             "and optional spot healing with LaMa. There is no separate 'skin AI model' in your list - this combines the real tools.")
    params = [P("strength", "Retouch strength", "int", 50, lo=0, hi=100, always=True),
              P("tone", "Skin tone evening", "int", 0, lo=0, hi=100),
              P("shine", "Reduce oily shine", "bool", False),
              P("ai", "AI texture restore (face restorer, light)", "bool", False),
              P("ai_amount", "AI texture amount", "int", 35, lo=10, hi=100),
              P("heal", "Heal spots / pimples with LaMa", "bool", False),
              P("sens", "Spot sensitivity", "int", 55, lo=10, hi=100)]
    def tier(self, q):
        return {"fast": dict(strength=40, ai=False, heal=False), "balanced": dict(strength=50, ai=True, ai_amount=35, heal=False),
                "best": dict(strength=55, ai=True, ai_amount=50, heal=True)}[q]
    def wants(self, st, auto, q):
        w = []
        if auto or st["ai"]: w.append("codeformer")
        if auto or st["heal"]: w.append("lama")
        return w
    def auto(self, ctx, env, src, st):
        d = self.defaults(env.quality); d["strength"] = st.get("strength", d["strength"]); why = []
        faces, used = faces_for(src, ctx)
        if faces:
            n = len(G["find_blemishes"](src, G["fi_face_tuple"](faces[0]), 60))
            if n >= 6: d["heal"] = env.quality != "fast"; why.append("%d spots found" % n)
            elif n == 0: d["heal"] = False
        d["ai"] = d["ai"] and bool(first_installed(["codeformer", "gfpgan_onnx"])); d["heal"] = d["heal"] and installed("lama")
        return d, "; ".join(why)
    def run(self, ctx, env, src, st):
        faces, used = faces_for(src, ctx)
        if not faces: raise RuntimeError("No face found. Try the Face Detection tool, or use 'Face Manual' in the main window.")
        fi = faces[0]; face = G["fi_face_tuple"](fi); cur = src; did = []; skipped = []
        steps = 1 + int(bool(st["ai"])) + int(bool(st["heal"])); i = 0
        if st["heal"]:
            ctx.check(); ctx.step(i, steps, "Healing spots..."); i += 1
            if not installed("lama"): skipped.append("spot healing (LaMa not downloaded)")
            else:
                spots = G["find_blemishes"](cur, face, st["sens"]); mk = np.zeros(cur.shape[:2], np.uint8)
                for cx, cy, r in spots: cv2.circle(mk, (int(cx), int(cy)), max(2, int(r * 1.3)), 255, -1)
                if spots:
                    cur, _n = lama_inpaint(cur, mk, ctx, env.device, 3.0, 2, 2, "Healing"); did.append("healed %d spots" % len(spots))
        if st["ai"]:
            ctx.check(); ctx.step(i, steps, "AI texture..."); i += 1
            key = first_installed(["codeformer", "gfpgan_onnx"])
            if key is None: skipped.append("AI texture (no face restorer downloaded)")
            else:
                cur, _n = restore_faces(cur, faces[:1], make_face_runner(key, 0.85, env.device), st["ai_amount"] / 100.0, 0.8, None)
                did.append("AI texture %d%%" % st["ai_amount"])
        ctx.check(); ctx.step(i, steps, "Retouching skin...")
        m = G["feature_guard"](G["skin_mask"](cur, face), fi)
        if st["strength"] > 0: cur = G["skin_retouch"](cur, m, face, st["strength"]); did.append("retouch %d%%" % st["strength"])
        if st["tone"] > 0: cur = G["skin_tone_up"](cur, m, st["tone"]); did.append("tone %d%%" % st["tone"])
        if st["shine"]: cur = G["skin_shine_fix"](cur, m, 70); did.append("shine reduced")
        return dict(rgb=cur, kind="rgb", info=", ".join(did) + (("   (skipped: " + "; ".join(skipped) + ")") if skipped else ""))

# ---------------------------------------------------------------- 4. object removal
class RemoveTool(Tool):
    key, title = "remove", "Object Removal  (LaMa)"
    blurb = ("Paint over what you want gone (left mouse; Ctrl or Erase button = erase marks) and press Run. "
             "Auto mode ALSO finds dust, small spots and thin scratches by itself (classical detector) - it cannot pick whole objects.")
    params = [P("auto_spots", "Also find dust / spots / scratches automatically", "bool", False),
              P("sens", "Auto sensitivity", "int", 50, lo=5, hi=100),
              P("dilate", "Grow marked area (px)", "int", 6, lo=0, hi=30),
              P("context", "Context around the object", "combo", 2.2, [("Small (fast)", 1.6), ("Normal", 2.2), ("Large (big objects)", 3.0)]),
              P("feather", "Edge softness (px)", "int", 3, lo=1, hi=12)]
    def tier(self, q):
        return {"fast": dict(dilate=4, context=1.6, feather=2), "balanced": dict(dilate=6, context=2.2, feather=3), "best": dict(dilate=8, context=3.0, feather=4)}[q]
    def needs(self, st, auto, q): return [["lama"]]
    def auto(self, ctx, env, src, st):
        d = self.defaults(env.quality); d["auto_spots"] = True
        return d, "dust / spot finder ON"
    def run(self, ctx, env, src, st):
        if not installed("lama"): raise NeedModel(["lama"])
        H, W = src.shape[:2]; mk = np.zeros((H, W), np.uint8); parts = []
        if env.mask is not None and env.mask.shape == (H, W) and env.mask.any(): mk = np.maximum(mk, env.mask); parts.append("painted area")
        det = None
        if st["auto_spots"]:
            ctx.msg("Looking for dust, spots and scratches...")
            faces, _u = faces_for(src, ctx); det = find_defects(src, faces, st["sens"])
            if det.any(): mk = np.maximum(mk, det); parts.append("auto-found defects")
        if not mk.any(): raise RuntimeError("Nothing to remove. Paint over the object first (or turn on the automatic finder).")
        out, n = lama_inpaint(src, mk, ctx, env.device, float(st["context"]), int(st["dilate"]), int(st["feather"]))
        return dict(rgb=out, kind="rgb", mask=det, info="Removed %d area(s) (%s) with LaMa on %s" % (n, " + ".join(parts), MGR.where("lama")))

# ---------------------------------------------------------------- 5. background remove / replace
class BgTool(Tool):
    key, title = "bg", "Background Remove / Replace"
    blurb = ("Cuts the person out with BiRefNet / MODNet / u2net. No automatic hair blur or edge smoothing is applied "
             "(the mask edge stays exactly as the model made it).")
    params = [P("mode", "Result", "combo", "transparent", [("Transparent (pick colour in main window)", "transparent"), ("Solid colour", "solid"),
                                                       ("Blurred original", "blur"), ("Picture", "image")], always=True),
              P("color", "Colour", "color", (255, 255, 255), always=True),
              P("blur", "Blur amount", "int", 18, lo=3, hi=60, always=True),
              P("image", "Picture file", "file", "", always=True),
              P("engine", "Model", "combo", "birefnet-general-lite", [(BG_ENGINE_LABEL[k], k) for k in BG_ENGINES]),
              P("edge", "Edge shift (px)  - means grow, - means shrink", "int", 0, lo=-8, hi=8),
              P("feather", "Edge feather (px)", "int", 0, lo=0, hi=10)]
    def tier(self, q): return {"fast": dict(engine="modnet"), "balanced": dict(engine="birefnet-general-lite"), "best": dict(engine="birefnet-portrait")}[q]
    def pref(self, q, has_face):
        o = {"fast": ["modnet", "u2net_human_seg", "silueta", "isnet-general-use"],
             "balanced": ["birefnet-general-lite", "modnet", "isnet-general-use", "u2net_human_seg"],
             "best": ["birefnet-portrait", "birefnet-general-lite", "modnet", "u2net_human_seg"]}[q]
        if not has_face: o = [k for k in o if k not in ("birefnet-portrait", "modnet", "u2net_human_seg")] + [k for k in o if k in ("modnet", "u2net_human_seg")]
        return o
    def needs(self, st, auto, q): return [self.pref(q, True)] if auto else [[st["engine"]]]
    def auto(self, ctx, env, src, st):
        d = self.defaults(env.quality)
        for k in ("mode", "color", "blur", "image"): d[k] = st.get(k, d[k])
        faces, _u = faces_for(src, ctx); pref = self.pref(env.quality, bool(faces))
        d["engine"] = first_installed(pref) or pref[0]
        return d, "%s (%s)" % (BG_ENGINE_LABEL.get(d["engine"], d["engine"]), "person found" if faces else "no face - general model")
    def run(self, ctx, env, src, st):
        ctx.msg("Cutting out the background (%s)..." % st["engine"]); ctx.check()
        a = segment_alpha(src, st["engine"], env.quality, env.device); ctx.check()
        if st["edge"]: a = shift_alpha(a, int(st["edge"]))
        if st["feather"]: a = cv2.GaussianBlur(a, (0, 0), float(st["feather"]))
        a = np.clip(a, 0, 1).astype(np.float32); mode = st["mode"]
        info = "Background removed with %s" % BG_ENGINE_LABEL.get(st["engine"], st["engine"])
        if mode in ("blur", "image"):
            if mode == "image" and not (st["image"] and os.path.isfile(st["image"])): raise RuntimeError("Choose a picture file for the new background.")
            flat = compose_bg(src, a, mode, st["color"], st["image"], st["blur"])
            return dict(rgb=flat, kind="flat", alpha=None, info=info + "; replaced with " + ("blurred original" if mode == "blur" else "your picture"))
        prev = compose_bg(src, a, "solid", st["color"]) if mode == "solid" else checker_composite(src, a)
        return dict(rgb=src, alpha=a, preview=prev, kind="alpha", bgcolor=tuple(st["color"]) if mode == "solid" else None, info=info)

# ---------------------------------------------------------------- 6. portrait cutout
class CutoutTool(Tool):
    key, title = "cutout", "Portrait Cutout  (MODNet)"
    blurb = "Soft, hair-friendly portrait matte as a transparent cut-out. Apply to the photo, or save a transparent PNG."
    params = [P("engine", "Model", "combo", "modnet", [("MODNet (soft portrait matte)", "modnet"), ("BiRefNet portrait (heavy)", "birefnet-portrait")], always=True),
              P("ref", "Working size (px)", "int", 640, lo=384, hi=1024),
              P("gamma", "Matte tightness  (<1 keeps more, >1 cuts tighter)", "float", 1.0, lo=0.6, hi=1.6, step=0.05),
              P("edge", "Edge shift (px)", "int", 0, lo=-8, hi=8)]
    def tier(self, q): return {"fast": dict(ref=512), "balanced": dict(ref=640), "best": dict(ref=768)}[q]
    def needs(self, st, auto, q): return [["modnet", "birefnet-portrait"]] if auto else [[st["engine"]]]
    def auto(self, ctx, env, src, st):
        d = self.defaults(env.quality); d["engine"] = st.get("engine", "modnet")
        if not installed(d["engine"]): d["engine"] = first_installed(["modnet", "birefnet-portrait"]) or d["engine"]
        return d, "%s, working size %d px" % (d["engine"], d["ref"])
    def run(self, ctx, env, src, st):
        eng = st["engine"]; ctx.msg("Matting (%s)..." % eng); ctx.check()
        if eng == "modnet":
            if not installed("modnet"): raise NeedModel(["modnet"])
            a = modnet_matte(src, int(st["ref"]), env.device)
        else:
            a = segment_alpha(src, eng, env.quality, env.device)
        if abs(float(st["gamma"]) - 1.0) > 1e-3: a = np.power(np.clip(a, 0, 1), float(st["gamma"]))
        if st["edge"]: a = shift_alpha(a.astype(np.float32), int(st["edge"]))
        a = np.clip(a, 0, 1).astype(np.float32)
        return dict(rgb=src, alpha=a, preview=checker_composite(src, a), kind="alpha", bgcolor=None, info="Portrait cutout with %s on %s" % (eng, MGR.where(eng)))

# ---------------------------------------------------------------- 7. old photo restoration
class OldPhotoTool(Tool):
    key, title = "oldphoto", "Old Photo Restoration"
    blurb = ("Runs the real steps one after another: colour fix -> scratches & dust (LaMa) -> noise -> faces (CodeFormer / GFPGAN) -> upscale (Real-ESRGAN). "
             "Auto decides which steps are needed from the photo itself. Colourising black & white is not included.")
    params = [P("color_fix", "Fix colour cast / faded contrast", "bool", True),
              P("scratch", "Remove scratches & dust (LaMa)", "bool", True),
              P("denoise", "Reduce noise / grain", "bool", True),
              P("faces", "Restore faces", "bool", True),
              P("upscale", "Upscale", "combo", 0, [("No", 0), ("2x", 2), ("4x", 4)]),
              P("sens", "Scratch sensitivity", "int", 50, lo=10, hi=100),
              P("dn", "Noise reduction strength", "int", 45, lo=10, hi=100),
              P("fidelity", "Face fidelity", "int", 65, lo=0, hi=100)]
    def tier(self, q): return {"fast": dict(dn=35, fidelity=70), "balanced": dict(dn=45, fidelity=65), "best": dict(dn=50, fidelity=55, sens=60)}[q]
    def wants(self, st, auto, q): return ["lama", "codeformer", "esrgan_ncnn"]
    def auto(self, ctx, env, src, st):
        d = self.defaults(env.quality); H, W = src.shape[:2]; why = []
        ctx.msg("Examining the photo...")
        gains = G["neutral_gains"](src); cast = float(np.abs(gains - 1.0).max())
        L = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY); d["color_fix"] = bool(cast > 0.05 or float(L.std()) < 45)
        d["denoise"] = bool(G["noise_sigma"](src) > 2.5)
        faces, _u = faces_for(src, ctx); d["faces"] = bool(faces)
        faces_for_def = faces
        cover = float((find_defects(src, faces_for_def, d["sens"]) > 0).mean()); d["scratch"] = bool(cover > 0.0003)
        long = max(H, W); d["upscale"] = 4 if long < 700 else (2 if long < 1600 else 0)
        if env.quality == "fast" and d["upscale"] == 4: d["upscale"] = 2
        why.append("cast %.2f, noise %.1f, %d face(s), defects %.2f%%, %d px" % (cast, G["noise_sigma"](src), len(faces), cover * 100, long))
        return d, "; ".join(why)
    def run(self, ctx, env, src, st):
        steps = []
        if st["color_fix"]: steps.append("color")
        if st["scratch"]: steps.append("scratch")
        if st["denoise"]: steps.append("denoise")
        if st["faces"]: steps.append("faces")
        if int(st["upscale"]): steps.append("upscale")
        if not steps: raise RuntimeError("No step selected.")
        cur = src; done = []; skipped = []; n = len(steps); faces, _u = faces_for(src, ctx)
        for i, name in enumerate(steps):
            ctx.check()
            try:
                if name == "color":
                    ctx.step(i, n, "Fixing colour...")
                    cur = G["auto_white_balance"](cur, 0.7); cur = G["gentle_enhance"](cur, 40); done.append("colour")
                elif name == "scratch":
                    ctx.step(i, n, "Looking for scratches and dust...")
                    if not installed("lama"): raise NeedModel(["lama"])
                    mk = find_defects(cur, faces, st["sens"])
                    if (mk > 0).sum() < 10: skipped.append("scratches (none found)")
                    else:
                        cur, k = lama_inpaint(cur, mk, ctx, env.device, 2.0, 2, 2, "Fixing defect"); done.append("scratches/dust (%d areas)" % k)
                elif name == "denoise":
                    ctx.step(i, n, "Reducing noise..."); cur = G["denoise_photo"](cur, int(st["dn"])); done.append("noise")
                elif name == "faces":
                    ctx.step(i, n, "Restoring faces...")
                    key = first_installed(["codeformer", "gfpgan_onnx"])
                    if key is None: raise NeedModel(["codeformer"])
                    if not faces: skipped.append("faces (none found)")
                    else:
                        cur, k = restore_faces(cur, faces, make_face_runner(key, st["fidelity"] / 100.0, env.device), 0.9, 0.8, None); done.append("%d face(s)" % k)
                elif name == "upscale":
                    ctx.step(i, n, "Upscaling...")
                    cur, _info = upscale_ncnn(ctx, cur, int(st["upscale"]), "photo", 0, False, "auto", env.device); done.append("upscale x%d" % int(st["upscale"]))
            except G["Cancelled"]:
                raise
            except NeedModel as e:
                skipped.append("%s (%s not downloaded)" % (name, ", ".join(e.keys)))
            except Exception as e:
                skipped.append("%s (%s)" % (name, _short(e, 60)))
        if not done: raise RuntimeError("Nothing could be done: " + "; ".join(skipped))
        kind = "upscale" if cur.shape != src.shape else "rgb"
        return dict(rgb=cur, kind=kind, info="Done: " + ", ".join(done) + (("   Skipped: " + "; ".join(skipped)) if skipped else ""))

# ---------------------------------------------------------------- 8. face detection
class FaceTool(Tool):
    key, title = "faces", "Face Detection"
    blurb = "Finds every face with eyes, nose and mouth points. Shows the result on the picture. 'Use as main face' hands the biggest face to the other tools."
    params = [P("detector", "Detector", "combo", "auto", [("Auto (best available)", "auto"), ("InsightFace", "insightface"), ("OpenCV YuNet", "yunet"), ("OpenCV Haar (1 face)", "haar")]),
              P("conf", "Minimum confidence %", "int", 50, lo=10, hi=95),
              P("minpx", "Minimum face width (px)", "int", 0, lo=0, hi=600),
              P("maxn", "Maximum faces", "int", 8, lo=1, hi=20)]
    def wants(self, st, auto, q): return ["yunet", "face_detect"]
    def run(self, ctx, env, src, st):
        ctx.msg("Detecting faces...")
        faces, used = detect_faces(src, st["detector"], st["conf"] / 100.0, int(st["minpx"]), int(st["maxn"]))
        H, W = src.shape[:2]; t = max(2, int(max(H, W) / 450.0)); out = src.copy()
        cols = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0), (255, 0, 255)]
        for i, f in enumerate(faces):
            x, y, w, h = [int(v) for v in f["bbox"]]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), t)
            for j, k in enumerate(("le", "re", "nose", "ml", "mr")): cv2.circle(out, (int(f[k][0]), int(f[k][1])), max(3, t * 2), cols[j], -1)
            cv2.putText(out, "%d  %.0f%%" % (i + 1, f["conf"] * 100), (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, t * 0.35), (0, 255, 0), max(1, t - 1), cv2.LINE_AA)
        if not faces: return dict(rgb=out, kind="face", faces=[], info="No face found (%s)" % (used or "no detector available"))
        lines = ["%d face(s) found with %s" % (len(faces), used)] + ["  #%d  %dx%d px  confidence %.0f%%%s" % (i + 1, f["bbox"][2], f["bbox"][3], f["conf"] * 100, "  (eyes estimated)" if f.get("est") else "") for i, f in enumerate(faces)]
        return dict(rgb=out, kind="face", faces=faces, info="\n".join(lines))

TOOLS = [UpscaleTool(), RestoreTool(), SkinTool(), RemoveTool(), BgTool(), CutoutTool(), OldPhotoTool(), FaceTool()]
TOOL_BY_KEY = {t.key: t for t in TOOLS}
TOOL_MENU = [(t.key, t.title) for t in TOOLS]

def run_tool(ctx, tool, env, src, st, auto):
    """cached: same picture + tool + settings (+ marks) never runs twice. Returns the result dict."""
    why = ""
    if auto: st, why = tool.auto(ctx, env, src, st)
    mk = fingerprint(env.mask) if (tool.key == "remove" and env.mask is not None) else ""
    key = (fingerprint(src), tool.key, json.dumps(st, sort_keys=True, default=str), env.device, env.quality if tool.key in ("bg", "cutout") else "", mk)
    hit = RC.get(key)
    if hit is not None:
        r = dict(hit); r["cached"] = True; r["settings"] = st; r["why"] = why; return r
    r = tool.run(ctx, env, src, st)
    r["settings"] = st; r["why"] = why; r["cached"] = False
    RC.put(key, r)
    return r

# =====================================================================
#  GUI  (PySide6)  -  Before/After canvas + the AI Studio window
# =====================================================================
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox, QSlider, QSpinBox,
                               QDoubleSpinBox, QGroupBox, QScrollArea, QFrame, QProgressBar, QRadioButton, QButtonGroup, QFileDialog,
                               QMessageBox, QColorDialog, QLineEdit, QSizePolicy)
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QPixmap, QBrush
from PySide6.QtCore import Qt, QRect, QRectF, QPointF, Signal, QTimer

def _qimg(arr):
    a = np.ascontiguousarray(arr)
    if a.ndim == 2: a = np.stack([a] * 3, 2)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()

class StudioCanvas(QWidget):
    """Before / After viewer: split slider, wheel zoom, right-drag pan, optional mask painting (object removal)."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 460); self.setMouseTracking(True); self.setFocusPolicy(Qt.StrongFocus)
        self.before = self.after = None; self.mode = "before"; self.split = 0.5
        self.s = None; self.ox = self.oy = 0.0           # scale (None = fit) and origin
        self.mask = None; self.extra = None; self.paint = False; self.erase = False; self.brush = 30
        self._undo = []; self._pan = None; self._stroke = None; self._mouse = None; self._cache = {}
        self.setStyleSheet("background:#2b2f33;")

    # ---- data
    def set_before(self, arr):
        self.before = arr; self.after = None; self.extra = None; self.mask = None; self._undo = []; self.mode = "before"; self.s = None; self._cache.clear(); self.update()
    def set_after(self, arr, mode="split"):
        self.after = arr; self.mode = mode if arr is not None else "before"; self._cache.clear(); self.update()
    def set_mode(self, m):
        self.mode = m if (m == "before" or self.after is not None) else "before"; self.update()
    def ensure_mask(self):
        if self.before is not None and (self.mask is None or self.mask.shape != self.before.shape[:2]):
            self.mask = np.zeros(self.before.shape[:2], np.uint8)
    def clear_mask(self):
        if self.mask is not None: self._undo.append(self.mask.copy()); self.mask[:] = 0; self.update()
    def undo_mask(self):
        if self._undo: self.mask = self._undo.pop(); self.update()
    def has_mask(self): return self.mask is not None and bool(self.mask.any())

    # ---- geometry
    def _view_size(self):
        a = self.after if self.after is not None else self.before
        return (a.shape[1], a.shape[0]) if a is not None else (1, 1)
    def _fit(self):
        vw, vh = self._view_size(); fit = min(self.width() / float(vw), self.height() / float(vh))
        if self.s is None: self.s = fit; self.ox = (self.width() - vw * fit) / 2.0; self.oy = (self.height() - vh * fit) / 2.0
        return fit
    def fit_view(self): self.s = None; self.update()
    def actual_size(self):
        if self.before is None: return
        self._fit(); vw, vh = self._view_size(); c = QPointF(self.width() / 2.0, self.height() / 2.0); self._zoom_to(1.0, c)
    def _zoom_to(self, ns, c):
        self._fit(); vx, vy = (c.x() - self.ox) / self.s, (c.y() - self.oy) / self.s
        self.s = ns; self.ox = c.x() - vx * ns; self.oy = c.y() - vy * ns; self.update()
    def _kb(self):
        vw, _ = self._view_size(); return vw / float(self.before.shape[1])
    def _to_before(self, p):
        k = self._kb() * self.s; return (p.x() - self.ox) / k, (p.y() - self.oy) / k

    # ---- drawing
    def _render(self, arr, k, vx0, vy0, vx1, vy1, dw, dh, tag):
        x0, y0 = max(0, int(math.floor(vx0 / k))), max(0, int(math.floor(vy0 / k)))
        x1, y1 = min(arr.shape[1], int(math.ceil(vx1 / k))), min(arr.shape[0], int(math.ceil(vy1 / k)))
        if x1 <= x0 or y1 <= y0 or dw < 1 or dh < 1: return None
        key = (tag, id(arr), x0, y0, x1, y1, dw, dh); hit = self._cache.get(key)
        if hit is not None: return hit
        crop = arr[y0:y1, x0:x1]; r = dw / float(x1 - x0)
        interp = cv2.INTER_AREA if r < 1 else (cv2.INTER_NEAREST if r >= 3 else cv2.INTER_LINEAR)
        if crop.ndim == 2 or crop.shape[2] == 1: crop = np.dstack([crop] * 3)
        img = cv2.resize(crop, (dw, dh), interpolation=interp)
        if len(self._cache) > 8: self._cache.clear()
        self._cache[key] = (img, x0, y0, x1, y1); return self._cache[key]

    def paintEvent(self, _e):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#2b2f33"))
        if self.before is None:
            p.setPen(QColor("#9aa5ad")); p.drawText(self.rect(), Qt.AlignCenter, "Open a photo first"); return
        self._fit(); vw, vh = self._view_size(); s = self.s
        vx0, vy0 = max(0.0, -self.ox / s), max(0.0, -self.oy / s)
        vx1, vy1 = min(float(vw), (self.width() - self.ox) / s), min(float(vh), (self.height() - self.oy) / s)
        if vx1 <= vx0 or vy1 <= vy0: return
        dx0, dy0 = int(round(self.ox + vx0 * s)), int(round(self.oy + vy0 * s)); dw, dh = int(round((vx1 - vx0) * s)), int(round((vy1 - vy0) * s))
        def draw(arr, tag, clip=None):
            k = vw / float(arr.shape[1]); r = self._render(arr, k, vx0, vy0, vx1, vy1, dw, dh, tag)
            if r is None: return
            img = r[0]; p.save()
            if clip is not None: p.setClipRect(clip)
            p.drawImage(QRect(dx0, dy0, dw, dh), _qimg(img)); p.restore()
        split_x = self.ox + self.split * vw * s
        if self.mode == "after" and self.after is not None: draw(self.after, "a")
        elif self.mode == "split" and self.after is not None:
            draw(self.before, "b", QRect(0, 0, int(split_x), self.height())); draw(self.after, "a", QRect(int(split_x), 0, self.width(), self.height()))
            p.setPen(QPen(QColor("#ffffff"), 2)); p.drawLine(int(split_x), 0, int(split_x), self.height())
            p.setBrush(QColor(255, 255, 255, 230)); p.setPen(QPen(QColor("#37474f"), 1)); p.drawEllipse(QPointF(split_x, self.height() / 2.0), 13, 13)
            p.setPen(QColor("#37474f")); p.drawText(QRectF(split_x - 13, self.height() / 2.0 - 13, 26, 26), Qt.AlignCenter, "◀▶")
        else: draw(self.before, "b")
        showing_before = self.mode in ("before", "split")
        for m, col, tag in ((self.mask, (255, 40, 40), "m"), (self.extra, (40, 120, 255), "x")):
            if m is None or not showing_before or m.shape != self.before.shape[:2] or not m.any(): continue
            k = vw / float(m.shape[1]); r = self._render(m, k, vx0, vy0, vx1, vy1, dw, dh, tag + str(id(m)) + str(int(m.sum() // 255)))
            if r is None: continue
            mm = r[0][..., 0] if r[0].ndim == 3 else r[0]; rgba = np.zeros(mm.shape + (4,), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = col; rgba[..., 3] = (mm > 100) * 120
            h2, w2 = mm.shape; p.drawImage(QRect(dx0, dy0, dw, dh), QImage(rgba.data, w2, h2, 4 * w2, QImage.Format_RGBA8888).copy())
        tag_txt = {"before": "BEFORE", "after": "AFTER"}.get(self.mode)
        if tag_txt:
            p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 150)); p.drawRoundedRect(8, 8, 74, 24, 5, 5); p.setPen(QColor("white")); p.drawText(QRect(8, 8, 74, 24), Qt.AlignCenter, tag_txt)
        if self.paint and self._mouse is not None:
            rr = self.brush * self._kb() * s / 2.0; p.setBrush(Qt.NoBrush); p.setPen(QPen(QColor(255, 255, 255) if not self.erase else QColor(80, 200, 255), 2))
            p.drawEllipse(self._mouse, rr, rr)

    # ---- mouse
    def wheelEvent(self, e):
        if self.before is None: return
        self._fit(); f = 1.2 ** (e.angleDelta().y() / 120.0); fit = min(self.width() / float(self._view_size()[0]), self.height() / float(self._view_size()[1]))
        self._zoom_to(max(fit * 0.5, min(24.0, self.s * f)), e.position())
    def _paint_to(self, pos):
        self.ensure_mask(); x, y = self._to_before(pos); r = max(1, int(self.brush / 2.0))
        pt = (int(round(x)), int(round(y))); col = 0 if self.erase else 255
        if self._stroke is None: cv2.circle(self.mask, pt, r, col, -1)
        else: cv2.line(self.mask, self._stroke, pt, col, r * 2); cv2.circle(self.mask, pt, r, col, -1)
        self._stroke = pt; self.update()
    def mousePressEvent(self, e):
        if self.before is None: return
        self._fit()
        if e.button() in (Qt.RightButton, Qt.MiddleButton): self._pan = (e.position(), self.ox, self.oy); return
        if e.button() == Qt.LeftButton:
            if self.paint and self.mode != "after":
                self.ensure_mask(); self._undo.append(self.mask.copy()); self._undo = self._undo[-12:]
                self.erase_now = bool(self.erase or (e.modifiers() & Qt.ControlModifier))
                keep = self.erase; self.erase = self.erase_now; self._stroke = None; self._paint_to(e.position()); self.erase = keep
            elif self.mode == "split" and self.after is not None: self._set_split(e.position().x())
    def _set_split(self, x):
        vw, _ = self._view_size(); self.split = min(1.0, max(0.0, (x - self.ox) / (vw * self.s))); self.update()
    def mouseMoveEvent(self, e):
        self._mouse = e.position()
        if self._pan is not None:
            p0, ox, oy = self._pan; self.ox = ox + e.position().x() - p0.x(); self.oy = oy + e.position().y() - p0.y(); self.update(); return
        if e.buttons() & Qt.LeftButton:
            if self.paint and self.mode != "after":
                keep = self.erase; self.erase = getattr(self, "erase_now", self.erase); self._paint_to(e.position()); self.erase = keep
            elif self.mode == "split" and self.after is not None: self._set_split(e.position().x())
        elif self.paint: self.update()
    def mouseReleaseEvent(self, e):
        self._pan = None; self._stroke = None
    def leaveEvent(self, e): self._mouse = None; self.update()

# ---------------------------------------------------------------------
#  one setting row  ->  (widget, getter, setter)
# ---------------------------------------------------------------------
def _make_param_widget(p, on_change):
    if p.kind == "combo":
        w = QComboBox()
        for lab, val in p.options: w.addItem(lab, val)
        def setv(v):
            i = w.findData(v)
            if i >= 0: w.setCurrentIndex(i)
        setv(p.default); w.currentIndexChanged.connect(lambda *_: on_change())
        return w, (lambda: w.currentData()), setv
    if p.kind == "int":
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0)
        sl = QSlider(Qt.Horizontal); sl.setRange(p.lo, p.hi); sp = QSpinBox(); sp.setRange(p.lo, p.hi); sp.setFixedWidth(62)
        sl.valueChanged.connect(sp.setValue); sp.valueChanged.connect(sl.setValue); sp.valueChanged.connect(lambda *_: on_change())
        sp.setValue(int(p.default)); l.addWidget(sl, 1); l.addWidget(sp)
        return w, (lambda: sp.value()), (lambda v: sp.setValue(int(v)))
    if p.kind == "float":
        w = QDoubleSpinBox(); w.setRange(p.lo, p.hi); w.setSingleStep(p.step); w.setDecimals(2); w.setValue(float(p.default)); w.valueChanged.connect(lambda *_: on_change())
        return w, (lambda: float(w.value())), (lambda v: w.setValue(float(v)))
    if p.kind == "bool":
        w = QCheckBox(p.label); w.setChecked(bool(p.default)); w.toggled.connect(lambda *_: on_change())
        return w, (lambda: w.isChecked()), (lambda v: w.setChecked(bool(v)))
    if p.kind == "color":
        w = QPushButton(); state = {"c": tuple(p.default)}
        def paint(): w.setText("  %d, %d, %d  " % state["c"]); w.setStyleSheet("background: rgb(%d,%d,%d); color:%s; border:1px solid #444;" % (state["c"] + ("#000" if sum(state["c"]) > 380 else "#fff",)))
        def pick():
            c = QColorDialog.getColor(QColor(*state["c"]), w, "Colour")
            if c.isValid(): state["c"] = (c.red(), c.green(), c.blue()); paint(); on_change()
        w.clicked.connect(lambda *_: pick()); paint()
        def setv(v): state["c"] = tuple(int(x) for x in v); paint()
        return w, (lambda: tuple(state["c"])), setv
    if p.kind == "file":
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); e = QLineEdit(); b = QPushButton("...")
        b.setFixedWidth(34)
        def pick():
            f, _ = QFileDialog.getOpenFileName(w, "Picture", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
            if f: e.setText(f)
        b.clicked.connect(lambda *_: pick()); e.textChanged.connect(lambda *_: on_change()); l.addWidget(e, 1); l.addWidget(b)
        return w, (lambda: e.text().strip()), (lambda v: e.setText(str(v or "")))
    raise ValueError(p.kind)

def _apply_to_host(host, res, tool):
    """put a result into the main program exactly like its own tools do (undo step, caches, redraw)"""
    kind = res["kind"]; hub = G["HUB"]
    if kind == "face":
        faces = res.get("faces") or []
        if not faces: raise RuntimeError("No face to use.")
        fi = faces[0]; host.finfo = fi; host.face = G["fi_face_tuple"](fi); host.hair_mask = None
        if host.rgb is host.orig: host.fi_orig = fi
        host._invalidate_cache(); host.update_quality(); host.banner.done("Main face set"); return
    host.push()
    if kind == "alpha":
        host.alpha = Image.fromarray((np.clip(res["alpha"], 0, 1) * 255).astype(np.uint8)); host._invalidate_cache()
        if res.get("bgcolor"): host.set_bg(res["bgcolor"])
        else: host.after_change()
    elif kind == "flat":
        host.set_rgb(res["rgb"]); host.alpha = None; host.after_change()
    elif kind == "upscale":
        old = host.rgb.size; big = Image.fromarray(res["rgb"]); k = big.width / float(old[0])
        if host.alpha is not None: host.alpha = host.alpha.resize(big.size, Image.BICUBIC)
        host.set_rgb(big)
        if host.finfo is not None: host.finfo = G["_scale_fi"](host.finfo, k); host.face = G["fi_face_tuple"](host.finfo)
        if host.box is not None: host.box = [host.box[0] * k, host.box[1] * k, host.box[2] * k, host.box[3]]; host.base_h *= k
        host.hair_mask = None; host.after_change()
    else:
        host.set_rgb(res["rgb"]); host.after_change()
    host._mark_dirty(); host.banner.done("AI Studio: " + tool.title)

# =====================================================================
#  THE WINDOW
# =====================================================================
class AIStudioDialog(QDialog):
    def __init__(self, host, tool_key=None):
        super().__init__(host); self.host = host
        self.setWindowTitle("AI Studio  -  Ripon Computer"); self.resize(1280, 820)
        self.cfg = G["SETTINGS"].setdefault("ais", {})
        self.src = np.array(host.rgb.convert("RGB")); self.result = None; self.getters = {}; self.setters = {}; self.pw = {}
        self.runner = G["TaskRunner"](self); self.runner.progress.connect(self._on_progress)
        MGR.reset(); self.busy = False; self._building = True
        self.canvas = StudioCanvas(); self.canvas.set_before(self.src)
        self._build()
        i = self.tool_box.findData(tool_key or self.cfg.get("tool", "upscale")); self.tool_box.setCurrentIndex(max(0, i)); self._tool_changed()
        self._building = False; self.canvas.fit_view(); self.refresh_status()

    # ---------------- layout
    def _build(self):
        L = QHBoxLayout(self)
        left = QVBoxLayout(); left.addWidget(self.canvas, 1)
        bar = QHBoxLayout(); self.vb = {}
        for key, text in (("before", "◀ Before"), ("split", "◀ Split ▶"), ("after", "After ▶")):
            b = QPushButton(text); b.setCheckable(True); b.clicked.connect(lambda _=False, k=key: self._set_view(k)); self.vb[key] = b; bar.addWidget(b)
        bar.addSpacing(10)
        for text, fn in (("Fit", self.canvas.fit_view), ("100%", self.canvas.actual_size)):
            b = QPushButton(text); b.clicked.connect(lambda _=False, f=fn: f()); bar.addWidget(b)
        bar.addStretch(1); bar.addWidget(QLabel("Wheel = zoom   Right-drag = move   Drag line = compare"))
        left.addLayout(bar)
        self.paint_row = QWidget(); pr = QHBoxLayout(self.paint_row); pr.setContentsMargins(0, 0, 0, 0)
        self.b_paint = QPushButton("🖌 Paint"); self.b_paint.setCheckable(True); self.b_paint.setChecked(True); self.b_erase = QPushButton("Erase"); self.b_erase.setCheckable(True)
        self.b_paint.clicked.connect(lambda: (self.b_erase.setChecked(False), setattr(self.canvas, "erase", False)))
        self.b_erase.clicked.connect(lambda: (self.b_paint.setChecked(not self.b_erase.isChecked()), setattr(self.canvas, "erase", self.b_erase.isChecked())))
        self.s_brush = QSlider(Qt.Horizontal); self.s_brush.setRange(4, 400); self.s_brush.setValue(40); self.s_brush.valueChanged.connect(lambda v: setattr(self.canvas, "brush", v))
        self.canvas.brush = 40
        for w in (self.b_paint, self.b_erase, QLabel("Brush"), self.s_brush): pr.addWidget(w, 1 if w is self.s_brush else 0)
        for text, fn in (("Undo mark", self.canvas.undo_mask), ("Clear marks", self.canvas.clear_mask)):
            b = QPushButton(text); b.clicked.connect(lambda _=False, f=fn: f()); pr.addWidget(b)
        left.addWidget(self.paint_row)
        L.addLayout(left, 1)
        right = QWidget(); right.setFixedWidth(420); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        # tool
        g = QGroupBox("TOOL"); gl = QVBoxLayout(g); self.tool_box = QComboBox()
        for t in TOOLS: self.tool_box.addItem(t.title, t.key)
        self.tool_box.currentIndexChanged.connect(self._tool_changed)
        self.blurb = QLabel(); self.blurb.setWordWrap(True); self.blurb.setStyleSheet("color:#455a64;")
        gl.addWidget(self.tool_box); gl.addWidget(self.blurb); rl.addWidget(g)
        # mode
        g = QGroupBox("MODE  /  QUALITY  /  DEVICE"); gl = QVBoxLayout(g)
        r1 = QHBoxLayout(); self.r_auto = QRadioButton("Auto"); self.r_man = QRadioButton("Manual"); grp = QButtonGroup(self); grp.addButton(self.r_auto); grp.addButton(self.r_man)
        (self.r_man if self.cfg.get("mode") == "manual" else self.r_auto).setChecked(True); self.r_auto.toggled.connect(self._mode_changed)
        r1.addWidget(self.r_auto); r1.addWidget(self.r_man); r1.addStretch(1); gl.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Quality / speed")); self.q_box = QComboBox()
        for k, lab in (("fast", "Fast  (quickest)"), ("balanced", "Balanced"), ("best", "Best  (slowest)")): self.q_box.addItem(lab, k)
        self.q_box.setCurrentIndex(max(0, self.q_box.findData(self.cfg.get("quality", "balanced")))); self.q_box.currentIndexChanged.connect(self._quality_changed); r2.addWidget(self.q_box, 1); gl.addLayout(r2)
        r3 = QHBoxLayout(); r3.addWidget(QLabel("Device")); self.dev_box = QComboBox(); self.dev_box.addItem("Auto  (GPU, CPU if needed)", "auto"); self.dev_box.addItem("CPU only", "cpu")
        self.dev_box.setCurrentIndex(max(0, self.dev_box.findData(self.cfg.get("device", "auto")))); self.dev_box.currentIndexChanged.connect(lambda *_: MGR.reset()); r3.addWidget(self.dev_box, 1); gl.addLayout(r3)
        self.gpu_lbl = QLabel(); self.gpu_lbl.setStyleSheet("color:#455a64;font-size:11px;"); gl.addWidget(self.gpu_lbl); rl.addWidget(g)
        # settings
        self.set_group = QGroupBox("SETTINGS"); self.set_lay = QVBoxLayout(self.set_group)
        self.auto_note = QLabel("Auto chooses the settings from your photo and tells you what it chose."); self.auto_note.setWordWrap(True); self.auto_note.setStyleSheet("color:#455a64;")
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); inner = QWidget(); self.form = QVBoxLayout(inner); sc.setWidget(inner)
        self.set_lay.addWidget(self.auto_note); self.set_lay.addWidget(sc, 1); rl.addWidget(self.set_group, 1)
        # models
        g = QGroupBox("MODELS"); gl = QVBoxLayout(g); self.model_lbl = QLabel(); self.model_lbl.setWordWrap(True)
        self.b_dl = QPushButton("⬇ Download needed models"); self.b_dl.clicked.connect(self._download); gl.addWidget(self.model_lbl); gl.addWidget(self.b_dl); rl.addWidget(g)
        # run
        self.b_run = QPushButton("▶  Run / Preview"); self.b_run.setMinimumHeight(40); self.b_run.setStyleSheet("QPushButton{background:#2e7d32;color:white;font-weight:bold;}")
        self.b_cancel = QPushButton("✖ Cancel"); self.b_cancel.setMinimumHeight(40); self.b_cancel.setEnabled(False)
        self.b_run.clicked.connect(self._run); self.b_cancel.clicked.connect(self._cancel)
        rr = QHBoxLayout(); rr.addWidget(self.b_run, 2); rr.addWidget(self.b_cancel, 1); rl.addLayout(rr)
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setValue(0); self.bar.setFixedHeight(16)
        self.msg = QLabel("Ready"); self.msg.setWordWrap(True); rl.addWidget(self.bar); rl.addWidget(self.msg)
        self.info = QLabel(""); self.info.setWordWrap(True); self.info.setStyleSheet("background:#eef4fa;border-radius:5px;padding:5px;"); rl.addWidget(self.info)
        ar = QHBoxLayout(); self.b_apply = QPushButton("✔ Apply to photo"); self.b_apply.setMinimumHeight(36); self.b_apply.setEnabled(False)
        self.b_apply.setStyleSheet("QPushButton{background:#1565c0;color:white;font-weight:bold;}"); self.b_apply.clicked.connect(self._apply)
        self.b_save = QPushButton("💾 Save..."); self.b_save.setEnabled(False); self.b_save.clicked.connect(self._save); self.b_close = QPushButton("Close"); self.b_close.clicked.connect(self.close)
        for b in (self.b_apply, self.b_save, self.b_close): ar.addWidget(b)
        rl.addLayout(ar); L.addWidget(right)

    # ---------------- state helpers
    @property
    def tool(self): return TOOL_BY_KEY[self.tool_box.currentData()]
    def quality(self): return self.q_box.currentData()
    def is_auto(self): return self.r_auto.isChecked()
    def device(self): return self.dev_box.currentData()

    def _set_view(self, k):
        self.canvas.set_mode(k)
        for kk, b in self.vb.items(): b.setChecked(kk == self.canvas.mode)

    def _tool_changed(self, *_):
        self._building = True
        t = self.tool; self.cfg["tool"] = t.key; self.blurb.setText(t.blurb)
        while self.form.count():
            it = self.form.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self.getters.clear(); self.setters.clear(); self.pw.clear()
        for p in t.params:
            w, g, s = _make_param_widget(p, self.refresh_status); self.getters[p.key] = g; self.setters[p.key] = s
            if p.kind == "bool": box = w
            else:
                box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 2, 0, 2); lab = QLabel(p.label); lab.setWordWrap(True); bl.addWidget(lab); bl.addWidget(w)
            if p.tip: box.setToolTip(p.tip)
            self.form.addWidget(box); self.pw[p.key] = (box, p)
        self.form.addStretch(1)
        self._load_defaults(); self._mode_changed(); self._building = False; self.canvas.paint = (t.key == "remove"); self.paint_row.setVisible(t.key == "remove")
        self.result = None; self.canvas.set_after(None); self.canvas.extra = None; self.b_apply.setEnabled(False); self.b_save.setEnabled(False); self.info.setText("")
        self.b_apply.setText("✔ Use as main face" if t.key == "faces" else "✔ Apply to photo"); self._set_view("before"); self.refresh_status()

    def _load_defaults(self):
        for k, v in self.tool.defaults(self.quality()).items():
            if k in self.setters: self.setters[k](v)

    def _quality_changed(self, *_):
        self.cfg["quality"] = self.quality(); self._load_defaults(); self.refresh_status()

    def _mode_changed(self, *_):
        auto = self.is_auto(); self.cfg["mode"] = "auto" if auto else "manual"; self.auto_note.setVisible(auto)
        for k, (box, p) in self.pw.items(): box.setVisible(p.always or not auto)
        self.refresh_status()

    def _groups(self):
        t = self.tool; st = {k: g() for k, g in self.getters.items()}
        if self.is_auto(): st = t.defaults(self.quality()) | {k: v for k, v in st.items() if k in self.pw and self.pw[k][1].always}
        return t.needs(st, self.is_auto(), self.quality()), t.wants(st, self.is_auto(), self.quality())

    def refresh_status(self):
        if not hasattr(self, "model_lbl") or getattr(self, "_building", False): return
        need, want = self._groups(); reg = G["MODEL_REGISTRY"]; lines = []; self.missing = []
        for grp in need:
            ok = [k for k in grp if installed(k)]
            if ok: lines.append("✔ " + reg[ok[0]]["label"])
            else: lines.append("⬇ needs: " + reg[grp[0]]["label"] + "  (" + reg[grp[0]]["size"] + ")"); self.missing.append(grp[0])
        for k in want:
            if installed(k): lines.append("✔ " + reg[k]["label"] + "  (optional)")
            else: lines.append("• optional: " + reg[k]["label"] + "  (" + reg[k]["size"] + ")" + ("" if G["HUB"].lib_available(k) else "  - package missing")); self.opt_missing = True
        self.model_lbl.setText("\n".join(lines) if lines else "No model needed for this tool.")
        self.opt = [k for k in want if not installed(k) and G["HUB"].lib_available(k)]
        self.b_dl.setVisible(bool(self.missing or self.opt)); self.b_dl.setText("⬇ Download needed models" if self.missing else "⬇ Download optional models")
        vi = vram_info(); hub = G["HUB"]; d = hub.device_info()
        gpu = "GPU: " + (vi["name"] + "  %d / %d MB free" % (vi["free"], vi["total"]) if vi else (d["gpu"] or "not detected"))
        ep = "CUDA" if d.get("ort_gpu") else ("DirectML" if d.get("ort_dml") else "CPU only (install onnxruntime-gpu)")
        self.gpu_lbl.setText(gpu + "\nONNX runtime: " + ep)

    # ---------------- jobs
    def _busy(self, on):
        self.busy = on; self.b_run.setEnabled(not on); self.b_cancel.setEnabled(on); self.b_dl.setEnabled(not on)
        for w in (self.tool_box, self.r_auto, self.r_man, self.q_box, self.dev_box): w.setEnabled(not on)

    def _on_progress(self, text, i, n):
        if text: self.msg.setText(text)
        if n < 0: self.bar.setRange(0, 1000); self.bar.setValue(int(1000 * i / float(-n)))
        elif n > 0 and i >= 0: self.bar.setRange(0, 1000); self.bar.setValue(int(1000 * i / float(n)))
        else: self.bar.setRange(0, 0)

    def _cancel(self):
        self.runner.cancel(); self.msg.setText("Cancelling...")

    def _download(self):
        keys = list(dict.fromkeys(self.missing + (self.opt if not self.missing else [])))
        if keys: self._download_keys(keys, None)

    def _download_keys(self, keys, then):
        hub = G["HUB"]
        def fn(ctx):
            for k in keys:
                ctx.check(); ctx.msg("Downloading %s ..." % G["MODEL_REGISTRY"][k]["label"])
                hub.download(k, lambda m, f: ctx.frac(m, f) if f is not None else ctx.msg(m))
            return True
        def ok(_):
            self._busy(False); self.msg.setText("Download finished"); self.bar.setRange(0, 1000); self.bar.setValue(1000); self.refresh_status()
            if then: QTimer.singleShot(0, then)
        def fail(m, e):
            self._busy(False); self.bar.setRange(0, 1000); self.bar.setValue(0)
            if m == "Cancelled": self.msg.setText("Download cancelled")
            else: self.msg.setText("Download failed"); QMessageBox.warning(self, "AI Studio", "Download failed:\n%s\n\nCheck the internet connection." % m[:500])
            self.refresh_status()
        self._busy(True); self.bar.setRange(0, 0)
        if not self.runner.submit(fn, ok, fail): self._busy(False)

    def _run(self):
        if self.busy: return
        if self.host.runner.busy: return QMessageBox.information(self, "AI Studio", "Please wait - the main program is busy.")
        t = self.tool; auto = self.is_auto(); st = {k: g() for k, g in self.getters.items()}
        need, _w = self._groups()
        miss = [grp[0] for grp in need if not any(installed(k) for k in grp)]
        if miss:
            lab = ", ".join(G["MODEL_REGISTRY"][k]["label"] for k in miss)
            if QMessageBox.question(self, "AI Studio", "Needed model is not downloaded:\n%s\n\nDownload it now and run?" % lab) == QMessageBox.Yes:
                self._download_keys(miss, self._run)
            return
        mask = None
        if t.key == "remove":
            self.canvas.ensure_mask(); mask = self.canvas.mask.copy() if self.canvas.has_mask() else None
        env = Env(self.device(), self.quality(), mask); src = self.src
        self.cfg["device"] = self.device(); MGR.take_notes(); t0 = time.time(); self.info.setText("")
        def fn(ctx): return run_tool(ctx, t, env, src, st, auto)
        def ok(res):
            self._busy(False); self.result = res; self.bar.setRange(0, 1000); self.bar.setValue(1000)
            notes = MGR.take_notes(); sec = time.time() - t0
            txt = res["info"] + ("\nAuto: " + res["why"] if res.get("why") else "") + ("\n(same settings as before - reused the earlier result)" if res.get("cached") else "") + "\nTime: %.1f s" % sec
            if notes: txt += "\n⚠ " + "\n⚠ ".join(notes)
            self.info.setText(txt); self.msg.setText("Done"); self.b_apply.setEnabled(True); self.b_save.setEnabled(t.key != "faces")
            prev = res.get("preview", res["rgb"]); self.canvas.extra = res.get("mask") if t.key == "remove" else None
            self.canvas.set_after(prev, "split"); self._set_view("split"); self.canvas.fit_view()
            if not auto:
                for k, v in res.get("settings", {}).items():
                    pass
        def fail(m, e):
            self._busy(False); self.bar.setRange(0, 1000); self.bar.setValue(0); notes = MGR.take_notes()
            if m == "Cancelled": self.msg.setText("Cancelled"); return
            if isinstance(e, NeedModel):
                self.msg.setText("Model missing"); self.refresh_status()
                if QMessageBox.question(self, "AI Studio", "%s\n\nDownload now?" % m) == QMessageBox.Yes: self._download_keys(e.keys, self._run)
                return
            self.msg.setText("Failed"); self.info.setText("✖ " + m[:400] + ("\n⚠ " + "\n⚠ ".join(notes) if notes else ""))
        self._busy(True); self.bar.setRange(0, 0); self.msg.setText("Starting...")
        if not self.runner.submit(fn, ok, fail): self._busy(False)

    def _apply(self):
        r = self.result
        if r is None: return
        try:
            _apply_to_host(self.host, r, self.tool)
        except Exception as e:
            return QMessageBox.warning(self, "AI Studio", "Could not apply:\n%s" % e)
        if r["kind"] == "face": self.msg.setText("Main face set"); return
        self.src = np.array(self.host.rgb.convert("RGB")); self.canvas.set_before(self.src); self.result = None
        self.b_apply.setEnabled(False); self.b_save.setEnabled(False); self._set_view("before"); self.canvas.fit_view()
        self.msg.setText("Applied - the picture on the left is now the new photo (use Undo in the main window to go back)")

    def _save(self):
        r = self.result
        if r is None: return
        alpha = r.get("alpha"); path, _ = QFileDialog.getSaveFileName(self, "Save result", os.path.join(os.path.expanduser("~"), "Pictures", "ai_result." + ("png" if alpha is not None else "jpg")),
                                                                    "PNG (*.png);;JPEG (*.jpg)")
        if not path: return
        try:
            im = Image.fromarray(r["rgb"])
            if alpha is not None:
                im = im.convert("RGBA"); im.putalpha(Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)))
                if not path.lower().endswith(".png"): path += ".png"
            im.save(path, quality=95); self.msg.setText("Saved: " + path)
        except Exception as e:
            QMessageBox.warning(self, "AI Studio", "Could not save:\n%s" % e)

    def closeEvent(self, e):
        if self.busy: self.runner.cancel()
        G["save_settings"](); e.accept()

def open_studio(host, tool_key=None):
    d = AIStudioDialog(host, tool_key); d.exec()
