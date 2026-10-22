# -*- coding: utf-8 -*-
"""
RIPON CORE  -  pure-logic engine for the Layout Studio / Document Studio upgrade.
No Qt in this file, so every function here can be (and is) tested headless.

  units & paper ........ mm / inch / px, A3 A4 A5 A6 Letter Legal Custom, portrait / landscape
  layout engines ....... grid packing, Bank Photo Duet, ID card front+back, passport sheet
  project model ........ Project / Page / Layer, save/open (.rproj zip), relink, autosave
  rendering ............ exact-size page render with DPI metadata, JPG / PNG / multi-page PDF
  documents ............ edge detection, perspective warp, deskew
  restoration .......... shadow / lighting / noise / whitening / faint text / B&W (never invents text)
  downloader ........... resumable, verified, cancellable, clear errors (404 etc.)
"""
import os, io, re, json, math, time, uuid, shutil, hashlib, zipfile, threading, urllib.request, urllib.error
import numpy as np
import cv2
from PIL import Image, ImageDraw

# =====================================================================
#  UNITS & PAPER
# =====================================================================
MM_PER_INCH = 25.4

def mm2px(mm, dpi=300):  return int(round(float(mm) / MM_PER_INCH * dpi))
def px2mm(px, dpi=300):  return float(px) / dpi * MM_PER_INCH
def in2mm(v):            return float(v) * MM_PER_INCH
def mm2in(v):            return float(v) / MM_PER_INCH

PAPERS_MM = {"A3": (297.0, 420.0), "A4": (210.0, 297.0), "A5": (148.0, 210.0), "A6": (105.0, 148.0),
             "Letter": (215.9, 279.4), "Legal": (215.9, 355.6)}
PAPER_NAMES = list(PAPERS_MM) + ["Custom"]

def page_size_mm(name="A4", orientation="Portrait", custom_mm=None):
    """(width, height) in mm for a named paper. Landscape swaps so width >= height."""
    if name == "Custom" or name not in PAPERS_MM:
        w, h = custom_mm if custom_mm else PAPERS_MM["A4"]
    else:
        w, h = PAPERS_MM[name]
    w, h = float(w), float(h)
    if orientation.lower().startswith("l"):
        w, h = max(w, h), min(w, h)
    else:
        w, h = min(w, h), max(w, h)
    return w, h

# Bank Photo Duet default: the WHOLE duet (both photos together) = 1.9 x 1.5 inch (width x height) = 570 x 450 px at 300 DPI
BANK_PHOTO_IN = (1.9, 1.5)
def bank_photo_px(dpi=300, w_in=BANK_PHOTO_IN[0], h_in=BANK_PHOTO_IN[1]):
    return int(round(w_in * dpi)), int(round(h_in * dpi))

# =====================================================================
#  LAYOUT ENGINES  (all geometry in mm, top-left origin, so print size is exact)
# =====================================================================
def _count(avail, cell, gap):
    if cell <= 0 or avail < cell - 1e-9: return 0
    return int(math.floor((avail + gap + 1e-9) / (cell + gap)))

def pack_grid(cell_w, cell_h, page_w, page_h, margin=5.0, gap=2.0, allow_rotate=True,
              count=None, align="center", gap_y=None):
    """Place as many (cell_w x cell_h) cells as fit. Returns dict(cols, rows, rot, items=[(x,y,w,h)], capacity).
    If allow_rotate, tries the cell turned 90 degrees too and keeps the better fit.
    margin may be a number or (left, top, right, bottom)."""
    if isinstance(margin, (int, float)): ml = mt = mr = mb = float(margin)
    else: ml, mt, mr, mb = [float(v) for v in margin]
    gx = float(gap); gy = float(gap if gap_y is None else gap_y)
    aw, ah = page_w - ml - mr, page_h - mt - mb
    best = None
    for rot in ((False, True) if allow_rotate and abs(cell_w - cell_h) > 1e-6 else (False,)):
        cw, ch = (cell_h, cell_w) if rot else (cell_w, cell_h)
        cols, rows = _count(aw, cw, gx), _count(ah, ch, gy)
        if best is None or cols * rows > best["cols"] * best["rows"]:
            best = dict(cols=cols, rows=rows, rot=rot, cw=cw, ch=ch)
    cols, rows, cw, ch = best["cols"], best["rows"], best["cw"], best["ch"]
    cap = cols * rows
    n = cap if count is None else min(int(count), cap)
    used_rows = int(math.ceil(n / cols)) if cols else 0
    used_cols = min(cols, n) if used_rows <= 1 else cols
    bw = used_cols * cw + max(0, used_cols - 1) * gx
    bh = used_rows * ch + max(0, used_rows - 1) * gy
    if align == "center": x0, y0 = ml + (aw - bw) / 2.0, mt + (ah - bh) / 2.0
    elif align == "top center": x0, y0 = ml + (aw - bw) / 2.0, mt
    else: x0, y0 = ml, mt
    items = []
    for i in range(n):
        r, c = divmod(i, cols)
        items.append((x0 + c * (cw + gx), y0 + r * (ch + gy), cw, ch))
    return dict(cols=cols, rows=rows, rot=best["rot"], capacity=cap, items=items, cell=(cw, ch))

def duet_layout(photo_w, photo_h, page_w, page_h, gap=2.0, margin=5.0, between=0.0, copies=None,
                arrangement="side", allow_rotate=False, align="center", gap_y=None):
    """Bank Photo Duet: two photos per duet, repeated as many times as fit.
      arrangement 'side'  = photos side by side (default),  'stack' = one above the other,
                  'auto'  = whichever of the two fits more duets on the page.
      between = distance in mm between the two photos of one duet; gap = distance between duets.
    Photos always stay upright unless allow_rotate=True (then rot=True and every image must be turned 90 deg when drawn).
    Returns dict(duets=[[rect_a, rect_b], ...], arrangement, rot, capacity, cols, rows); rect = (x, y, w, h) in mm."""
    best = None
    for arr in (("side", "stack") if arrangement == "auto" else (arrangement,)):
        uw, uh = (2 * photo_w + between, photo_h) if arr == "side" else (photo_w, 2 * photo_h + between)
        g = pack_grid(uw, uh, page_w, page_h, margin, gap, allow_rotate, copies, align, gap_y)
        if best is None or g["capacity"] > best[1]["capacity"]: best = (arr, g)
    arr, g = best; duets = []
    for (x, y, w, h) in g["items"]:
        if not g["rot"]:
            if arr == "side": duets.append([(x, y, photo_w, photo_h), (x + photo_w + between, y, photo_w, photo_h)])
            else:             duets.append([(x, y, photo_w, photo_h), (x, y + photo_h + between, photo_w, photo_h)])
        else:                 # unit turned 90 deg
            if arr == "side": duets.append([(x, y, photo_h, photo_w), (x, y + photo_w + between, photo_h, photo_w)])
            else:             duets.append([(x, y, photo_h, photo_w), (x + photo_h + between, y, photo_h, photo_w)])
    return dict(duets=duets, arrangement=arr, rot=g["rot"], capacity=g["capacity"], cols=g["cols"], rows=g["rows"])

def duet_photo_size(duet_w, duet_h, between=0.0, arrangement="side"):
    """Size (mm) of ONE photo when the two photos TOGETHER (incl. the gap between them) must fill duet_w x duet_h.
       side  = two photos left/right  -> each ((duet_w - between)/2) x duet_h
       stack = two photos top/bottom  -> each duet_w x ((duet_h - between)/2)"""
    if arrangement == "stack": pw, ph = duet_w, (duet_h - between) / 2.0
    else:                      pw, ph = (duet_w - between) / 2.0, duet_h
    if pw <= 0 or ph <= 0: raise ValueError("The gap between the 2 photos is too big for this duet size.")
    return pw, ph

def duet_layout_total(duet_w, duet_h, page_w, page_h, gap=2.0, margin=5.0, between=0.0, copies=None,
                      arrangement="side", allow_rotate=False, align="center", gap_y=None):
    """Like duet_layout, but duet_w x duet_h is the size of the WHOLE duet (photo A + gap + photo B), e.g. 1.9 x 1.5 in.
    Returns the same dict as duet_layout plus photo=(w, h) mm of one photo for the chosen arrangement."""
    best = None
    for arr in (("side", "stack") if arrangement == "auto" else (arrangement,)):
        pw, ph = duet_photo_size(duet_w, duet_h, between, arr)
        d = duet_layout(pw, ph, page_w, page_h, gap, margin, between, copies, arr, allow_rotate, align, gap_y)
        if best is None or d["capacity"] > best["capacity"]: d["photo"] = (pw, ph); best = d
    return best

def id_card_layout(card_w, card_h, page_w, page_h, gap=10.0, margin=15.0, arrangement="stack",
                   cards=1, card_gap=8.0, allow_rotate=False, align="center"):
    """Front + back on one page. arrangement 'stack' = back below front, 'side' = back beside front.
    `cards` = how many front/back pairs (as many as fit). Returns dict(pairs=[[front_rect, back_rect], ...], capacity, rot).
    rot=True: each card is turned 90 deg (images must be rotated 90 deg when drawn)."""
    if arrangement == "side": uw, uh = 2 * card_w + gap, card_h
    else: uw, uh = card_w, 2 * card_h + gap
    g = pack_grid(uw, uh, page_w, page_h, margin, card_gap, allow_rotate, cards, align)
    pairs = []
    for (x, y, w, h) in g["items"]:
        if g["rot"]:
            if arrangement == "side": f, b = (x, y, card_h, card_w), (x, y + card_w + gap, card_h, card_w)
            else:                     f, b = (x, y, card_h, card_w), (x + card_h + gap, y, card_h, card_w)
        elif arrangement == "side":   f, b = (x, y, card_w, card_h), (x + card_w + gap, y, card_w, card_h)
        else:                         f, b = (x, y, card_w, card_h), (x, y + card_h + gap, card_w, card_h)
        pairs.append([f, b])
    return dict(pairs=pairs, capacity=g["capacity"], rot=g["rot"])

def passport_sheet(photo_w, photo_h, page_w, page_h, copies, gap=2.0, margin=5.0, allow_rotate=True, align="top left"):
    """Passport / any photo sheet. Returns list of pages; each page = list of (x,y,w,h). Extra copies spill to new pages."""
    g = pack_grid(photo_w, photo_h, page_w, page_h, margin, gap, allow_rotate, None, align)
    cap = g["capacity"]
    if cap <= 0: return dict(pages=[], capacity=0, fits=False)
    pages, left = [], int(copies)
    while left > 0:
        n = min(cap, left)
        pages.append(pack_grid(photo_w, photo_h, page_w, page_h, margin, gap, allow_rotate, n, align)["items"])
        left -= n
    return dict(pages=pages, capacity=cap, fits=True)

def rect_bounds_rotated(x, y, w, h, rot_deg):
    """axis-aligned bounds (x0,y0,x1,y1) of a w x h rect at (x,y) turned rot_deg about its centre"""
    cx, cy = x + w / 2.0, y + h / 2.0
    a = math.radians(rot_deg); ca, sa = abs(math.cos(a)), abs(math.sin(a))
    bw, bh = w * ca + h * sa, w * sa + h * ca
    return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)

def out_of_page(layers, page_w, page_h, tol=0.01):
    """layers: iterable of objects with x_mm,y_mm,w_mm,h_mm,rot,visible,name. Returns names of layers partly/fully outside the page."""
    bad = []
    for L in layers:
        if not getattr(L, "visible", True): continue
        x0, y0, x1, y1 = rect_bounds_rotated(L.x_mm, L.y_mm, L.w_mm, L.h_mm, L.rot)
        if x0 < -tol or y0 < -tol or x1 > page_w + tol or y1 > page_h + tol: bad.append(L.name)
    return bad

# =====================================================================
#  PROJECT MODEL  (pages, layers, save / open / relink / autosave)
# =====================================================================
def _id(): return uuid.uuid4().hex[:10]

class Layer:
    """One image on a page. Position/size in mm (physical), rotation in degrees about the centre.
    The original file is never modified: `path` points at it, `asset` is the embedded copy inside the project."""
    FIELDS = ("id", "name", "path", "asset", "x_mm", "y_mm", "w_mm", "h_mm", "rot", "opacity", "visible",
              "locked", "group", "crop", "flip_h", "flip_v", "adjust", "border_mm")
    def __init__(self, name="Image", path="", x_mm=0.0, y_mm=0.0, w_mm=50.0, h_mm=50.0, rot=0.0):
        self.id = _id(); self.name = name; self.path = path; self.asset = ""
        self.x_mm, self.y_mm, self.w_mm, self.h_mm, self.rot = float(x_mm), float(y_mm), float(w_mm), float(h_mm), float(rot)
        self.opacity = 1.0; self.visible = True; self.locked = False; self.group = ""
        self.crop = None            # (l,t,r,b) fractions of the source image, or None
        self.flip_h = False; self.flip_v = False
        self.adjust = {}            # brightness/contrast/gray ... non-destructive
        self.border_mm = 0.0
    def to_dict(self): return {k: getattr(self, k) for k in self.FIELDS}
    @classmethod
    def from_dict(cls, d):
        L = cls()
        for k in cls.FIELDS:
            if k in d: setattr(L, k, d[k])
        if isinstance(L.crop, list): L.crop = tuple(L.crop)
        return L
    def duplicate(self, dx=5.0, dy=5.0):
        d = self.to_dict(); d["id"] = _id(); d["name"] = self.name + " copy"; d["x_mm"] += dx; d["y_mm"] += dy
        return Layer.from_dict(d)

class Page:
    def __init__(self, name="Page 1", w_mm=210.0, h_mm=297.0, paper="A4", orientation="Portrait"):
        self.id = _id(); self.name = name; self.w_mm = float(w_mm); self.h_mm = float(h_mm)
        self.paper = paper; self.orientation = orientation
        self.bg = (255, 255, 255); self.margin_mm = 5.0
        self.guides_h = []; self.guides_v = []        # mm positions
        self.layers = []                              # bottom -> top
    def to_dict(self):
        return dict(id=self.id, name=self.name, w_mm=self.w_mm, h_mm=self.h_mm, paper=self.paper, orientation=self.orientation,
                    bg=list(self.bg), margin_mm=self.margin_mm, guides_h=self.guides_h, guides_v=self.guides_v,
                    layers=[L.to_dict() for L in self.layers])
    @classmethod
    def from_dict(cls, d):
        p = cls(d.get("name", "Page"), d.get("w_mm", 210), d.get("h_mm", 297), d.get("paper", "A4"), d.get("orientation", "Portrait"))
        p.id = d.get("id", p.id); p.bg = tuple(d.get("bg", (255, 255, 255))); p.margin_mm = d.get("margin_mm", 5.0)
        p.guides_h = list(d.get("guides_h", [])); p.guides_v = list(d.get("guides_v", []))
        p.layers = [Layer.from_dict(x) for x in d.get("layers", [])]
        return p

class Project:
    VERSION = 1
    def __init__(self, name="Untitled"):
        self.name = name; self.dpi = 300; self.pages = [Page()]; self.meta = {}
    def to_dict(self):
        return dict(version=self.VERSION, name=self.name, dpi=self.dpi, meta=self.meta, pages=[p.to_dict() for p in self.pages])
    @classmethod
    def from_dict(cls, d):
        pr = cls(d.get("name", "Untitled")); pr.dpi = d.get("dpi", 300); pr.meta = d.get("meta", {})
        pr.pages = [Page.from_dict(x) for x in d.get("pages", [])] or [Page()]
        return pr

def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def save_project(proj, path, embed=True):
    """Save as a .rproj zip: project.json (+ assets/<sha1>.<ext> copies of the originals when embed=True).
    Written to a temp file first, then renamed, so a crash never leaves a half-written project."""
    d = proj.to_dict()
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        done = {}
        if embed:
            for pg in d["pages"]:
                for L in pg["layers"]:
                    src = L.get("path", "")
                    if src and os.path.isfile(src):
                        if src not in done:
                            name = "assets/%s%s" % (_sha1_file(src), os.path.splitext(src)[1].lower())
                            if name not in z.namelist(): z.write(src, name)
                            done[src] = name
                        L["asset"] = done[src]
        z.writestr("project.json", json.dumps(d, indent=1, ensure_ascii=False))
    os.replace(tmp, path)
    for pg, pd_ in zip(proj.pages, d["pages"]):
        for L, Ld in zip(pg.layers, pd_["layers"]): L.asset = Ld.get("asset", "")
    return path

def open_project(path, extract_dir=None):
    """Returns (project, missing_layers). Embedded assets are extracted to extract_dir (default: next to the project).
    A layer whose original path is gone but whose embedded copy exists is repointed at the copy automatically."""
    extract_dir = extract_dir or (os.path.splitext(path)[0] + "_assets")
    with zipfile.ZipFile(path) as z:
        d = json.loads(z.read("project.json").decode("utf-8"))
        proj = Project.from_dict(d)
        missing = []
        for pg in proj.pages:
            for L in pg.layers:
                if L.path and os.path.isfile(L.path): continue
                if L.asset and L.asset in z.namelist():
                    os.makedirs(extract_dir, exist_ok=True)
                    out = os.path.join(extract_dir, os.path.basename(L.asset))
                    if not os.path.isfile(out):
                        with open(out, "wb") as f: f.write(z.read(L.asset))
                    L.path = out
                else:
                    missing.append(L)
    return proj, missing

def relink(layers, search_dir):
    """Try to find missing originals by file name under search_dir. Returns how many were relinked."""
    index = {}
    for root, _, files in os.walk(search_dir):
        for f in files: index.setdefault(f.lower(), os.path.join(root, f))
    n = 0
    for L in layers:
        hit = index.get(os.path.basename(L.path).lower()) if L.path else None
        if hit: L.path = hit; n += 1
    return n

class Autosaver:
    """Writes the project (json only, small + fast) to a recovery file; `pending()` tells the next start if a crash left one."""
    def __init__(self, folder, name="layout_autosave.json"):
        os.makedirs(folder, exist_ok=True); self.path = os.path.join(folder, name)
    def save(self, proj):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(proj.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, self.path)
    def pending(self): return os.path.isfile(self.path)
    def load(self):
        with open(self.path, "r", encoding="utf-8") as f: return Project.from_dict(json.load(f))
    def clear(self):
        try: os.remove(self.path)
        except OSError: pass

# =====================================================================
#  RENDERING  (exact physical size + DPI metadata)
# =====================================================================
def _apply_adjust(im, adj):
    if not adj: return im
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    b, c = float(adj.get("brightness", 0)), float(adj.get("contrast", 0))
    if b or c: a = (a - 128.0) * (1.0 + c / 100.0) + 128.0 + b * 2.55
    a = np.clip(a, 0, 255).astype(np.uint8)
    out = Image.fromarray(a)
    if adj.get("gray"): out = out.convert("L").convert("RGB")
    return out

def prepare_layer_image(L, src_im, dpi):
    """source image -> RGBA at exactly the layer's physical size (w_mm x h_mm at dpi), crop/flip/adjust/opacity applied.
    The source is only ever read. Rotation is applied separately by the caller (about the layer centre)."""
    im = src_im
    if L.crop:
        W, H = im.size; l, t, r, b = L.crop
        im = im.crop((int(l * W), int(t * H), max(int(l * W) + 1, int(r * W)), max(int(t * H) + 1, int(b * H))))
    if L.flip_h: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if L.flip_v: im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = _apply_adjust(im, L.adjust)
    tw, th = max(1, mm2px(L.w_mm, dpi)), max(1, mm2px(L.h_mm, dpi))
    if im.size != (tw, th):
        # LANCZOS when shrinking, BICUBIC when enlarging (no fake detail either way)
        im = im.convert("RGBA").resize((tw, th), Image.LANCZOS if tw * th < im.size[0] * im.size[1] else Image.BICUBIC)
    im = im.convert("RGBA")
    if L.border_mm > 0:
        d = ImageDraw.Draw(im); bw = max(1, mm2px(L.border_mm, dpi))
        d.rectangle([0, 0, tw - 1, th - 1], outline=(0, 0, 0, 255), width=bw)
    if L.opacity < 1.0:
        al = im.getchannel("A").point(lambda v: int(v * L.opacity)); im.putalpha(al)
    return im

def render_page(page, dpi=300, loader=None, show_guides=False):
    """Composite one page to an RGB PIL image. Size = page mm * dpi exactly. loader(layer)->PIL.Image (default: open layer.path)."""
    W, H = mm2px(page.w_mm, dpi), mm2px(page.h_mm, dpi)
    canvas = Image.new("RGBA", (W, H), tuple(page.bg) + (255,))
    loader = loader or (lambda L: Image.open(L.path))
    for L in page.layers:
        if not L.visible: continue
        src = loader(L)
        if src is None: continue
        im = prepare_layer_image(L, src, dpi)
        if abs(L.rot) > 1e-6: im = im.rotate(-L.rot, resample=Image.BICUBIC, expand=True)   # clockwise like the canvas
        cx, cy = (L.x_mm + L.w_mm / 2.0) / MM_PER_INCH * dpi, (L.y_mm + L.h_mm / 2.0) / MM_PER_INCH * dpi
        px, py = int(round(cx - im.size[0] / 2.0)), int(round(cy - im.size[1] / 2.0))
        canvas.paste(im, (px, py), im)
    out = canvas.convert("RGB")
    if show_guides:
        d = ImageDraw.Draw(out)
        for g in page.guides_h: d.line([(0, mm2px(g, dpi)), (W, mm2px(g, dpi))], fill=(0, 200, 255), width=1)
        for g in page.guides_v: d.line([(mm2px(g, dpi), 0), (mm2px(g, dpi), H)], fill=(0, 200, 255), width=1)
    out.info["dpi"] = (dpi, dpi)
    return out

def save_image(im, path, dpi=300, quality=95):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"): im.convert("RGB").save(path, "JPEG", quality=quality, dpi=(dpi, dpi), subsampling=0)
    elif ext == ".png": im.save(path, "PNG", dpi=(dpi, dpi))
    else: raise ValueError("Unsupported image type: " + ext)
    return path

def save_pdf(pages_pil, path, dpi=300, quality=92):
    """Multi-page PDF, page order kept. PDF page size = pixel size / dpi, so a 2480x3508 px page at 300 dpi is exactly A4."""
    if not pages_pil: raise ValueError("No pages to export")
    rgb = [p.convert("RGB") for p in pages_pil]
    rgb[0].save(path, "PDF", resolution=float(dpi), save_all=True, append_images=rgb[1:], quality=quality)
    return path

def resample_to_dpi(im, src_dpi, target_dpi=300):
    """Optional: resize so the SAME physical size has target_dpi pixels. This changes pixel count, not real detail."""
    if src_dpi == target_dpi: return im
    k = target_dpi / float(src_dpi)
    return im.resize((max(1, int(round(im.size[0] * k))), max(1, int(round(im.size[1] * k)))),
                     Image.LANCZOS if k < 1 else Image.BICUBIC)

def placements_to_layers(rects, path, name="Photo", fit="cover"):
    """helper: convert layout rects into Layer objects (rotation 90 for turned cells is handled by caller)"""
    return [Layer("%s %d" % (name, i + 1), path, x, y, w, h) for i, (x, y, w, h) in enumerate(rects)]

def fit_image_to_box(im, w_px, h_px, mode="cover", bg=(255, 255, 255)):
    """cover = fill the box (centre crop), contain = whole image + bg bars, stretch = distort. Never touches the source."""
    im = im.convert("RGB")
    if mode == "stretch": return im.resize((w_px, h_px), Image.LANCZOS)
    sw, sh = im.size
    k = max(w_px / sw, h_px / sh) if mode == "cover" else min(w_px / sw, h_px / sh)
    nw, nh = max(1, int(round(sw * k))), max(1, int(round(sh * k)))
    r = im.resize((nw, nh), Image.LANCZOS if k < 1 else Image.BICUBIC)
    if mode == "cover":
        l, t = (nw - w_px) // 2, (nh - h_px) // 2
        return r.crop((l, t, l + w_px, t + h_px))
    out = Image.new("RGB", (w_px, h_px), bg); out.paste(r, ((w_px - nw) // 2, (h_px - nh) // 2)); return out

def render_rects(page_w_mm, page_h_mm, rects_images, dpi=300, bg=(255, 255, 255), fit="cover", border_mm=0.0, cut_marks=False):
    """Fast renderer for layouts: rects_images = [((x,y,w,h) mm, PIL image or None, rot90 bool)]. Used by Duet / ID / passport sheets."""
    W, H = mm2px(page_w_mm, dpi), mm2px(page_h_mm, dpi)
    page = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(page)
    for (x, y, w, h), im, rot in rects_images:
        x0, y0 = mm2px(x, dpi), mm2px(y, dpi)
        wp, hp = mm2px(w, dpi), mm2px(h, dpi)
        if im is not None:
            tile = fit_image_to_box(im.rotate(90, expand=True) if rot else im, wp, hp, fit, bg)
            page.paste(tile, (x0, y0))
        if border_mm > 0: d.rectangle([x0, y0, x0 + wp - 1, y0 + hp - 1], outline=(0, 0, 0), width=max(1, mm2px(border_mm, dpi)))
        if cut_marks:
            m = mm2px(2, dpi)
            for cx, cy in ((x0, y0), (x0 + wp, y0), (x0, y0 + hp), (x0 + wp, y0 + hp)):
                d.line([(cx - m, cy), (cx + m, cy)], fill=(120, 120, 120), width=1); d.line([(cx, cy - m), (cx, cy + m)], fill=(120, 120, 120), width=1)
    page.info["dpi"] = (dpi, dpi)
    return page

# =====================================================================
#  DOCUMENTS: edge detection, perspective warp, deskew
# =====================================================================
def order_quad(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # TL TR BR BL

def _quad_area(q): return abs(cv2.contourArea(np.array(q, dtype=np.float32).reshape(-1, 1, 2)))

def detect_document_quad(rgb, min_frac=0.12):
    """Find the 4 corners of the largest document/paper. Returns (quad[4x2 float32 TL,TR,BR,BL] in ORIGINAL pixels, confidence 0..1)
    or (None, 0.0). Two strategies: edges+contours, then bright-paper segmentation. Never guesses silently: low confidence is reported."""
    H, W = rgb.shape[:2]; k = 800.0 / max(H, W) if max(H, W) > 800 else 1.0
    sm = cv2.resize(rgb, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA) if k != 1.0 else rgb.copy()
    h, w = sm.shape[:2]; total = float(h * w)
    gray = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    cands = []
    edges = cv2.Canny(gray, 40, 120); edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    for src in (edges, th, 255 - th):
        cs, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cs, key=cv2.contourArea, reverse=True)[:5]:
            a = cv2.contourArea(c)
            if a < min_frac * total or a > 0.995 * total: continue
            peri = cv2.arcLength(c, True)
            ap = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(ap) == 4 and cv2.isContourConvex(ap): q = ap.reshape(4, 2); conf = 0.9
            else:
                q = cv2.boxPoints(cv2.minAreaRect(c)); conf = 0.55            # rotated-rectangle fallback = lower confidence
            q = order_quad(q); qa = _quad_area(q)
            if qa < min_frac * total: continue
            cands.append((qa * conf, conf, q))
    if not cands: return None, 0.0
    cands.sort(key=lambda t: t[0], reverse=True)
    _, conf, q = cands[0]
    return (q / k).astype(np.float32), float(conf)

def warp_quad(rgb, quad, out_size=None):
    """Perspective-correct the quad to a flat rectangle. Output size from the real edge lengths unless given. Source untouched."""
    q = order_quad(quad)
    wt, wb = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    hl, hr = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    W, H = (int(round(max(wt, wb))), int(round(max(hl, hr)))) if out_size is None else out_size
    W, H = max(8, W), max(8, H)
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32))
    return cv2.warpPerspective(rgb, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def straighten(rgb, angle_deg, fill="replicate"):
    """rotate by angle (deg, positive = counter-clockwise visually) keeping the full image (canvas grows)."""
    h, w = rgb.shape[:2]; c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cs, sn = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * sn + w * cs), int(h * cs + w * sn)
    M[0, 2] += nw / 2.0 - c[0]; M[1, 2] += nh / 2.0 - c[1]
    return cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE if fill == "replicate" else cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))

def estimate_skew(rgb, max_angle=10.0, step=0.25):
    """Text-line skew by projection-profile sharpness. Returns degrees to rotate (use straighten(rgb, result))."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); k = 900.0 / max(g.shape)
    if k < 1: g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    if bw.sum() < 255 * 200: return 0.0
    h, w = bw.shape; best, ba = -1.0, 0.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, 1.0)
        r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
        s = float(np.var(r.sum(axis=1).astype(np.float64)))
        if s > best: best, ba = s, float(a)
    return ba

def detect_text_regions(rgb, min_area_frac=0.0005):
    """Read-only helper: boxes (x,y,w,h) of text-like blocks (morphology). Used for preview / protection, never to edit."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); k = 1000.0 / max(g.shape)
    s = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else g.copy(); k = min(k, 1.0)
    bw = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area_frac * s.shape[0] * s.shape[1] and h > 6: out.append((int(x / k), int(y / k), int(w / k), int(h / k)))
    return sorted(out, key=lambda b: (b[1], b[0]))

# =====================================================================
#  RESTORATION  (cleans what is there; never creates or rewrites content)
# =====================================================================
class RestoreOptions:
    def __init__(self, **kw):
        self.mode = "color"               # color | gray | bw
        self.intensity = 70               # 0..100 overall strength
        self.shadow_removal = True        # also fixes uneven lighting
        self.whiten = True
        self.denoise = True
        self.faint_text = True
        self.despeckle = False            # conservative dust removal, OFF by default (could touch dots / matras / stamps)
        self.despeckle_max_px = 6
        self.adaptive_threshold = False   # forced on for bw mode
        self.block = 35; self.c = 12
        self.sharpen = 40
        self.__dict__.update(kw)

def _bg_estimate(gray, k=None):
    """large-scale background (paper) brightness map: dilate removes dark text, big median smooths"""
    h, w = gray.shape[:2]; k = k or max(15, (min(h, w) // 25) | 1)
    g = cv2.GaussianBlur(gray, (0, 0), 2.5)                  # paper grain/noise would otherwise push the local max up
    d = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.medianBlur(d, k | 1)

def remove_shadows(rgb, strength=1.0):
    """divide each channel by its own background estimate -> flat illumination. strength blends with the original."""
    out = np.empty_like(rgb)
    for i in range(3):
        ch = rgb[:, :, i]; bg = _bg_estimate(ch).astype(np.float32)
        out[:, :, i] = np.clip(ch.astype(np.float32) / np.maximum(bg, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.addWeighted(out, strength, rgb, 1.0 - strength, 0) if strength < 1.0 else out

def whiten_background(rgb, strength=1.0):
    """levels: paper -> white. Only brightens the paper tone; darker content keeps its relative darkness."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hi = float(np.percentile(g, 90)); hi = max(hi, 120.0)
    scale = 255.0 / hi
    out = np.clip(rgb.astype(np.float32) * (1.0 + (scale - 1.0) * strength), 0, 255).astype(np.uint8)
    return out

def ink_mask(rgb, thresh=28):
    """pixels clearly darker than their local paper = content (text, strokes, seals, signatures)"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); bg = _bg_estimate(g)
    return (bg.astype(np.int16) - g.astype(np.int16)) > thresh

def _boost_faint(rgb, gain):
    """amplify existing deviations from local paper tone (faint pencil / faded ink). Pure contrast, adds no strokes."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32); bg = _bg_estimate(g.astype(np.uint8)).astype(np.float32)
    dev = np.clip(bg - g - 10.0, 0, 255)                # how much darker than paper, minus a dead-zone for grain/noise
    out = rgb.astype(np.float32) - (gain * dev)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)

def _despeckle(rgb, max_px):
    """remove ONLY tiny isolated dark specks that have no neighbour within ~3 text heights (dust), keeps dots/matras/stamps"""
    m = ink_mask(rgb, 40).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2: return rgb
    areas = st[1:, cv2.CC_STAT_AREA]; hs = st[1:, cv2.CC_STAT_HEIGHT]
    big = areas > max_px * 3
    ref = float(np.median(hs[big])) if big.any() else 12.0
    reach = max(10.0, 3.0 * ref)
    pts = cen[1:]; tiny = np.where(areas <= max_px)[0]
    if len(tiny) == 0 or len(tiny) > 20000: return rgb
    out = rgb.copy(); bgfill = cv2.medianBlur(rgb, 21)
    notiny = np.where(areas > max_px)[0]
    for i in tiny:
        if len(notiny):
            dist = np.min(np.hypot(pts[notiny, 0] - pts[i, 0], pts[notiny, 1] - pts[i, 1]))
            if dist < reach: continue                 # close to real content -> leave it (could be a dot / mark)
        mask = (lab == i + 1)
        out[mask] = bgfill[mask]
    return out

def restore_document(rgb, opt=None):
    """Clean an existing scan. Order: shadows/lighting -> denoise -> whiten -> faint text -> (despeckle) -> mode.
    Returns a NEW array; the input is never modified. No generative step, so no text/numbers/signatures are invented."""
    opt = opt or RestoreOptions(); k = max(0.0, min(1.0, opt.intensity / 100.0))
    out = rgb.copy()
    if opt.shadow_removal:
        out = remove_shadows(out, 0.6 + 0.4 * k)
        if k >= 0.5: out = remove_shadows(out, 0.8)          # second, gentler pass removes the small residual gradient
    if opt.denoise:
        h = 2.0 + 8.0 * k
        out = cv2.fastNlMeansDenoisingColored(out, None, h, h, 7, 21)
    if opt.whiten: out = whiten_background(out, 0.5 + 0.5 * k)
    if opt.faint_text: out = _boost_faint(out, 0.35 + 0.9 * k)
    if opt.despeckle: out = _despeckle(out, int(opt.despeckle_max_px))
    if opt.sharpen:
        a = opt.sharpen / 100.0 * k; bl = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = np.clip(out.astype(np.float32) * (1 + a) - bl.astype(np.float32) * a, 0, 255).astype(np.uint8)
    if opt.mode == "gray":
        g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY); out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    elif opt.mode == "bw" or opt.adaptive_threshold:
        g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY); blk = int(opt.block) | 1
        t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, max(3, blk), int(opt.c))
        out = cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)
    return out

def ink_retention(original, restored, tol_px=1):
    """Safety metric: share of the ORIGINAL's content pixels still present (within tol_px) in the restored image. 1.0 = nothing lost."""
    a = ink_mask(original, 40)
    if a.sum() == 0: return 1.0
    b = ink_mask(restored, 40).astype(np.uint8)
    if tol_px: b = cv2.dilate(b, np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8))
    return float((a & (b > 0)).sum()) / float(a.sum())

def difference_map(original, restored):
    """Before/After helper: heat-map of how much each pixel changed (so the operator can see what was touched)"""
    d = cv2.absdiff(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY))
    return cv2.applyColorMap(cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)[:, :, ::-1]

class OcrUnavailable(Exception): pass

def ocr_text(rgb, lang="ben+eng"):
    """Optional READ-ONLY OCR through Tesseract (never writes into the document). Raises OcrUnavailable with a clear reason."""
    try: import pytesseract
    except Exception: raise OcrUnavailable("pytesseract is not installed (pip install pytesseract) and Tesseract-OCR must be installed.")
    try: return pytesseract.image_to_string(Image.fromarray(rgb), lang=lang)
    except pytesseract.TesseractNotFoundError: raise OcrUnavailable("Tesseract-OCR program not found. Install it from the UB-Mannheim Windows build and add it to PATH.")
    except Exception as e: raise OcrUnavailable("OCR failed (language data '%s' installed?): %s" % (lang, e))

# =====================================================================
#  SAFE DOWNLOADER  (progress, cancel, retry, resume, verify, clear errors)
# =====================================================================
class DownloadError(Exception): pass
class DownloadCancelled(DownloadError): pass

def download_file(url, dest, expected_sha256=None, min_bytes=1, progress=None, cancel=None, retries=3, timeout=30, user_agent="RiponComputer/3"):
    """Download url -> dest.  Writes dest+'.part' and resumes it after an interruption (HTTP Range), verifies size + optional SHA-256,
    then renames atomically.  progress(done_bytes, total_bytes or 0).  cancel() -> True to stop (the .part is kept for resume).
    Raises DownloadError with a human-readable message (404, no internet, bad checksum ...). Never reports success unless the file verified."""
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_bytes:
        if not expected_sha256 or _sha256(dest) == expected_sha256.lower(): return dest          # already have it: no re-download
        os.remove(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"; last = "unknown error"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        if have: req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                code = getattr(r, "status", 200)
                total = int(r.headers.get("Content-Length") or 0)
                if code == 206:
                    m = re.search(r"/(\d+)$", r.headers.get("Content-Range", "")); total = int(m.group(1)) if m else have + total
                    mode = "ab"
                else:                                                   # server ignored Range: restart from zero
                    have = 0; mode = "wb"
                done = have
                with open(part, mode) as f:
                    while True:
                        if cancel and cancel(): raise DownloadCancelled("Download cancelled (partial file kept, it will resume next time).")
                        chunk = r.read(1 << 16)
                        if not chunk: break
                        f.write(chunk); done += len(chunk)
                        if progress: progress(done, total)
            size = os.path.getsize(part)
            if total and size != total:
                last = "incomplete download (%d of %d bytes)" % (size, total); continue          # retry -> resume
            if size < min_bytes: raise DownloadError("Downloaded file is too small (%d bytes) - the link may not point to the model." % size)
            if expected_sha256 and _sha256(part) != expected_sha256.lower():
                os.remove(part); raise DownloadError("Checksum mismatch - the file is damaged or not the expected model. It was deleted; try again.")
            os.replace(part, dest); return dest
        except DownloadError: raise
        except urllib.error.HTTPError as e:
            if e.code == 404: raise DownloadError("Download link is broken (HTTP 404 Not Found):\n%s\nThe model may have moved. Update its URL in doc_models.json or download it manually into the models folder." % url)
            if e.code == 416:                                           # our .part is bigger than/equal to the file: treat as complete-check
                try: os.remove(part)
                except OSError: pass
                last = "server rejected resume, restarting"; continue
            if e.code in (401, 403): raise DownloadError("Access denied (HTTP %d) for %s. The model may need a login or licence acceptance on its website." % (e.code, url))
            last = "HTTP %d %s" % (e.code, e.reason)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = "network problem: %s" % getattr(e, "reason", e)
        time.sleep(min(2 * attempt, 5) if attempt < retries else 0)
    raise DownloadError("Download failed after %d tries: %s\n(Partial data is kept; press Retry to resume.)" % (retries, last))

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

# ---- document-model registry (user-extendable; nothing is downloaded unless the operator asks) ----
DOC_MODELS_BUILTIN = {}     # intentionally empty: the built-in OpenCV methods need no model. Entries are added via doc_models.json

def load_doc_models(models_dir):
    """merge optional <models_dir>/doc_models.json = {key:{label,url,file,sha256,min_bytes,license,purpose}}"""
    reg = dict(DOC_MODELS_BUILTIN); p = os.path.join(models_dir, "doc_models.json")
    try:
        with open(p, "r", encoding="utf-8") as f: extra = json.load(f)
        for k, v in extra.items():
            if isinstance(v, dict) and v.get("url") and v.get("file"): reg[k] = v
    except FileNotFoundError: pass
    except Exception: pass
    return reg

def doc_model_path(models_dir, entry): return os.path.join(models_dir, "document", entry["file"])
def doc_model_installed(models_dir, entry):
    p = doc_model_path(models_dir, entry)
    return os.path.isfile(p) and os.path.getsize(p) >= int(entry.get("min_bytes", 1000))
def doc_model_download(models_dir, entry, progress=None, cancel=None):
    return download_file(entry["url"], doc_model_path(models_dir, entry), entry.get("sha256"), int(entry.get("min_bytes", 1000)), progress, cancel)
