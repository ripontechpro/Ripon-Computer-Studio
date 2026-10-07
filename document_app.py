import sys, os, io, time, tempfile
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont, features
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QComboBox, QFileDialog, QMessageBox, QSpinBox, QCheckBox,
                               QGroupBox, QScrollArea, QRubberBand, QInputDialog)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QRect, QSize

DPI = 300
A4 = (2480, 3508)
FONTS = {"Nirmala UI (Bangla)": "nirmala.ttf", "Arial": "arial.ttf", "Times New Roman": "times.ttf"}
COLORS = {"Black": (0, 0, 0), "Blue": (0, 0, 160), "Red": (200, 0, 0)}

def mm2px(mm): return int(round(mm / 25.4 * DPI))

# ================= SCANNER (Windows WIA) =================
def scan_with_wia():
    import win32com.client
    dlg = win32com.client.Dispatch("WIA.CommonDialog")
    png = "{B96B3CAF-0728-11D3-9D7B-0000F81EF32E}"
    img = dlg.ShowAcquireImage(1, 1, 131072, png, True, True, False)  # scanner, color, max quality
    if img is None:
        return None
    path = os.path.join(tempfile.gettempdir(), "wia_scan_%d.png" % int(time.time() * 1000))
    img.SaveFile(path)
    im = Image.open(path).convert("RGB"); im.load()
    try: os.remove(path)
    except OSError: pass
    return im

def load_pdf(path, dpi=200):
    import fitz  # pymupdf
    pages = []
    for p in fitz.open(path):
        pix = p.get_pixmap(dpi=dpi)
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return pages

# ================= AUTO CLEAN =================
def order_pts(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)

def find_doc(rgb):
    h, w = rgb.shape[:2]
    sc = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(rgb, None, fx=sc, fy=sc) if sc < 1 else rgb
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    e = cv2.dilate(cv2.Canny(g, 50, 150), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(ap) == 4 and cv2.contourArea(ap) > 0.4 * small.shape[0] * small.shape[1]:
            return order_pts(ap / sc)
    return None

def warp(rgb, pts):
    tl, tr, br, bl = pts
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(pts, dst), (W, H))

def deskew(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sc = min(1.0, 600.0 / max(g.shape))
    small = cv2.resize(g, None, fx=sc, fy=sc) if sc < 1 else g
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape; c = (w / 2, h / 2)
    def score(a):
        r = cv2.warpAffine(bw, cv2.getRotationMatrix2D(c, a, 1.0), (w, h), flags=cv2.INTER_NEAREST)
        return np.sum(np.diff(r.sum(1).astype(np.float64)) ** 2)
    best = max(np.arange(-5, 5.01, 0.5), key=score)
    best = max(np.arange(best - 0.5, best + 0.51, 0.1), key=score)
    if abs(best) < 0.2:
        return rgb
    H, W = rgb.shape[:2]
    return cv2.warpAffine(rgb, cv2.getRotationMatrix2D((W / 2, H / 2), best, 1.0), (W, H),
                          flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

def whiten(rgb):
    """Shadow, hat-er dag, hoyud bhab shoriye kagoj shada kore"""
    h, w = rgb.shape[:2]
    sc = 0.25 if max(h, w) > 1200 else 1.0
    out = []
    for ch in cv2.split(rgb):
        s = cv2.resize(ch, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else ch
        bg = cv2.medianBlur(cv2.dilate(s, np.ones((7, 7), np.uint8)), 21)
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(np.clip(ch.astype(np.float32) * 255.0 / np.maximum(bg, 1), 0, 255).astype(np.uint8))
    return cv2.merge(out)

def to_bw(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bs = max(15, (min(g.shape) // 40) | 1)
    return cv2.cvtColor(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, bs, 15), cv2.COLOR_GRAY2RGB)

def sharpen(rgb):
    return cv2.addWeighted(rgb, 1.5, cv2.GaussianBlur(rgb, (0, 0), 1.5), -0.5, 0)

def auto_clean(img, o):
    rgb = np.array(img.convert("RGB")); msgs = []
    if o["crop"]:
        pts = find_doc(rgb)
        if pts is not None: rgb = warp(rgb, pts); msgs.append("border crop hoyeche")
        else: msgs.append("border paini (crop skip)")
    if o["straight"]: rgb = deskew(rgb)
    if o["whiten"]: rgb = whiten(rgb)
    if o["mode"] == "Gray":
        rgb = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    elif o["mode"] == "Black & White":
        rgb = to_bw(rgb)
    if o["sharp"] and o["mode"] != "Black & White":
        rgb = sharpen(rgb)
    return Image.fromarray(rgb), ", ".join(msgs)

# ================= EDIT TOOLS =================
def erase_fill(img, box):
    """Selected jayga ta charpasher kagoj er rong diye dhake"""
    arr = np.array(img.convert("RGB")); x0, y0, x1, y1 = box; m = 12
    ex0, ey0 = max(0, x0 - m), max(0, y0 - m)
    ex1, ey1 = min(arr.shape[1], x1 + m), min(arr.shape[0], y1 + m)
    sub = arr[ey0:ey1, ex0:ex1]
    ring = np.ones(sub.shape[:2], bool)
    ring[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = False
    col = np.median(sub[ring], axis=0).astype(np.uint8) if ring.sum() else np.array([255, 255, 255], np.uint8)
    arr[y0:y1, x0:x1] = col
    return Image.fromarray(arr)

def inpaint_rect(img, box):
    arr = np.array(img.convert("RGB")); x0, y0, x1, y1 = box
    mask = np.zeros(arr.shape[:2], np.uint8); mask[y0:y1, x0:x1] = 255
    return Image.fromarray(cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA))

def add_text(img, text, box, font_file, color):
    x0, y0, x1, y1 = box
    size = max(10, int((y1 - y0) * 0.8))
    try:
        eng = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
        font = ImageFont.truetype(font_file, size, layout_engine=eng)
    except Exception:
        font = ImageFont.load_default()
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    try: d.text((x0, (y0 + y1) // 2), text, font=font, fill=color, anchor="lm")
    except Exception: d.text((x0, y0), text, font=font, fill=color)
    return out

# ================= SIZE / PDF / LAYOUT =================
def compress_to_kb(img, kb, w=0, h=0):
    img = img.convert("RGB")
    if w > 0 and h > 0:
        img = img.resize((w, h), Image.LANCZOS)
    while True:
        lo, hi, best = 5, 95, None
        while lo <= hi:
            q = (lo + hi) // 2
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() <= kb * 1024: best = buf.getvalue(); lo = q + 1
            else: hi = q - 1
        if best: return best, img.size
        if img.width < 100: return buf.getvalue(), img.size
        img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.LANCZOS)

def fit_a4(img):
    sheet = Image.new("RGB", A4, (255, 255, 255))
    s = min(A4[0] / img.width, A4[1] / img.height)
    im = img.convert("RGB").resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    sheet.paste(im, ((A4[0] - im.width) // 2, (A4[1] - im.height) // 2))
    return sheet

def id_layout(front, back, width_mm):
    sheet = Image.new("RGB", A4, (255, 255, 255))
    y = mm2px(30)
    for im in (front, back):
        w = mm2px(width_mm); h = int(im.height * w / im.width)
        sheet.paste(im.convert("RGB").resize((w, h), Image.LANCZOS), ((A4[0] - w) // 2, y))
        y += h + mm2px(15)
    return sheet

def save_pdf(pages, path, a4):
    pp = [fit_a4(p) if a4 else p.convert("RGB") for p in pages]
    pp[0].save(path, "PDF", resolution=DPI, save_all=True, append_images=pp[1:])

# ================= GUI =================
def pil_to_pixmap(im):
    im = im.convert("RGB")
    q = QImage(im.tobytes(), im.width, im.height, 3 * im.width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q)

class ImageView(QLabel):
    """Mouse diye drag kore box select kora jay"""
    def __init__(self):
        super().__init__("Scan koro ba file open koro")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(560, 650)
        self.setStyleSheet("border:1px solid #888; background:#eee;")
        self.pil = None; self.sel = None; self.origin = None
        self.rb = QRubberBand(QRubberBand.Rectangle, self)

    def set_image(self, pil):
        self.pil = pil; self.sel = None; self.rb.hide(); self.refresh()

    def refresh(self):
        if self.pil is not None:
            self.setPixmap(pil_to_pixmap(self.pil).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        self.sel = None; self.rb.hide(); self.refresh()

    def _map(self, p):
        pm = self.pixmap()
        if pm is None or pm.isNull(): return 0, 0
        ox, oy = (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2
        sc = self.pil.width / pm.width()
        x = min(max((p.x() - ox) * sc, 0), self.pil.width)
        y = min(max((p.y() - oy) * sc, 0), self.pil.height)
        return int(x), int(y)

    def mousePressEvent(self, e):
        if self.pil is None: return
        self.origin = e.position().toPoint()
        self.rb.setGeometry(QRect(self.origin, QSize())); self.rb.show()

    def mouseMoveEvent(self, e):
        if self.origin is not None:
            self.rb.setGeometry(QRect(self.origin, e.position().toPoint()).normalized())

    def mouseReleaseEvent(self, e):
        if self.origin is None or self.pil is None: return
        a, b = self._map(self.origin), self._map(e.position().toPoint())
        self.origin = None
        x0, x1 = sorted((a[0], b[0])); y0, y1 = sorted((a[1], b[1]))
        if (x1 - x0) < 5 or (y1 - y0) < 5: self.sel = None; self.rb.hide()
        else: self.sel = (x0, y0, x1, y1)

def btn(text, fn, style=""):
    b = QPushButton(text); b.setMinimumHeight(36); b.clicked.connect(fn)
    if style: b.setStyleSheet(style)
    return b

def group(title, widgets):
    g = QGroupBox(title); l = QVBoxLayout(g)
    for w in widgets:
        l.addLayout(w) if hasattr(w, "addWidget") else l.addWidget(w)
    return g

class DocApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Studio")
        self.resize(1100, 780)
        self.img = None; self.history = []; self.pages = []
        self.view = ImageView()
        self.status = QLabel("")

        # auto clean options
        self.mode = QComboBox(); self.mode.addItems(["Color", "Gray", "Black & White"])
        self.c_crop = QCheckBox("Auto Border Crop (kalo/extra dhar kete felo)")
        self.c_straight = QCheckBox("Auto Straighten (tirchha thik koro)")
        self.c_whiten = QCheckBox("Shadow / Dag remove (kagoj shada)")
        self.c_sharp = QCheckBox("Sharpen (lekha spasto)")
        for c in (self.c_crop, self.c_straight, self.c_whiten, self.c_sharp): c.setChecked(True)

        # text add
        self.font_box = QComboBox(); self.font_box.addItems(FONTS.keys())
        self.col_box = QComboBox(); self.col_box.addItems(COLORS.keys())

        # pages
        self.c_a4 = QCheckBox("PDF e A4 fit koro"); self.c_a4.setChecked(True)
        self.card_w = QSpinBox(); self.card_w.setRange(40, 190); self.card_w.setValue(100); self.card_w.setSuffix(" mm width")

        # size fix
        self.kb = QSpinBox(); self.kb.setRange(10, 5000); self.kb.setValue(100); self.kb.setSuffix(" KB max")
        self.sw = QSpinBox(); self.sw.setRange(0, 10000); self.sw.setSpecialValueText("Width auto")
        self.sh = QSpinBox(); self.sh.setRange(0, 10000); self.sh.setSpecialValueText("Height auto")

        def row(*ws):
            l = QHBoxLayout()
            for w in ws: l.addWidget(w)
            return l

        side = QVBoxLayout()
        side.addWidget(group("1. Input", [
            btn("Scanner theke Scan", self.scan, "background:#1565c0;color:white;font-weight:bold;"),
            btn("File Open (Image / PDF)", self.open_files)]))
        side.addWidget(group("2. AUTO CLEAN", [
            QLabel("Mode"), self.mode, self.c_crop, self.c_straight, self.c_whiten, self.c_sharp,
            btn("AUTO CLEAN CHALAO", self.run_auto, "background:#2e7d32;color:white;font-weight:bold;")]))
        side.addWidget(group("3. Edit (age mouse diye box select koro)", [
            row(btn("Rotate Left", lambda: self.rotate(90)), btn("Rotate Right", lambda: self.rotate(-90))),
            btn("Selected Crop", self.crop_sel),
            btn("Mochon (charpasher rong diye)", self.erase),
            btn("Dag Remove (Inpaint)", self.inpaint),
            row(self.font_box, self.col_box),
            btn("Lekha Add (box er jaygay)", self.add_txt),
            btn("Undo", self.undo)]))
        side.addWidget(group("4. Pages / PDF", [
            btn("Ei page ta list e Joma koro", self.add_page),
            btn("Pages Clear", self.clear_pages),
            self.c_a4,
            btn("PDF Save (sob page)", self.save_pdf_btn),
            row(self.card_w),
            btn("NID/Card: Front+Back ek A4 te", self.card_sheet)]))
        side.addWidget(group("5. Size Fix (online form)", [
            self.kb, row(self.sw, self.sh), btn("KB e Save (JPG)", self.save_kb)]))
        side.addWidget(group("6. Save / Print", [
            btn("Save (JPG/PNG)", self.save_current), btn("Print", self.print_current)]))
        side.addWidget(self.status); side.addStretch()

        panel = QWidget(); panel.setLayout(side)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); scroll.setFixedWidth(360)
        root = QHBoxLayout(self); root.addWidget(self.view, 1); root.addWidget(scroll)

    # ---- helpers ----
    def set_img(self, im, push=True, msg=""):
        if push and self.img is not None:
            self.history.append(self.img); self.history = self.history[-15:]
        self.img = im; self.view.set_image(im)
        self.status.setText("%s  [%dx%d | pages: %d]" % (msg, im.width, im.height, len(self.pages)))

    def need_img(self):
        if self.img is None: QMessageBox.warning(self, "Oops", "Age scan ba file open koro"); return False
        return True

    def need_sel(self):
        if not self.need_img(): return None
        if self.view.sel is None: QMessageBox.information(self, "Box select koro", "Mouse diye chobi te drag kore ekta box select koro"); return None
        return self.view.sel

    # ---- input ----
    def scan(self):
        try:
            im = scan_with_wia()
            if im is not None: self.set_img(im, msg="Scan holo")
        except ImportError:
            QMessageBox.critical(self, "Error", "pip install pywin32 chalao")
        except Exception as e:
            QMessageBox.critical(self, "Scanner error", str(e))

    def open_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Open", "", "Files (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.pdf)")
        imgs = []
        try:
            for p in ps:
                if p.lower().endswith(".pdf"): imgs += load_pdf(p)
                else: imgs.append(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        except ImportError:
            return QMessageBox.critical(self, "Error", "PDF er jonno: pip install pymupdf")
        if imgs:
            if len(imgs) > 1: self.pages.extend(imgs)
            self.history = []; self.set_img(imgs[0], push=False, msg="Opened %d file/page" % len(imgs))

    # ---- auto ----
    def run_auto(self):
        if not self.need_img(): return
        self.status.setText("Processing..."); QApplication.processEvents()
        o = dict(mode=self.mode.currentText(), crop=self.c_crop.isChecked(), straight=self.c_straight.isChecked(),
                 whiten=self.c_whiten.isChecked(), sharp=self.c_sharp.isChecked())
        try:
            im, msg = auto_clean(self.img, o); self.set_img(im, msg=msg or "Done")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ---- edit ----
    def rotate(self, deg):
        if self.need_img(): self.set_img(self.img.rotate(deg, expand=True, fillcolor=(255, 255, 255)), msg="Rotated")

    def crop_sel(self):
        b = self.need_sel()
        if b: self.set_img(self.img.crop(b), msg="Cropped")

    def erase(self):
        b = self.need_sel()
        if b: self.set_img(erase_fill(self.img, b), msg="Mochon holo")

    def inpaint(self):
        b = self.need_sel()
        if b: self.set_img(inpaint_rect(self.img, b), msg="Dag remove holo")

    def add_txt(self):
        b = self.need_sel()
        if not b: return
        t, ok = QInputDialog.getText(self, "Lekha", "Ki lekha add korbe?")
        if ok and t:
            self.set_img(add_text(self.img, t, b, FONTS[self.font_box.currentText()],
                                  COLORS[self.col_box.currentText()]), msg="Lekha add holo")

    def undo(self):
        if self.history: self.set_img(self.history.pop(), push=False, msg="Undo")

    # ---- pages ----
    def add_page(self):
        if self.need_img(): self.pages.append(self.img.copy()); self.status.setText("Pages: %d" % len(self.pages))

    def clear_pages(self):
        self.pages = []; self.status.setText("Pages clear")

    def save_pdf_btn(self):
        pages = self.pages or ([self.img] if self.img is not None else [])
        if not pages: return QMessageBox.warning(self, "Oops", "Kono page nai")
        p, _ = QFileDialog.getSaveFileName(self, "PDF", "document.pdf", "PDF (*.pdf)")
        if p: save_pdf(pages, p, self.c_a4.isChecked()); self.status.setText("PDF saved (%d page)" % len(pages))

    def card_sheet(self):
        if len(self.pages) < 2:
            return QMessageBox.information(self, "Card", "Age front ar back duto side crop kore 'list e Joma koro' chapo (2 ta page lagbe)")
        self.set_img(id_layout(self.pages[0], self.pages[1], self.card_w.value()), msg="Card layout ready")

    # ---- save ----
    def save_kb(self):
        if not self.need_img(): return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "photo.jpg", "JPG (*.jpg)")
        if p:
            data, size = compress_to_kb(self.img, self.kb.value(), self.sw.value(), self.sh.value())
            open(p, "wb").write(data)
            self.status.setText("Saved: %d KB, %dx%d" % (len(data) // 1024, size[0], size[1]))

    def save_current(self):
        if not self.need_img(): return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "scan.jpg", "Images (*.jpg *.png)")
        if p: self.img.save(p, quality=95, dpi=(DPI, DPI)); self.status.setText("Saved")

    def print_current(self):
        if not self.need_img(): return
        p = os.path.join(os.environ.get("TEMP", "."), "print_doc.jpg")
        fit_a4(self.img).save(p, quality=95, dpi=(DPI, DPI))
        os.startfile(p, "print")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DocApp(); w.show(); sys.exit(app.exec())
