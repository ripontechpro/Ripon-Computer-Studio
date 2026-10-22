# -*- coding: utf-8 -*-
"""Dialogs of the Layout Studio: Crop / Perspective crop, batch auto-crop, Document restoration, Document models + OCR setup.
Originals are never overwritten: results are NEW files in the layout cache folder and the layer remembers its original (Tools > Revert)."""
import os, json, traceback
import numpy as np
import cv2
from PIL import Image
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF, QPixmap, QImage
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSlider, QCheckBox,
                               QMessageBox, QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QSpinBox, QAbstractItemView)
import ripon_core as rc
from ripon_studio_base import (CACHE_DIR, MODELS_DIR, pil_open, pil_to_qimage, unique_path, run_task, TaskThread, log)

def _np(im): return np.asarray(im.convert("RGB"))

def _save_result(arr, src_path, tag, bw=False):
    """write a derived image into the cache folder (jpg q97 for photos, png for black&white / png sources)"""
    base = os.path.splitext(os.path.basename(src_path))[0][:40]; png = bw or src_path.lower().endswith(".png")
    p = unique_path(os.path.join(CACHE_DIR, "%s_%s.%s" % (base, tag, "png" if png else "jpg")))
    Image.fromarray(arr).save(p, **({"dpi": (300, 300)} if png else {"quality": 97, "dpi": (300, 300), "subsampling": 0})); return p

def _effective_rgb(L):
    """the layer's current picture as RGB, with its non-destructive crop baked in (the file on disk is not changed)"""
    im = pil_open(L.path).convert("RGB")
    if L.crop:
        W, H = im.size; l, t, r, b = L.crop; im = im.crop((int(l * W), int(t * H), max(int(l * W) + 1, int(r * W)), max(int(t * H) + 1, int(b * H))))
    return _np(im)

# =====================================================================
#  CROP  (rectangle with ratio presets  /  perspective with 4 draggable corners)
# =====================================================================
class QuadView(QWidget):
    """picture + 4 corner handles. Points are kept in SOURCE-image pixels. mode 'rect' keeps the box axis-aligned (optional ratio lock)."""
    changed = Signal()
    def __init__(self):
        super().__init__(); self.qimg = None; self.sw = self.sh = 1; self.quad = [QPointF(0, 0)] * 4; self.mode = "rect"; self.ratio = None; self._drag = -1; self.setMinimumSize(420, 420); self.setMouseTracking(True)
    def set_image(self, qimg, sw, sh):
        self.qimg, self.sw, self.sh = qimg, sw, sh; self.reset(); self.update()
    def reset(self):
        m = 0.05; self.quad = [QPointF(self.sw * m, self.sh * m), QPointF(self.sw * (1 - m), self.sh * m), QPointF(self.sw * (1 - m), self.sh * (1 - m)), QPointF(self.sw * m, self.sh * (1 - m))]
        if self.mode == "rect" and self.ratio: self._apply_ratio(2)
        self.changed.emit(); self.update()
    def _geom(self):
        s = min((self.width() - 20) / float(self.sw), (self.height() - 20) / float(self.sh)); return s, (self.width() - self.sw * s) / 2.0, (self.height() - self.sh * s) / 2.0
    def _w(self, p): s, ox, oy = self._geom(); return QPointF(ox + p.x() * s, oy + p.y() * s)
    def _i(self, pt):
        s, ox, oy = self._geom(); return QPointF(min(max((pt.x() - ox) / s, 0), self.sw), min(max((pt.y() - oy) / s, 0), self.sh))
    def set_quad(self, pts):
        self.quad = [QPointF(float(x), float(y)) for x, y in pts]; self.changed.emit(); self.update()
    def quad_np(self): return np.array([[p.x(), p.y()] for p in self.quad], np.float32)
    def to_rect_mode(self):
        xs = [p.x() for p in self.quad]; ys = [p.y() for p in self.quad]; x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        self.quad = [QPointF(x0, y0), QPointF(x1, y0), QPointF(x1, y1), QPointF(x0, y1)]; self.changed.emit(); self.update()
    def _apply_ratio(self, fixed):
        """keep corner `fixed` where it is and force the box to self.ratio (w/h)"""
        if not self.ratio: return
        o = self.quad[fixed]; c = self.quad[(fixed + 2) % 4]; dx, dy = c.x() - o.x(), c.y() - o.y(); sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1); ax, ay = abs(dx) or 1.0, abs(dy) or 1.0
        if ax / ay > self.ratio: ax = ay * self.ratio
        else: ay = ax / self.ratio
        mx, my = (self.sw - o.x()) if sx > 0 else o.x(), (self.sh - o.y()) if sy > 0 else o.y()
        k = min(1.0, mx / ax, my / ay); ax, ay = ax * k, ay * k; nc = QPointF(o.x() + sx * ax, o.y() + sy * ay)
        x0, x1 = sorted([o.x(), nc.x()]); y0, y1 = sorted([o.y(), nc.y()]); self.quad = [QPointF(x0, y0), QPointF(x1, y0), QPointF(x1, y1), QPointF(x0, y1)]
    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor(50, 50, 55))
        if self.qimg is None: return
        s, ox, oy = self._geom(); p.drawImage(QRectF(ox, oy, self.sw * s, self.sh * s), self.qimg)
        poly = QPolygonF([self._w(q) for q in self.quad]); path_out = QPolygonF([QPointF(ox, oy), QPointF(ox + self.sw * s, oy), QPointF(ox + self.sw * s, oy + self.sh * s), QPointF(ox, oy + self.sh * s)])
        p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 110)); p.drawPolygon(path_out.subtracted(poly)); p.setBrush(Qt.NoBrush); p.setPen(QPen(QColor(0, 220, 120), 2)); p.drawPolygon(poly)
        for q in self.quad: w = self._w(q); p.setBrush(QColor(255, 255, 255)); p.setPen(QPen(QColor(0, 120, 215), 2)); p.drawEllipse(w, 8, 8)
    def mousePressEvent(self, e):
        pt = e.position(); best, bd = -1, 20.0
        for i, q in enumerate(self.quad):
            w = self._w(q); d = ((w.x() - pt.x()) ** 2 + (w.y() - pt.y()) ** 2) ** 0.5
            if d < bd: best, bd = i, d
        self._drag = best
    def mouseMoveEvent(self, e):
        if self._drag < 0: return
        pt = self._i(e.position()); i = self._drag
        if self.mode == "persp": self.quad[i] = pt
        else:
            o = self.quad[(i + 2) % 4]; x0, x1 = sorted([o.x(), pt.x()]); y0, y1 = sorted([o.y(), pt.y()])
            self.quad = [QPointF(x0, y0), QPointF(x1, y0), QPointF(x1, y1), QPointF(x0, y1)]
            # after sorting, the dragged corner may have changed index; keep the opposite corner fixed for the ratio step
            if self.ratio:
                fx, fy = o.x(), o.y(); idx = min(range(4), key=lambda k: (self.quad[k].x() - fx) ** 2 + (self.quad[k].y() - fy) ** 2); self._apply_ratio(idx)
            self._drag = min(range(4), key=lambda k: (self.quad[k].x() - pt.x()) ** 2 + (self.quad[k].y() - pt.y()) ** 2)
        self.changed.emit(); self.update()
    def mouseReleaseEvent(self, e): self._drag = -1

RATIOS = [("Free", None), ("Same as image", "orig"), ("1 : 1", 1.0), ("35 × 45 mm passport (7:9)", 35 / 45.0), ("40 × 50 mm visa (4:5)", 0.8), ("3 : 4", 0.75), ("4 : 3", 4 / 3.0),
          ("3 : 2", 1.5), ("16 : 9", 16 / 9.0), ("A4 portrait", 210 / 297.0), ("Bank photo 1.9 × 1.5 in", 1.9 / 1.5), ("ID card 85.6 × 54 mm", 85.6 / 54.0), ("Custom (below)", "custom")]

class CropDialog(QDialog):
    def __init__(self, studio, layer, mode="rect"):
        super().__init__(studio); self.studio, self.L = studio, layer; self.setWindowTitle("Crop - " + layer.name); self.resize(1180, 760)
        self.src = _effective_rgb(layer); H, W = self.src.shape[:2]; k = min(1.0, 1400.0 / max(W, H)); self.k = k
        self.small = cv2.resize(self.src, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else self.src
        self.view = QuadView(); qi = pil_to_qimage(Image.fromarray(self.src if k >= 1 else self.small)); self.view.set_image(qi, W, H)
        self.result = QLabel("Result preview"); self.result.setAlignment(Qt.AlignCenter); self.result.setMinimumSize(360, 360); self.result.setStyleSheet("background:#2b2b30;color:#aaa;")
        self.mode = QComboBox(); self.mode.addItems(["Rectangle crop (manual)", "Perspective crop (4 corners)"]); self.mode.setCurrentIndex(0 if mode == "rect" else 1)
        self.ratio = QComboBox(); [self.ratio.addItem(t, v) for t, v in RATIOS]; self.cw, self.ch = QDoubleSpinBox(), QDoubleSpinBox()
        for s, v in ((self.cw, 4), (self.ch, 3)): s.setRange(0.1, 1000); s.setValue(v); s.setDecimals(2)
        self.angle = QSlider(Qt.Horizontal); self.angle.setRange(-150, 150); self.angle.setValue(0); self.alab = QLabel("0.0°")
        self.info = QLabel(""); self.info.setWordWrap(True)
        side = QVBoxLayout(); side.addWidget(QLabel("<b>Mode</b>")); side.addWidget(self.mode); side.addWidget(QLabel("<b>Aspect ratio</b> (rectangle mode)")); side.addWidget(self.ratio)
        r = QHBoxLayout(); r.addWidget(self.cw); r.addWidget(QLabel(":")); r.addWidget(self.ch); side.addLayout(r)
        side.addWidget(QLabel("<b>Straighten</b> (rotate the result)")); r = QHBoxLayout(); r.addWidget(self.angle); r.addWidget(self.alab); side.addLayout(r)
        for t, fn in (("Auto detect document edges", self.auto), ("Auto straighten text", self.auto_skew), ("Reset", self.reset)): b = QPushButton(t); b.clicked.connect(lambda _=False, f=fn: f()); side.addWidget(b)
        side.addWidget(self.info); side.addStretch(1)
        b = QPushButton("Apply crop"); b.setStyleSheet("background:#2e7d32;color:white;font-weight:bold;padding:8px;"); b.clicked.connect(self.apply); side.addWidget(b)
        c = QPushButton("Cancel"); c.clicked.connect(self.reject); side.addWidget(c); side.addWidget(QLabel("The original file is never changed:\nthe cropped picture is saved as a new file\n(Tools > Revert brings the original back)."))
        lay = QHBoxLayout(self); lay.addWidget(self.view, 3); lay.addWidget(self.result, 2); box = QWidget(); box.setLayout(side); box.setFixedWidth(260); lay.addWidget(box)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(60); self.timer.timeout.connect(self.refresh)
        self.view.changed.connect(self.timer.start); self.angle.valueChanged.connect(self._angle); self.mode.currentIndexChanged.connect(self._mode); self.ratio.currentIndexChanged.connect(self._ratio); self.cw.valueChanged.connect(self._ratio); self.ch.valueChanged.connect(self._ratio)
        self._mode(); self.refresh()
    def _mode(self, *_):
        self.view.mode = "rect" if self.mode.currentIndex() == 0 else "persp"
        if self.view.mode == "rect": self.view.to_rect_mode(); self.ratio.setEnabled(True)
        else: self.ratio.setEnabled(False); self.view.ratio = None
        self._ratio()
    def _ratio(self, *_):
        d = self.ratio.currentData(); H, W = self.src.shape[:2]
        self.view.ratio = None if self.view.mode != "rect" else (W / float(H) if d == "orig" else (self.cw.value() / self.ch.value() if d == "custom" else d))
        self.cw.setEnabled(d == "custom"); self.ch.setEnabled(d == "custom")
        if self.view.ratio and self.view.mode == "rect": self.view._apply_ratio(0); self.view.changed.emit(); self.view.update()
    def _angle(self, v): self.alab.setText("%.1f°" % (v / 10.0)); self.timer.start()
    def reset(self): self.angle.setValue(0); self.view.reset()
    def auto(self):
        q, conf = rc.detect_document_quad(self.src)
        if q is None: return QMessageBox.information(self, "Auto detect", "No clear document edges were found.\nDrag the 4 corners by hand (Perspective mode).")
        self.mode.setCurrentIndex(1); self.view.set_quad(q); self.info.setText("Edges found (confidence %.0f%%). Check the corners and drag them if needed." % (conf * 100))
    def auto_skew(self):
        a = rc.estimate_skew(self.small); self.angle.setValue(int(round(-a * 10))); self.info.setText("Estimated text skew: %.2f°" % a)
    def _make(self, rgb, quad):
        out = rc.warp_quad(rgb, quad); a = self.angle.value() / 10.0
        return rc.straighten(out, a) if abs(a) > 0.01 else out
    def refresh(self):
        try:
            out = self._make(self.small, self.view.quad_np() * self.k); H, W = out.shape[:2]
            self.result.setPixmap(QPixmap.fromImage(pil_to_qimage(Image.fromarray(out))).scaled(self.result.width() - 8, self.result.height() - 8, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            fw = self.view.quad_np(); wpx = int(max(np.linalg.norm(fw[1] - fw[0]), np.linalg.norm(fw[2] - fw[3]))); hpx = int(max(np.linalg.norm(fw[3] - fw[0]), np.linalg.norm(fw[2] - fw[1])))
            self.info.setText("Result about %d × %d px" % (wpx, hpx))
        except Exception as e: self.info.setText("Preview: %s" % e)
    def apply(self):
        try:
            out = self._make(self.src, self.view.quad_np()); p = _save_result(out, self.L.path, "crop")
        except Exception as e:
            log.error("crop failed: %s\n%s", e, traceback.format_exc()); return QMessageBox.warning(self, "Crop", "Crop failed:\n%s" % e)
        st = self.studio; st.push_undo(); st.replace_layer_image(self.L, p); st.cache.clear(); st.refresh_page_scene(); self.accept()

def batch_auto_crop(studio, layers):
    """auto-detect the document in each selected layer, correct perspective, save as new files. Layers with no clear edges are skipped and listed."""
    jobs = [(l.id, l.path, list(l.crop) if l.crop else None) for l in layers]
    def job(task):
        done, skipped = [], []
        for n, (lid, path, crop) in enumerate(jobs):
            task.progress.emit(n, len(jobs), os.path.basename(path))
            if task.cancelled(): break
            try:
                im = pil_open(path).convert("RGB")
                if crop: W, H = im.size; im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
                rgb = _np(im); q, conf = rc.detect_document_quad(rgb)
                if q is None or conf < 0.5: skipped.append(os.path.basename(path)); continue
                done.append((lid, _save_result(rc.warp_quad(rgb, q), path, "autocrop")))
            except Exception as e: log.warning("auto crop %s: %s", path, e); skipped.append(os.path.basename(path))
        return done, skipped
    def finished(res):
        done, skipped = res
        if done:
            studio.push_undo(); byid = {l.id: l for pg in studio.project.pages for l in pg.layers}
            for lid, p in done:
                if lid in byid: studio.replace_layer_image(byid[lid], p)
            studio.cache.clear(); studio.refresh_page_scene()
        QMessageBox.information(studio, "Auto crop", "%d layer(s) cropped and straightened.%s" % (len(done), ("\n\nNo clear edges found (left unchanged, crop them manually):\n" + "\n".join(skipped[:12])) if skipped else ""))
    run_task(studio, "Auto cropping documents", job, finished)

# =====================================================================
#  DOCUMENT RESTORATION
# =====================================================================
class SplitView(QWidget):
    """Before | After with a draggable divider; 'diff' shows what changed"""
    def __init__(self):
        super().__init__(); self.a = self.b = None; self.diff = None; self.pos = 0.5; self.show_diff = False; self.setMinimumSize(520, 520); self.setMouseTracking(True)
    def set_images(self, a, b, diff):
        self.a, self.b, self.diff = pil_to_qimage(Image.fromarray(a)), pil_to_qimage(Image.fromarray(b)), pil_to_qimage(Image.fromarray(diff)); self.update()
    def _r(self):
        if self.a is None: return QRectF()
        s = min((self.width() - 10) / self.a.width(), (self.height() - 10) / self.a.height()); w, h = self.a.width() * s, self.a.height() * s; return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor(45, 45, 50))
        if self.a is None: p.setPen(QColor(170, 170, 170)); p.drawText(self.rect(), Qt.AlignCenter, "Press 'Preview' to see Before / After"); return
        r = self._r()
        if self.show_diff: p.drawImage(r, self.diff); return
        p.drawImage(r, self.b); x = r.left() + r.width() * self.pos; p.save(); p.setClipRect(QRectF(r.left(), r.top(), x - r.left(), r.height())); p.drawImage(r, self.a); p.restore()
        p.setPen(QPen(QColor(255, 255, 0), 2)); p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom())); p.setPen(QColor(255, 255, 255)); p.drawText(int(r.left()) + 6, int(r.top()) + 16, "BEFORE"); p.drawText(int(r.right()) - 52, int(r.top()) + 16, "AFTER")
    def _set(self, e):
        r = self._r()
        if r.width() > 0: self.pos = min(1.0, max(0.0, (e.position().x() - r.left()) / r.width())); self.update()
    def mousePressEvent(self, e): self._set(e)
    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton: self._set(e)

PRESETS = {"Light clean-up": dict(intensity=40, mode="color"), "Standard (recommended)": dict(intensity=70, mode="color"), "Strong (old / faded)": dict(intensity=90, mode="color"),
           "Black & white document": dict(intensity=75, mode="bw"), "Grayscale": dict(intensity=70, mode="gray")}

class RestoreDialog(QDialog):
    def __init__(self, studio, layers):
        super().__init__(studio); self.studio = studio; self.layers = layers; self.setWindowTitle("Document restoration - %d layer(s)" % len(layers)); self.resize(1250, 800); self._prev = None; self._tasks = []
        self.view = SplitView(); self.preset = QComboBox(); self.preset.addItems(PRESETS); self.preset.setCurrentText("Standard (recommended)")
        self.mode = QComboBox(); self.mode.addItems(["Color", "Gray", "Black & White"]); self.inten = QSlider(Qt.Horizontal); self.inten.setRange(0, 100); self.inten.setValue(70); self.ilab = QLabel("70")
        self.c_shadow = QCheckBox("Remove shadows / even out lighting"); self.c_white = QCheckBox("Whiten background"); self.c_noise = QCheckBox("Reduce noise"); self.c_faint = QCheckBox("Enhance faint text")
        self.c_dust = QCheckBox("Remove isolated dust specks (careful)"); self.c_dust.setToolTip("Only tiny dots far from any writing. OFF by default: dots, matras and stamps near text are never touched, but check the result.")
        for c in (self.c_shadow, self.c_white, self.c_noise, self.c_faint): c.setChecked(True)
        self.sharp = QSlider(Qt.Horizontal); self.sharp.setRange(0, 100); self.sharp.setValue(40); self.block = QSpinBox(); self.block.setRange(11, 151); self.block.setValue(35); self.block.setSingleStep(2); self.cc = QSpinBox(); self.cc.setRange(0, 40); self.cc.setValue(12)
        self.c_diff = QCheckBox("Show what changed (difference map)"); self.safety = QLabel(""); self.safety.setWordWrap(True)
        f = QVBoxLayout(); f.addWidget(QLabel("<b>Preset</b>")); f.addWidget(self.preset); f.addWidget(QLabel("<b>Output</b>")); f.addWidget(self.mode)
        r = QHBoxLayout(); r.addWidget(QLabel("Restore intensity")); r.addWidget(self.inten); r.addWidget(self.ilab); f.addLayout(r)
        for c in (self.c_shadow, self.c_white, self.c_noise, self.c_faint, self.c_dust): f.addWidget(c)
        r = QHBoxLayout(); r.addWidget(QLabel("Sharpen")); r.addWidget(self.sharp); f.addLayout(r)
        r = QHBoxLayout(); r.addWidget(QLabel("B&&W block")); r.addWidget(self.block); r.addWidget(QLabel("C")); r.addWidget(self.cc); f.addLayout(r); f.addWidget(self.c_diff); f.addWidget(self.safety)
        b = QPushButton("Preview (first selected layer)"); b.clicked.connect(self.preview); f.addWidget(b)
        b = QPushButton("Read text (OCR, view only)"); b.clicked.connect(self.ocr); f.addWidget(b)
        b = QPushButton("Apply to ALL %d selected layer(s)" % len(layers)); b.setStyleSheet("background:#2e7d32;color:white;font-weight:bold;padding:8px;"); b.clicked.connect(self.apply_all); f.addWidget(b)
        n = QLabel("This only cleans what is already on the page. It never guesses or re-writes text, numbers, dates, seals or signatures. The scan file is kept; Tools > Revert brings it back."); n.setWordWrap(True); f.addWidget(n); f.addStretch(1)
        box = QWidget(); box.setLayout(f); box.setFixedWidth(330); lay = QHBoxLayout(self); lay.addWidget(self.view, 1); lay.addWidget(box)
        self.preset.currentTextChanged.connect(self._preset); self.inten.valueChanged.connect(lambda v: self.ilab.setText(str(v))); self.c_diff.toggled.connect(self._diff)
    def _preset(self, name):
        p = PRESETS[name]; self.inten.setValue(p["intensity"]); self.mode.setCurrentIndex({"color": 0, "gray": 1, "bw": 2}[p["mode"]])
    def _diff(self, v): self.view.show_diff = v; self.view.update()
    def options(self):
        return rc.RestoreOptions(mode=["color", "gray", "bw"][self.mode.currentIndex()], intensity=self.inten.value(), shadow_removal=self.c_shadow.isChecked(), whiten=self.c_white.isChecked(), denoise=self.c_noise.isChecked(),
                                 faint_text=self.c_faint.isChecked(), despeckle=self.c_dust.isChecked(), sharpen=self.sharp.value(), block=self.block.value(), c=self.cc.value())
    def preview(self):
        L = self.layers[0]; opt = self.options()
        def job(task):
            rgb = _effective_rgb(L); k = min(1.0, 1500.0 / max(rgb.shape[:2]))
            small = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else rgb; out = rc.restore_document(small, opt)
            return small, out, rc.difference_map(small, out), rc.ink_retention(small, out)
        def done(res):
            a, b, d, ret = res; self._prev = (a, b); self.view.set_images(a, b, d)
            self.safety.setText("Content kept: %.1f%% of the writing/marks are still there%s" % (ret * 100, "" if ret >= 0.95 else "\n⚠ Lower than expected - lower the intensity or switch off B&W / dust removal and compare."))
        run_task(self, "Building preview", job, done)
    def ocr(self):
        if self._prev is None: return QMessageBox.information(self, "OCR", "Press Preview first.")
        try: txt = rc.ocr_text(self._prev[1], "ben+eng")
        except rc.OcrUnavailable as e: return QMessageBox.information(self, "OCR not available", str(e))
        d = QDialog(self); d.setWindowTitle("Recognised text (view only, nothing is changed in the document)"); t = QTextEdit(); t.setPlainText(txt); lay = QVBoxLayout(d); lay.addWidget(t); d.resize(600, 500); d.exec()
    def apply_all(self):
        opt = self.options(); jobs = [(l.id, l.path, list(l.crop) if l.crop else None) for l in self.layers]; bw = opt.mode == "bw"
        def job(task):
            out = []
            for n, (lid, path, crop) in enumerate(jobs):
                task.progress.emit(n, len(jobs), os.path.basename(path))
                if task.cancelled(): break
                im = pil_open(path).convert("RGB")
                if crop: W, H = im.size; im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
                rgb = _np(im); res = rc.restore_document(rgb, opt); out.append((lid, _save_result(res, path, "restored", bw), rc.ink_retention(rgb, res)))
            return out
        def done(res):
            if not res: return
            st = self.studio; st.push_undo(); byid = {l.id: l for pg in st.project.pages for l in pg.layers}; low = []
            for lid, p, ret in res:
                if lid in byid: st.replace_layer_image(byid[lid], p, keep_width=True); byid[lid].adjust["restored"] = True
                if ret < 0.95: low.append("%s (%.0f%%)" % (byid[lid].name if lid in byid else lid, ret * 100))
            st.cache.clear(); st.refresh_page_scene()
            QMessageBox.information(self, "Restoration", "%d layer(s) restored (batch).%s" % (len(res), ("\n\nCheck these carefully, less of the original content was kept:\n" + "\n".join(low)) if low else "\nContent check passed for all."))
            self.accept()
        run_task(self, "Restoring documents", job, done)

# =====================================================================
#  DOCUMENT MODELS / OCR SETUP
# =====================================================================
EXAMPLE_MODELS = {"example-dewarp-model": {"label": "Document rectification (example entry - edit before use)", "purpose": "page dewarping", "url": "https://example.com/path/to/model.onnx",
                  "file": "model.onnx", "sha256": "<64-hex sha256 of the file>", "min_bytes": 1000000, "license": "<check the model's licence allows your use>"}}

class ModelsDialog(QDialog):
    def __init__(self, studio):
        super().__init__(studio); self.studio = studio; self.setWindowTitle("Document tools: models, OCR, GPU"); self.resize(900, 640); self.reg = rc.load_doc_models(MODELS_DIR); self.errors = {}; self.th = None
        lay = QVBoxLayout(self); self.status = QTextEdit(); self.status.setReadOnly(True); self.status.setMaximumHeight(210); lay.addWidget(self.status)
        self.table = QTableWidget(0, 5); self.table.setHorizontalHeaderLabels(["Model", "Purpose", "Licence", "Size", "Status"]); self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.setEditTriggers(QAbstractItemView.NoEditTriggers); lay.addWidget(self.table)
        self.bar = QProgressBar(); self.bar.setVisible(False); lay.addWidget(self.bar); row = QHBoxLayout()
        self.b_dl = QPushButton("Download / Retry (resumes)"); self.b_cancel = QPushButton("Cancel download"); self.b_err = QPushButton("Error details"); self.b_open = QPushButton("Open models folder"); self.b_ex = QPushButton("Write example doc_models.json")
        self.b_dl.clicked.connect(self.download); self.b_cancel.clicked.connect(self.cancel); self.b_err.clicked.connect(self.show_err); self.b_open.clicked.connect(self.open_folder); self.b_ex.clicked.connect(self.write_example)
        for b in (self.b_dl, self.b_cancel, self.b_err, self.b_open, self.b_ex): row.addWidget(b)
        lay.addLayout(row); self.b_cancel.setEnabled(False); self.refresh()
    def refresh(self):
        lines = ["<b>Built-in methods (no download needed, work offline on CPU):</b> edge detection, perspective correction, deskew, shadow / lighting removal, noise reduction, whitening, faint text enhancement, adaptive threshold, text-region detection."]
        try:
            import onnxruntime as ort; prov = ort.get_available_providers(); gpu = [p for p in prov if p in ("CUDAExecutionProvider", "DmlExecutionProvider")]
            lines.append("<b>AI runtime:</b> onnxruntime found. GPU acceleration: %s. Without a compatible GPU everything runs on the CPU automatically." % (", ".join(gpu) if gpu else "not available (CPU will be used)"))
        except Exception: lines.append("<b>AI runtime:</b> onnxruntime not installed (pip install onnxruntime). Only the built-in OpenCV methods are available; no AI model can run.")
        try:
            import pytesseract; v = pytesseract.get_tesseract_version(); langs = pytesseract.get_languages(config="")
            lines.append("<b>OCR (Tesseract %s):</b> ready. Languages: %s. %s" % (v, ", ".join(langs[:12]), "" if "ben" in langs else "<span style='color:#b00'>Bangla data 'ben' not installed - add ben.traineddata to Tesseract's tessdata folder.</span>"))
        except Exception as e: lines.append("<b>OCR:</b> not ready (%s). Install Tesseract-OCR for Windows (UB-Mannheim build, tick Bengali) and 'pip install pytesseract'. OCR here is read-only." % str(e).splitlines()[0][:80])
        if not self.reg: lines.append("<b>Downloadable models:</b> none registered. The program ships no download links because they could not be verified offline. To add a model, put an entry (url, file, sha256, licence) in <code>%s</code> - use the button below to create an example." % os.path.join(MODELS_DIR, "doc_models.json"))
        self.status.setHtml("<br>".join(lines)); self.table.setRowCount(len(self.reg))
        for r, (k, e) in enumerate(self.reg.items()):
            ok = rc.doc_model_installed(MODELS_DIR, e)
            for c, v in enumerate((e.get("label", k), e.get("purpose", ""), e.get("license", "unknown - check before use"), "%.0f MB" % (e.get("min_bytes", 0) / 1e6) if e.get("min_bytes") else "?", "Installed" if ok else ("Incomplete download (can resume)" if os.path.isfile(rc.doc_model_path(MODELS_DIR, e) + ".part") else "Not downloaded"))): self.table.setItem(r, c, QTableWidgetItem(v))
            self.table.item(r, 0).setData(Qt.UserRole, k)
    def _key(self):
        r = self.table.currentRow(); return self.table.item(r, 0).data(Qt.UserRole) if r >= 0 else None
    def download(self):
        k = self._key()
        if not k: return QMessageBox.information(self, "Models", "Select a model in the list first.")
        e = self.reg[k]
        if rc.doc_model_installed(MODELS_DIR, e): return QMessageBox.information(self, "Models", "Already installed - nothing to download.")
        if QMessageBox.question(self, "Download", "Download '%s'?\n\nLicence: %s\nFrom: %s" % (e.get("label", k), e.get("license", "unknown"), e["url"])) != QMessageBox.Yes: return
        self.bar.setVisible(True); self.bar.setRange(0, 0); self.b_dl.setEnabled(False); self.b_cancel.setEnabled(True)
        def job(task):
            return rc.doc_model_download(MODELS_DIR, e, progress=lambda d, t: task.progress.emit(d, t, ""), cancel=task.cancelled)
        self.th = TaskThread(job, self); self.th.progress.connect(self._prog); self.th.done.connect(lambda p: self._finish(k, None)); self.th.failed.connect(lambda m: self._finish(k, m)); self.th.start()
    def _prog(self, d, t, _):
        if t > 0: self.bar.setRange(0, 1000); self.bar.setValue(int(1000.0 * d / t)); self.bar.setFormat("%.1f / %.1f MB" % (d / 1e6, t / 1e6))
    def _finish(self, k, err):
        self.bar.setVisible(False); self.b_dl.setEnabled(True); self.b_cancel.setEnabled(False)
        if err: self.errors[k] = err; QMessageBox.warning(self, "Download", "%s\n\nUse 'Download / Retry' to try again (a partial file is resumed)." % err)
        else: self.errors.pop(k, None); QMessageBox.information(self, "Download", "Model downloaded and verified.")
        self.refresh()
    def cancel(self):
        if self.th: self.th.cancel()
    def show_err(self):
        k = self._key(); QMessageBox.information(self, "Error details", self.errors.get(k, "No error recorded for this model."))
    def open_folder(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        try: os.startfile(MODELS_DIR)
        except Exception: QMessageBox.information(self, "Models folder", MODELS_DIR)
    def write_example(self):
        p = os.path.join(MODELS_DIR, "doc_models.example.json")
        with open(p, "w", encoding="utf-8") as f: json.dump(EXAMPLE_MODELS, f, indent=1)
        QMessageBox.information(self, "Example written", "%s\n\nCopy it to doc_models.json, fill in the real URL / sha256 / licence, then reopen this window." % p)
