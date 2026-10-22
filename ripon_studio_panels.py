# -*- coding: utf-8 -*-
"""Tool panels of the Layout Studio: Bank Photo Duet, ID Card Studio, Passport / photo sheet, Print & Export.
Each panel computes a layout with ripon_core (exact mm geometry), shows a preview with the REAL size, and builds editable layers."""
import os, traceback
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
                               QFileDialog, QMessageBox, QInputDialog, QGroupBox)
import ripon_core as rc
from ripon_studio_base import IMAGE_FILTER, load_layout_presets, store_layout_presets, cover_crop, pil_open, unique_path, CACHE_DIR, log

# =====================================================================
#  small reusable widgets
# =====================================================================
class SizeEdit(QWidget):
    """width x height with an mm / inch switch; the value is always available in mm"""
    changed = Signal()
    def __init__(self, w_mm, h_mm, unit="mm"):
        super().__init__(); self._u = unit; lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.sw, self.sh = QDoubleSpinBox(), QDoubleSpinBox(); self.unit = QComboBox(); self.unit.addItems(["mm", "inch"]); self.unit.setCurrentText(unit)
        for s in (self.sw, self.sh): s.setRange(1, 5000); s.setKeyboardTracking(False); s.valueChanged.connect(lambda *_: self.changed.emit())
        lay.addWidget(self.sw); lay.addWidget(QLabel("×")); lay.addWidget(self.sh); lay.addWidget(self.unit); self._decimals(); self.set_mm(w_mm, h_mm)
        self.unit.currentTextChanged.connect(self._switch)
    def _decimals(self):
        for s in (self.sw, self.sh): s.setDecimals(2 if self._u == "mm" else 3); s.setSingleStep(0.5 if self._u == "mm" else 0.05)
    def mm(self):
        f = 25.4 if self._u == "inch" else 1.0; return self.sw.value() * f, self.sh.value() * f
    def set_mm(self, w, h):
        f = 25.4 if self._u == "inch" else 1.0
        for s, v in ((self.sw, w), (self.sh, h)): s.blockSignals(True); s.setValue(v / f); s.blockSignals(False)
        self.changed.emit()
    def _switch(self, u):
        w, h = self.mm(); self._u = u; self._decimals(); self.set_mm(w, h)

class PaperPick(QWidget):
    changed = Signal()
    def __init__(self, studio):
        super().__init__(); self.studio = studio; lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.use_cur = QCheckBox("Use the current page's size"); self.use_cur.setChecked(True)
        self.paper = QComboBox(); self.paper.addItems(rc.PAPER_NAMES); self.paper.setCurrentText("A4"); self.orient = QComboBox(); self.orient.addItems(["Portrait", "Landscape"])
        self.custom = SizeEdit(210, 297); r = QHBoxLayout(); r.addWidget(self.paper); r.addWidget(self.orient)
        lay.addWidget(self.use_cur); lay.addLayout(r); lay.addWidget(self.custom)
        for w in (self.use_cur, self.paper, self.orient): (w.toggled if isinstance(w, QCheckBox) else w.currentTextChanged).connect(self._upd)
        self.custom.changed.connect(lambda: self.changed.emit()); self._upd()
    def _upd(self, *_):
        cur = self.use_cur.isChecked(); self.paper.setEnabled(not cur); self.orient.setEnabled(not cur); self.custom.setEnabled(not cur and self.paper.currentText() == "Custom"); self.changed.emit()
    def size(self):
        """(w_mm, h_mm, paper name, orientation, margin_mm)"""
        pg = self.studio.cur_page()
        if self.use_cur.isChecked() and pg: return pg.w_mm, pg.h_mm, pg.paper, pg.orientation, pg.margin_mm
        w, h = rc.page_size_mm(self.paper.currentText(), self.orient.currentText(), self.custom.mm()); return w, h, self.paper.currentText(), self.orient.currentText(), 5.0

def _spin(lo, hi, val, dec=1, suffix=" mm", step=0.5):
    s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val); s.setSuffix(suffix); s.setSingleStep(step); s.setKeyboardTracking(False); return s

class FilePick(QWidget):
    changed = Signal()
    def __init__(self, studio, label):
        super().__init__(); self.studio = studio; self.path = ""; lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lbl = QLabel("(none)"); self.lbl.setMinimumWidth(80); b = QPushButton("File..."); b.clicked.connect(self._pick); b2 = QPushButton("Selected layer"); b2.setToolTip("use the image of the layer selected on the canvas"); b2.clicked.connect(self._from_layer)
        lay.addWidget(self.lbl, 1); lay.addWidget(b); lay.addWidget(b2); self.setToolTip(label)
    def _set(self, p): self.path = p or ""; self.lbl.setText(os.path.basename(p) if p else "(none)"); self.changed.emit()
    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose image", "", IMAGE_FILTER)
        if p: self._set(p)
    def _from_layer(self):
        ls = self.studio.selected_layers()
        if ls and ls[0].path: self._set(ls[0].path)
        else: QMessageBox.information(self, "Image", "Select an image layer on the canvas first.")

# =====================================================================
#  base panel
# =====================================================================
class LayoutPanel(QWidget):
    KIND = "panel"
    def __init__(self, studio):
        super().__init__(); self.studio = studio; self.root = QVBoxLayout(self); self.form = QFormLayout(); self.root.addLayout(self.form)
        self.preview = QLabel(); self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumHeight(220); self.preview.setStyleSheet("background:#e9e9ee;")
        self.info = QLabel(""); self.info.setWordWrap(True); self.target = QComboBox(); self.target.addItems(["Build on a NEW page", "Replace everything on the current page", "Add to the current page"])
        self.preset = QComboBox(); self.preset.setEditable(False); self.fit = QComboBox(); self.fit.addItems(["cover", "contain", "stretch"])
        self.fit.setToolTip("cover = fill the box (centre crop, no distortion); contain = whole image inside the box; stretch = fill and distort")
    def finish(self):
        pr = QHBoxLayout(); pr.addWidget(QLabel("Preset")); pr.addWidget(self.preset, 1)
        for t, fn in (("Save", self.preset_save), ("Delete", self.preset_delete)): b = QPushButton(t); b.clicked.connect(lambda _=False, f=fn: f()); pr.addWidget(b)
        self.root.addLayout(pr); self.root.addWidget(self.preview); self.root.addWidget(self.info); self.root.addWidget(self.target)
        r = QHBoxLayout(); b1 = QPushButton("Update preview"); b1.clicked.connect(self.update_preview); b2 = QPushButton("Build layout"); b2.setStyleSheet("background:#2e7d32;color:white;font-weight:bold;"); b2.clicked.connect(self.build)
        b3 = QPushButton("Real-size print preview"); b3.clicked.connect(self.print_preview); r.addWidget(b1); r.addWidget(b2); self.root.addLayout(r); self.root.addWidget(b3); self.root.addStretch(1)
        self.preset.activated.connect(self.preset_load); self.refresh_presets(); self.update_preview()
    # presets
    def refresh_presets(self, select=None):
        d = load_layout_presets().get(self.KIND, {}); self.preset.blockSignals(True); self.preset.clear(); self.preset.addItem("(current settings)")
        for k in d: self.preset.addItem(k)
        if select: self.preset.setCurrentText(select)
        self.preset.blockSignals(False)
    def preset_save(self):
        t, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not t.strip(): return
        allp = load_layout_presets(); allp.setdefault(self.KIND, {})[t.strip()] = self.get_state(); store_layout_presets(allp); self.refresh_presets(t.strip())
    def preset_load(self, i):
        if i <= 0: return
        st = load_layout_presets().get(self.KIND, {}).get(self.preset.currentText())
        if st: self.set_state(st); self.update_preview()
    def preset_delete(self):
        n = self.preset.currentText(); allp = load_layout_presets()
        if n in allp.get(self.KIND, {}) and QMessageBox.question(self, "Delete preset", "Delete preset '%s'?" % n) == QMessageBox.Yes:
            del allp[self.KIND][n]; store_layout_presets(allp); self.refresh_presets()
    # preview + build
    def compute(self): raise NotImplementedError
    def get_state(self): return {}
    def set_state(self, st): pass
    def update_preview(self, *_):
        try: res = self.compute()
        except Exception as e: self.info.setText("⚠ %s" % e); self.preview.clear(); return
        self.info.setText(res["info"])
        if not res["pages"]: self.preview.setText("Nothing fits on this page size."); return
        W, H = res["page"][0], res["page"][1]; items = []
        for pl in res["pages"][0]:
            im = self.studio.cache.pil(pl["path"]) if pl.get("path") else None
            items.append((pl["rect"], im, pl["rot"]))
        pm = rc.render_rects(W, H, items, 40, fit=self.fit.currentText(), border_mm=res.get("border", 0.0), cut_marks=False)
        from ripon_studio_base import pil_to_qimage
        self.preview.setPixmap(QPixmap.fromImage(pil_to_qimage(pm)).scaled(340, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def to_layer(self, pl, fit, border):
        x, y, w, h = pl["rect"]; rot = pl["rot"]
        pw, ph = (h, w) if rot else (w, h); lx, ly = ((x + w / 2.0 - pw / 2.0), (y + h / 2.0 - ph / 2.0)) if rot else (x, y)
        q = self.studio.cache.qimage(pl["path"]) if pl.get("path") else None; crop = None
        if q is not None:
            if fit == "cover": crop = cover_crop(q.width(), q.height(), pw / ph)
            elif fit == "contain":
                r = q.width() / float(q.height())
                if r > pw / ph: nw, nh = pw, pw / r
                else: nw, nh = ph * r, ph
                cx, cy = lx + pw / 2.0, ly + ph / 2.0; pw, ph = nw, nh; lx, ly = cx - pw / 2.0, cy - ph / 2.0
        L = rc.Layer(pl["name"], pl.get("path") or "", lx, ly, pw, ph, 90.0 if rot else 0.0); L.crop = crop; L.border_mm = border; return L
    def build(self):
        try: res = self.compute()
        except Exception as e: return QMessageBox.warning(self, "Layout", str(e))
        if not res["pages"]: return QMessageBox.warning(self, "Layout", "The layout does not fit on this page size.\nChange the size, margins, gap or paper.")
        st = self.studio; W, H, paper, orient, margin = res["page"]; mode = self.target.currentIndex(); st.push_undo(); first_idx = None
        for k, plist in enumerate(res["pages"]):
            layers = [self.to_layer(pl, self.fit.currentText(), res.get("border", 0.0)) for pl in plist]
            if mode == 0 or k > 0: idx = st.add_page_silent(W, H, "%s %d" % (self.TITLE, len(st.project.pages) + 1), paper, orient, margin)
            else:
                idx = st.cur_index(); pg = st.project.pages[idx]
                if abs(pg.w_mm - W) > 0.01 or abs(pg.h_mm - H) > 0.01: pg.w_mm, pg.h_mm, pg.paper, pg.orientation = W, H, paper, orient
                if mode == 1: pg.layers = []
            st.add_layers_to_page(layers, idx); first_idx = idx if first_idx is None else first_idx
        st.rebuild_all(select=first_idx); st.say("%s built: %s" % (self.TITLE, res["info"].splitlines()[0]))
    def print_preview(self):
        self.build(); self.studio.print_dialog(True)

# =====================================================================
#  BANK PHOTO DUET   (whole duet = 2 photos together = 1.9 x 1.5 inch = 570 x 450 px at 300 DPI)
# =====================================================================
class DuetPanel(LayoutPanel):
    KIND = "duet"; TITLE = "Duet"
    def __init__(self, studio):
        super().__init__(studio); f = self.form
        self.size_e = SizeEdit(*[v * 25.4 for v in rc.BANK_PHOTO_IN], unit="inch"); self.dpi = QSpinBox(); self.dpi.setRange(72, 1200); self.dpi.setValue(300)
        self.pa, self.pb = FilePick(studio, "Photo A"), FilePick(studio, "Photo B"); self.same = QCheckBox("Use photo A twice (same photo in both places)"); self.same.setChecked(False)
        self.between = _spin(0, 50, 0.0); self.gap = _spin(0, 50, 3.0); self.margin = _spin(0, 60, 5.0); self.copies = QSpinBox(); self.copies.setRange(0, 500); self.copies.setSpecialValueText("max that fit")
        self.arr = QComboBox(); self.arr.addItems(["side", "stack", "auto"]); self.arr.setToolTip("side = two photos side by side; stack = one above the other; auto = whichever fits more")
        self.align = QComboBox(); self.align.addItems(["center", "top left", "top center"]); self.rot = QCheckBox("Allow turning duets 90° to fit more"); self.border = _spin(0, 2, 0.0, 2, " mm", 0.05)
        self.paper = PaperPick(studio)
        f.addRow("Duet size (both photos together)", self.size_e); f.addRow("Pixels at", self.dpi); f.addRow("Photo A", self.pa); f.addRow("Photo B", self.pb); f.addRow(self.same)
        f.addRow("Gap between the 2 photos (inside the duet size)", self.between); f.addRow("Gap between duets", self.gap); f.addRow("Page margins", self.margin); f.addRow("Number of duets", self.copies)
        f.addRow("Arrangement", self.arr); f.addRow("Alignment", self.align); f.addRow(self.rot); f.addRow("Image fit", self.fit); f.addRow("Thin border", self.border); f.addRow("Paper", self.paper)
        for w in (self.size_e, self.pa, self.pb, self.paper): w.changed.connect(self.update_preview)
        for w in (self.between, self.gap, self.margin, self.border, self.dpi, self.copies): w.valueChanged.connect(self.update_preview)
        for w in (self.arr, self.align, self.fit): w.currentTextChanged.connect(self.update_preview)
        for w in (self.same, self.rot): w.toggled.connect(self.update_preview)
        self.pb.changed.connect(lambda: self.same.setChecked(False) if self.pb.path else None)   # choosing Photo B means: use two different photos
        self.finish()
    def get_state(self):
        w, h = self.size_e.mm(); return dict(w=w, h=h, dpi=self.dpi.value(), between=self.between.value(), gap=self.gap.value(), margin=self.margin.value(), copies=self.copies.value(), arr=self.arr.currentText(), align=self.align.currentText(), fit=self.fit.currentText(), border=self.border.value())
    def set_state(self, s):
        self.size_e.set_mm(s["w"], s["h"]); self.dpi.setValue(s.get("dpi", 300)); self.between.setValue(s.get("between", 0)); self.gap.setValue(s.get("gap", 3)); self.margin.setValue(s.get("margin", 5)); self.copies.setValue(s.get("copies", 0))
        self.arr.setCurrentText(s.get("arr", "side")); self.align.setCurrentText(s.get("align", "center")); self.fit.setCurrentText(s.get("fit", "cover")); self.border.setValue(s.get("border", 0))
    def compute(self):
        dw, dh = self.size_e.mm(); W, H, paper, orient, _ = self.paper.size(); n = self.copies.value() or None; margin = self.margin.value()
        d = rc.duet_layout_total(dw, dh, W, H, self.gap.value(), margin, self.between.value(), None, self.arr.currentText(), self.rot.isChecked(), self.align.currentText().lower())
        pw, ph = d["photo"]; arr_used = d["arrangement"]
        cap = d["capacity"]; a = self.pa.path; b = a if (self.same.isChecked() or not self.pb.path) else self.pb.path; pages = []; left = n if n else cap
        wpx, hpx = rc.mm2px(dw, self.dpi.value()), rc.mm2px(dh, self.dpi.value()); ppx, qpx = rc.mm2px(pw, self.dpi.value()), rc.mm2px(ph, self.dpi.value())
        head = "Duet (2 photos together) %.2f × %.2f mm = %.2f × %.2f in = %d × %d px @ %d DPI\nEach photo: %.2f × %.2f mm = %d × %d px" % (dw, dh, rc.mm2in(dw), rc.mm2in(dh), wpx, hpx, self.dpi.value(), pw, ph, ppx, qpx)
        if cap == 0: return dict(pages=[], page=(W, H, paper, orient, margin), info=head + "\n⚠ Not even one duet fits on %.0f × %.0f mm." % (W, H))
        k = 1
        while left > 0:
            m = min(cap, left); dd = rc.duet_layout(pw, ph, W, H, self.gap.value(), margin, self.between.value(), m, arr_used, self.rot.isChecked(), self.align.currentText().lower()); pl = []
            for i, du in enumerate(dd["duets"]):
                for s, rect in enumerate(du): pl.append(dict(rect=rect, path=(a if s == 0 else b), rot=dd["rot"], name="Duet %d%s" % (k + i, "AB"[s])))
            pages.append(pl); k += m; left -= m
        return dict(pages=pages, page=(W, H, paper, orient, margin), border=self.border.value(),
                    info="%s\n%d duet(s) per page (%s), %d page(s); fits %d on %.0f × %.0f mm.%s" % (head, min(cap, n or cap), d["arrangement"], len(pages), cap, W, H, ("\nPhoto A + Photo B (2 different photos)" if (a and b != a) else ("\nPhoto A used in both places" if a else "\n(No photo chosen yet: grey boxes. Pick Photo A.)"))))

# =====================================================================
#  ID CARD STUDIO   (front + back on one page, many cards per page)
# =====================================================================
class IDCardPanel(LayoutPanel):
    KIND = "idcard"; TITLE = "ID card"
    def __init__(self, studio):
        super().__init__(studio); f = self.form
        self.size_e = SizeEdit(85.6, 54.0, "mm"); self.front, self.back = FilePick(studio, "Front"), FilePick(studio, "Back")
        self.gap = _spin(0, 100, 10.0); self.margin = _spin(0, 60, 15.0); self.cardgap = _spin(0, 60, 8.0); self.cards = QSpinBox(); self.cards.setRange(0, 100); self.cards.setValue(1); self.cards.setSpecialValueText("max that fit")
        self.arr = QComboBox(); self.arr.addItems(["stack", "side"]); self.arr.setToolTip("stack = back below front; side = back beside front"); self.border = _spin(0, 2, 0.2, 2, " mm", 0.05); self.rot = QCheckBox("Allow turning cards 90° to fit more")
        self.paper = PaperPick(studio)
        f.addRow("Card size", self.size_e); f.addRow("Front", self.front); f.addRow("Back", self.back); f.addRow("Gap front ↔ back", self.gap); f.addRow("Page margins", self.margin)
        f.addRow("Gap between card pairs", self.cardgap); f.addRow("Number of cards", self.cards); f.addRow("Arrangement", self.arr); f.addRow(self.rot); f.addRow("Image fit", self.fit); f.addRow("Cut line (border)", self.border); f.addRow("Paper", self.paper)
        self.fit.setCurrentText("stretch")
        for w in (self.size_e, self.front, self.back, self.paper): w.changed.connect(self.update_preview)
        for w in (self.gap, self.margin, self.cardgap, self.cards, self.border): w.valueChanged.connect(self.update_preview)
        self.arr.currentTextChanged.connect(self.update_preview); self.fit.currentTextChanged.connect(self.update_preview); self.rot.toggled.connect(self.update_preview)
        self.finish()
    def get_state(self):
        w, h = self.size_e.mm(); return dict(w=w, h=h, gap=self.gap.value(), margin=self.margin.value(), cardgap=self.cardgap.value(), cards=self.cards.value(), arr=self.arr.currentText(), border=self.border.value(), fit=self.fit.currentText())
    def set_state(self, s):
        self.size_e.set_mm(s["w"], s["h"]); self.gap.setValue(s.get("gap", 10)); self.margin.setValue(s.get("margin", 15)); self.cardgap.setValue(s.get("cardgap", 8)); self.cards.setValue(s.get("cards", 1)); self.arr.setCurrentText(s.get("arr", "stack")); self.border.setValue(s.get("border", 0.2)); self.fit.setCurrentText(s.get("fit", "stretch"))
    def compute(self):
        cw, ch = self.size_e.mm(); W, H, paper, orient, _ = self.paper.size(); margin = self.margin.value(); n = self.cards.value() or None
        r = rc.id_card_layout(cw, ch, W, H, self.gap.value(), margin, self.arr.currentText(), n, self.cardgap.value(), self.rot.isChecked())
        head = "Card %.1f × %.1f mm (%.2f × %.2f in)" % (cw, ch, rc.mm2in(cw), rc.mm2in(ch))
        if r["capacity"] == 0: return dict(pages=[], page=(W, H, paper, orient, margin), info=head + "\n⚠ Front + back do not fit on %.0f × %.0f mm with these margins." % (W, H))
        pl = []
        for i, (fr, bk) in enumerate(r["pairs"]):
            pl.append(dict(rect=fr, path=self.front.path, rot=r["rot"], name="Card %d front" % (i + 1))); pl.append(dict(rect=bk, path=self.back.path or self.front.path, rot=r["rot"], name="Card %d back" % (i + 1)))
        return dict(pages=[pl], page=(W, H, paper, orient, margin), border=self.border.value(), info="%s\n%d card(s) front+back on %.0f × %.0f mm (capacity %d)." % (head, len(r["pairs"]), W, H, r["capacity"]))

# =====================================================================
#  PASSPORT / PHOTO SHEET
# =====================================================================
class PassportPanel(LayoutPanel):
    KIND = "passport"; TITLE = "Photo sheet"
    def __init__(self, studio):
        super().__init__(studio); f = self.form
        self.size_e = SizeEdit(35, 45, "mm"); self.photo = FilePick(studio, "Photo"); self.copies = QSpinBox(); self.copies.setRange(1, 500); self.copies.setValue(8)
        self.gap = _spin(0, 30, 2.0); self.margin = _spin(0, 60, 5.0); self.align = QComboBox(); self.align.addItems(["top left", "center", "top center"]); self.rot = QCheckBox("Allow turning photos 90° to fit more"); self.border = _spin(0, 2, 0.1, 2, " mm", 0.05)
        self.paper = PaperPick(studio); self.maxb = QPushButton("Set copies = max that fit"); self.maxb.clicked.connect(self._max)
        f.addRow("Photo size", self.size_e); f.addRow("Photo", self.photo); f.addRow("Copies", self.copies); f.addRow(self.maxb); f.addRow("Spacing", self.gap); f.addRow("Page margins", self.margin); f.addRow("Alignment", self.align); f.addRow(self.rot); f.addRow("Image fit", self.fit); f.addRow("Thin border", self.border); f.addRow("Paper", self.paper)
        for w in (self.size_e, self.photo, self.paper): w.changed.connect(self.update_preview)
        for w in (self.gap, self.margin, self.copies, self.border): w.valueChanged.connect(self.update_preview)
        for w in (self.align, self.fit): w.currentTextChanged.connect(self.update_preview)
        self.rot.toggled.connect(self.update_preview); self.finish()
    def _max(self):
        pw, ph = self.size_e.mm(); W, H, *_ = self.paper.size(); self.copies.setValue(max(1, rc.pack_grid(pw, ph, W, H, self.margin.value(), self.gap.value(), self.rot.isChecked())["capacity"]))
    def get_state(self):
        w, h = self.size_e.mm(); return dict(w=w, h=h, copies=self.copies.value(), gap=self.gap.value(), margin=self.margin.value(), align=self.align.currentText(), fit=self.fit.currentText(), border=self.border.value())
    def set_state(self, s):
        self.size_e.set_mm(s["w"], s["h"]); self.copies.setValue(s.get("copies", 8)); self.gap.setValue(s.get("gap", 2)); self.margin.setValue(s.get("margin", 5)); self.align.setCurrentText(s.get("align", "top left")); self.fit.setCurrentText(s.get("fit", "cover")); self.border.setValue(s.get("border", 0.1))
    def compute(self):
        pw, ph = self.size_e.mm(); W, H, paper, orient, _ = self.paper.size(); margin = self.margin.value()
        r = rc.passport_sheet(pw, ph, W, H, self.copies.value(), self.gap.value(), margin, self.rot.isChecked(), self.align.currentText())
        head = "Photo %.1f × %.1f mm (%.2f × %.2f in)" % (pw, ph, rc.mm2in(pw), rc.mm2in(ph))
        if not r["fits"]: return dict(pages=[], page=(W, H, paper, orient, margin), info=head + "\n⚠ One photo does not fit on this page.")
        rot = rc.pack_grid(pw, ph, W, H, margin, self.gap.value(), self.rot.isChecked())["rot"]; pages = []; k = 1
        for items in r["pages"]:
            pl = []
            for rect in items: pl.append(dict(rect=rect, path=self.photo.path, rot=rot, name="Photo %d" % k)); k += 1
            pages.append(pl)
        return dict(pages=pages, page=(W, H, paper, orient, margin), border=self.border.value(), info="%s\n%d copies on %d page(s); %d fit per page on %.0f × %.0f mm." % (head, self.copies.value(), len(pages), r["capacity"], W, H))

# =====================================================================
#  PRINT & EXPORT
# =====================================================================
class PrintPanel(QWidget):
    def __init__(self, studio):
        super().__init__(); self.studio = studio; lay = QVBoxLayout(self)
        self.info = QLabel("Everything is printed at its exact physical size: the page is sent full-size with zero margins and NO fit-to-page scaling.\nIn the printer dialog choose the same paper size and 'Actual size / 100%'."); self.info.setWordWrap(True); lay.addWidget(self.info)
        g = QGroupBox("Print"); gl = QVBoxLayout(g)
        for t, fn in (("Print preview (real size)...", lambda: studio.print_dialog(True)), ("Print... (printer, pages, copies)", lambda: studio.print_dialog(False))): b = QPushButton(t); b.clicked.connect(lambda _=False, f=fn: f()); gl.addWidget(b)
        lay.addWidget(g); g = QGroupBox("Export"); gl = QVBoxLayout(g)
        for t, fmt in (("Export JPG...", "jpg"), ("Export PNG...", "png"), ("Export multi-page PDF...", "pdf")): b = QPushButton(t); b.clicked.connect(lambda _=False, f=fmt: studio.export_dialog(f)); gl.addWidget(b)
        lay.addWidget(g); g = QGroupBox("Resolution of the selected image(s)"); gl = QVBoxLayout(g)
        self.rinfo = QLabel("Select a layer to see its effective DPI."); self.rinfo.setWordWrap(True); gl.addWidget(self.rinfo)
        r = QHBoxLayout(); self.target = QSpinBox(); self.target.setRange(100, 1200); self.target.setValue(300); self.target.setSuffix(" DPI"); b = QPushButton("Resample to this DPI"); b.clicked.connect(self.resample); r.addWidget(self.target); r.addWidget(b); gl.addLayout(r)
        n = QLabel("Resampling makes a NEW file with exactly the pixels needed for the printed size. It does not add real detail; a blurry photo stays blurry. The original is kept (Tools > Revert)."); n.setWordWrap(True); gl.addWidget(n)
        b = QPushButton("Refresh DPI info"); b.clicked.connect(self.update_info); gl.addWidget(b); lay.addWidget(g); lay.addStretch(1)
    def update_info(self):
        ls = self.studio.selected_layers()
        if not ls: self.rinfo.setText("Select a layer to see its effective DPI."); return
        rows = []
        for L in ls[:8]:
            try:
                from PySide6.QtGui import QImageReader
                s = QImageReader(L.path).size(); eff = max(s.width(), s.height()) / (max(L.w_mm, L.h_mm) / 25.4) if s.isValid() else 0
                rows.append("%s: %d × %d px → %.0f DPI at %.1f × %.1f mm%s" % (L.name, s.width(), s.height(), eff, L.w_mm, L.h_mm, "  ⚠ low" if eff < 150 else ""))
            except Exception: rows.append("%s: unknown" % L.name)
        self.rinfo.setText("\n".join(rows))
    def resample(self):
        ls = self.studio.selected_layers()
        if not ls: return QMessageBox.information(self, "Resample", "Select one or more layers first.")
        dpi = self.target.value(); st = self.studio; st.push_undo(); done = 0
        for L in ls:
            try:
                im = pil_open(L.path)
                if L.crop: W, H = im.size; l, t, r, b = L.crop; im = im.crop((int(l * W), int(t * H), int(r * W), int(b * H)))
                tw, th = max(1, rc.mm2px(L.w_mm, dpi)), max(1, rc.mm2px(L.h_mm, dpi))
                if (tw, th) == im.size: continue
                im = im.convert("RGB").resize((tw, th), Image.LANCZOS if tw * th < im.size[0] * im.size[1] else Image.BICUBIC)
                p = unique_path(os.path.join(CACHE_DIR, "%s_%ddpi.png" % (os.path.splitext(os.path.basename(L.path))[0], dpi))); im.save(p, dpi=(dpi, dpi))
                if "orig_path" not in L.adjust: L.adjust["orig_path"] = L.path; L.adjust["orig_crop"] = list(L.crop) if L.crop else None
                L.path = p; L.crop = None; done += 1
            except Exception as e: log.warning("resample failed %s: %s", L.name, e)
        st.cache.clear(); st.refresh_page_scene(); QMessageBox.information(self, "Resample", "%d layer(s) resampled to %d DPI at their current physical size." % (done, dpi))

def install_panels(studio, tabs):
    panels = []
    for cls, title in ((DuetPanel, "Bank Photo Duet"), (IDCardPanel, "ID Card"), (PassportPanel, "Passport Sheet"), (PrintPanel, "Print && Export")):
        p = cls(studio); tabs.addTab(p, title.replace("&&", "&")); panels.append(p)
    def page_changed():
        for p in panels:
            if hasattr(p, "update_preview") and p.isVisible(): p.update_preview()
            if isinstance(p, PrintPanel): p.update_info()
    studio.panels_page_changed = page_changed; studio.panels = panels
    tabs.currentChanged.connect(lambda i: page_changed())
