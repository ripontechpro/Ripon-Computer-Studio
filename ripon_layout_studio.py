# -*- coding: utf-8 -*-
"""
RIPON LAYOUT STUDIO  -  Photoshop-style multi-image / multi-page editor + print layout.
Keep next to Photostudio.py (+ ripon_core.py, ripon_studio_base.py, ripon_studio_tools.py, ripon_studio_panels.py).
Menu in the main program:  Studio > Layout Studio  (Ctrl+Shift+L).   Standalone:  python ripon_layout_studio.py

Everything is stored in millimetres (physical size). Zoom only changes the view, never the print size.
Original image files are only ever read. Cropped / restored versions are NEW files in the layout cache folder.
"""
import os, sys, json, math, copy, time, re, tempfile, traceback
from PIL import Image
from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, QTimer, QSize, QMimeData, QPoint, QMarginsF
from PySide6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QBrush, QIcon, QAction, QKeySequence, QPageSize,
                           QPageLayout, QFont, QImageReader, QDrag)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QTabWidget,
                               QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsRectItem, QDockWidget, QListWidget, QListWidgetItem,
                               QAbstractItemView, QFormLayout, QDoubleSpinBox, QCheckBox, QComboBox, QSlider, QFileDialog, QMessageBox,
                               QInputDialog, QDialog, QDialogButtonBox, QSpinBox, QLineEdit, QColorDialog, QMenu, QToolBar, QStatusBar,
                               QScrollArea, QFrame)
from PySide6.QtPrintSupport import QPrinter, QPrintDialog, QPrintPreviewDialog
import ripon_core as rc
from ripon_studio_base import (SC, APP_DIR, CACHE_DIR, RECOVERY_DIR, IMAGE_FILTER, IMAGE_EXTS, ImageCache, pil_open, pil_to_qimage, qimage_to_pil,
                               run_task, unique_path, log)

LAYER_MIME = "application/x-ripon-layer-ids"

# =====================================================================
#  CANVAS ITEMS
# =====================================================================
class Handle(QGraphicsRectItem):
    """resize (bottom-right) and rotate (above top-centre) handle; constant size on screen"""
    def __init__(self, kind, parent):
        super().__init__(QRectF(-6, -6, 12, 12), parent); self.kind = kind
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setBrush(QBrush(QColor(255, 255, 255) if kind == "resize" else QColor(255, 200, 0)))
        self.setPen(QPen(QColor(0, 120, 215), 1.5)); self.setZValue(10 ** 6)
        self.setCursor(Qt.SizeFDiagCursor if kind == "resize" else Qt.CrossCursor)
        self.setToolTip("Drag to resize (Shift = free aspect)" if kind == "resize" else "Drag to rotate (Shift = 15° steps)")
    def mousePressEvent(self, e):
        self.parentItem().studio.begin_gesture(); e.accept()
    def mouseMoveEvent(self, e):
        self.parentItem().handle_drag(self.kind, e.scenePos(), e.modifiers()); e.accept()
    def mouseReleaseEvent(self, e):
        self.parentItem().studio.end_gesture(); e.accept()

class LayerItem(QGraphicsItem):
    def __init__(self, studio, layer):
        super().__init__(); self.studio = studio; self.layer = layer; self._resizing = False
        self.setFlag(QGraphicsItem.ItemIsSelectable, True); self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True); self.setAcceptHoverEvents(True)
        self.h_resize = Handle("resize", self); self.h_rot = Handle("rot", self)
        self.sync()
    def boundingRect(self):
        L = self.layer; return QRectF(-2, -2, L.w_mm * SC + 4, L.h_mm * SC + 4)
    def sync(self):
        """model -> item"""
        L = self.layer; self.prepareGeometryChange(); self._resizing = True
        self.setTransformOriginPoint(L.w_mm * SC / 2.0, L.h_mm * SC / 2.0)
        self.setPos(L.x_mm * SC, L.y_mm * SC); self.setRotation(L.rot); self.setOpacity(L.opacity); self.setVisible(L.visible)
        self.setFlag(QGraphicsItem.ItemIsMovable, not L.locked); self._resizing = False
        self.h_resize.setPos(L.w_mm * SC, L.h_mm * SC); self.h_rot.setPos(L.w_mm * SC / 2.0, -14 / max(0.05, self._zoom()))
        self.update_handles(); self.update()
    def _zoom(self):
        try: return self.scene().views()[0].transform().m11() * 1.0
        except Exception: return 1.0
    def update_handles(self):
        sc = self.scene(); show = bool(sc and self.isSelected() and not self.layer.locked and len(sc.selectedItems()) == 1)
        self.h_resize.setVisible(show); self.h_rot.setVisible(show)
        if show: self.h_rot.setPos(self.layer.w_mm * SC / 2.0, -22.0 / max(0.05, self._zoom()))
    def pull(self):
        """item -> model"""
        L = self.layer; L.x_mm, L.y_mm = self.pos().x() / SC, self.pos().y() / SC; L.rot = self.rotation()
    def paint(self, p, opt, widget=None):
        L = self.layer; r = QRectF(0, 0, L.w_mm * SC, L.h_mm * SC); img = self.studio.display_image(L)
        if img is None:
            p.fillRect(r, QColor(220, 80, 80, 70)); p.setPen(QColor(160, 0, 0)); p.drawRect(r)
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, "Missing file\n" + os.path.basename(L.path or L.name)); return
        W, H = img.width(), img.height(); l, t, rr, b = L.crop or (0.0, 0.0, 1.0, 1.0)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True); p.save()
        if L.flip_h or L.flip_v:
            p.translate(r.center()); p.scale(-1 if L.flip_h else 1, -1 if L.flip_v else 1); p.translate(-r.center())
        p.drawImage(r, img, QRectF(l * W, t * H, (rr - l) * W, (b - t) * H)); p.restore()
        if L.border_mm > 0: p.setPen(QPen(QColor(0, 0, 0), L.border_mm * SC)); p.setBrush(Qt.NoBrush); p.drawRect(r)
        if self.isSelected():
            p.setPen(QPen(QColor(0, 120, 215), 0, Qt.DashLine)); p.setBrush(Qt.NoBrush); p.drawRect(r)
        if L.locked and self.isSelected():
            p.setPen(QColor(200, 0, 0)); p.drawText(QPointF(4, 14), "🔒")
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not self._resizing and self.scene() is not None:
            return self.studio.snap_pos(self, value)
        return super().itemChange(change, value)
    def handle_drag(self, kind, scene_pos, mods):
        L = self.layer
        if kind == "resize":
            p = self.mapFromScene(scene_pos); nw = max(2.0, p.x() / SC); nh = max(2.0, p.y() / SC)
            if self.studio.keep_aspect and not (mods & Qt.ShiftModifier): nh = nw * (L.h_mm / L.w_mm)
            old = self.mapToScene(QPointF(0, 0)); self._resizing = True
            self.prepareGeometryChange(); L.w_mm, L.h_mm = nw, nh
            self.setTransformOriginPoint(nw * SC / 2.0, nh * SC / 2.0)
            new = self.mapToScene(QPointF(0, 0)); self.setPos(self.pos() + (old - new)); self._resizing = False
            self.h_resize.setPos(nw * SC, nh * SC); self.h_rot.setPos(nw * SC / 2.0, -22.0 / max(0.05, self._zoom())); self.pull(); self.update()
        else:
            c = self.mapToScene(QPointF(L.w_mm * SC / 2.0, L.h_mm * SC / 2.0))
            ang = math.degrees(math.atan2(scene_pos.y() - c.y(), scene_pos.x() - c.x())) + 90.0
            if mods & Qt.ShiftModifier: ang = round(ang / 15.0) * 15.0
            ang = (ang + 180.0) % 360.0 - 180.0; self.setRotation(ang); self.pull()
        self.studio.props_from_selection()

class SelHandle(QGraphicsRectItem):
    """bottom-right handle of the whole multi-selection: drag to scale every selected layer together (about the selection's top-left)"""
    def __init__(self, scene):
        super().__init__(QRectF(-7, -7, 14, 14)); self.sc = scene; self.base = None
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True); self.setBrush(QBrush(QColor(255, 140, 0))); self.setPen(QPen(QColor(255, 255, 255), 1.5)); self.setZValue(10 ** 7)
        self.setCursor(Qt.SizeFDiagCursor); self.setToolTip("Drag to scale the whole selection"); self.setVisible(False)
    def mousePressEvent(self, e):
        st = self.sc.studio; ls = st.selected_layers()
        if len(ls) < 2: return
        st.begin_gesture(); bs = [rc.rect_bounds_rotated(l.x_mm, l.y_mm, l.w_mm, l.h_mm, l.rot) for l in ls]
        self.base = dict(box=(min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)),
                         items=[(l, l.x_mm + l.w_mm / 2.0, l.y_mm + l.h_mm / 2.0, l.w_mm, l.h_mm) for l in ls if not l.locked]); e.accept()
    def mouseMoveEvent(self, e):
        if not self.base: return
        x0, y0, x1, y1 = self.base["box"]; k = max(0.05, (e.scenePos().x() / SC - x0) / max(1e-6, (x1 - x0)))
        for L, cx, cy, w, h in self.base["items"]:
            L.w_mm, L.h_mm = w * k, h * k; ncx, ncy = x0 + (cx - x0) * k, y0 + (cy - y0) * k; L.x_mm, L.y_mm = ncx - L.w_mm / 2.0, ncy - L.h_mm / 2.0
            it = self.sc.items_by_id.get(L.id)
            if it: it.sync()
        self.sc.update_sel_handle(keep_pos=False); e.accept()
    def mouseReleaseEvent(self, e):
        self.base = None; self.sc.studio.end_gesture(); self.sc.update_sel_handle(); e.accept()

class PageScene(QGraphicsScene):
    def __init__(self, studio, page):
        super().__init__(studio); self.studio = studio; self.page = page; self.items_by_id = {}
        m = 25 * SC; self.setSceneRect(-m, -m, page.w_mm * SC + 2 * m, page.h_mm * SC + 2 * m); self.setBackgroundBrush(QColor(90, 90, 96))
        sh = QGraphicsRectItem(2 * SC, 2 * SC, page.w_mm * SC, page.h_mm * SC); sh.setBrush(QColor(0, 0, 0, 90)); sh.setPen(Qt.NoPen); sh.setZValue(-2000); self.addItem(sh)
        self.paper = QGraphicsRectItem(0, 0, page.w_mm * SC, page.h_mm * SC); self.paper.setBrush(QColor(*page.bg)); self.paper.setPen(QPen(QColor(40, 40, 40), 0)); self.paper.setZValue(-1000); self.addItem(self.paper)
        self.sel_handle = SelHandle(self); self.addItem(self.sel_handle)
        self.selectionChanged.connect(self._sel)
        self.rebuild()
    def rebuild(self):
        for it in list(self.items_by_id.values()): self.removeItem(it)
        self.items_by_id = {}
        for z, L in enumerate(self.page.layers):
            it = LayerItem(self.studio, L); it.setZValue(z); self.addItem(it); self.items_by_id[L.id] = it
    def update_sel_handle(self, keep_pos=False):
        ls = [it.layer for it in self.items_by_id.values() if it.isSelected()]
        if len(ls) < 2: self.sel_handle.setVisible(False); return
        bs = [rc.rect_bounds_rotated(l.x_mm, l.y_mm, l.w_mm, l.h_mm, l.rot) for l in ls]
        self.sel_handle.setPos(max(b[2] for b in bs) * SC, max(b[3] for b in bs) * SC); self.sel_handle.setVisible(True)
    def _sel(self):
        for it in self.items_by_id.values(): it.update_handles()
        self.update_sel_handle(); self.studio.scene_selection_changed(self)
    def drawForeground(self, p, rect):
        st, pg = self.studio, self.page
        if st.show_grid and st.grid_mm > 0:
            p.setPen(QPen(QColor(0, 150, 255, 70), 0)); g = st.grid_mm
            x = 0.0
            while x <= pg.w_mm + 1e-6: p.drawLine(QPointF(x * SC, 0), QPointF(x * SC, pg.h_mm * SC)); x += g
            y = 0.0
            while y <= pg.h_mm + 1e-6: p.drawLine(QPointF(0, y * SC), QPointF(pg.w_mm * SC, y * SC)); y += g
        if st.show_margins and pg.margin_mm > 0:
            p.setPen(QPen(QColor(255, 90, 90, 160), 0, Qt.DashLine)); m = pg.margin_mm * SC
            p.drawRect(QRectF(m, m, pg.w_mm * SC - 2 * m, pg.h_mm * SC - 2 * m))
        p.setPen(QPen(QColor(0, 200, 220), 0))
        for g in pg.guides_h: p.drawLine(QPointF(-25 * SC, g * SC), QPointF((pg.w_mm + 25) * SC, g * SC))
        for g in pg.guides_v: p.drawLine(QPointF(g * SC, -25 * SC), QPointF(g * SC, (pg.h_mm + 25) * SC))
        pv = st.preview_guide
        if pv and pv[2] is self.page:
            p.setPen(QPen(QColor(255, 255, 0), 0, Qt.DashLine))
            if pv[0] == "h": p.drawLine(QPointF(-25 * SC, pv[1] * SC), QPointF((pg.w_mm + 25) * SC, pv[1] * SC))
            else: p.drawLine(QPointF(pv[1] * SC, -25 * SC), QPointF(pv[1] * SC, (pg.h_mm + 25) * SC))

class CanvasView(QGraphicsView):
    def __init__(self, studio, scene):
        super().__init__(scene); self.studio = studio; self._space = False
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform); self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse); self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate); self.setFrameShape(QFrame.NoFrame); self.setAcceptDrops(True)
        self.setMouseTracking(True); self.setFocusPolicy(Qt.StrongFocus)
        self.horizontalScrollBar().valueChanged.connect(lambda *_: studio.update_rulers()); self.verticalScrollBar().valueChanged.connect(lambda *_: studio.update_rulers())
    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            self.scale_by(1.15 if e.angleDelta().y() > 0 else 1 / 1.15); e.accept()
        else: super().wheelEvent(e)
    def scale_by(self, f):
        cur = self.transform().m11(); new = max(0.02, min(60.0, cur * f)); f = new / cur; self.scale(f, f)
        self.studio.update_rulers(); self.studio.status_zoom()
    def resizeEvent(self, e): super().resizeEvent(e); self.studio.update_rulers()
    def mousePressEvent(self, e):
        self.studio.begin_gesture(); super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        row = self.studio.page_row_at(e.globalPosition().toPoint())
        ids = [l.id for l in self.studio.selected_layers()]
        if row >= 0 and row != self.studio.cur_index() and ids:           # dropped on another page's thumbnail: send the layers there
            self.studio._gesture_pre = None; self.studio.send_layers_to_page(ids, row, copy_=bool(e.modifiers() & Qt.ControlModifier)); return
        self.studio.end_gesture()
    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e); p = self.mapToScene(e.position().toPoint()); self.studio.status_cursor(p.x() / SC, p.y() / SC)
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True; self.setDragMode(QGraphicsView.ScrollHandDrag); return
        step = 10.0 if e.modifiers() & Qt.ShiftModifier else (0.1 if e.modifiers() & Qt.ControlModifier else 1.0)
        d = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0), Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}.get(e.key())
        if d: self.studio.nudge(*d); return
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace): self.studio.delete_selected(); return
        super().keyPressEvent(e)
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat() and self._space:
            self._space = False; self.setDragMode(QGraphicsView.RubberBandDrag); return
        super().keyReleaseEvent(e)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile().lower().endswith(IMAGE_EXTS)]
        if paths:
            p = self.mapToScene(e.position().toPoint()); self.studio.add_images(paths, at_mm=(p.x() / SC, p.y() / SC)); e.acceptProposedAction()

class Ruler(QWidget):
    def __init__(self, studio, orient):
        super().__init__(); self.studio = studio; self.orient = orient; self.setFixedHeight(20) if orient == "h" else self.setFixedWidth(20)
        self._drag = False; self.setCursor(Qt.SplitVCursor if orient == "h" else Qt.SplitHCursor); self.setToolTip("Drag onto the page to create a guide")
    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor(60, 60, 66)); v = self.studio.cur_view()
        if v is None: return
        ppm = v.transform().m11() * SC
        if ppm <= 0: return
        step = next((s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500) if s * ppm >= 7), 500); major = step * 5 if step in (1, 2) else step * 2
        p.setPen(QColor(200, 200, 200)); f = QFont(); f.setPixelSize(9); p.setFont(f)
        if self.orient == "h":
            a, b = v.mapToScene(0, 0).x() / SC, v.mapToScene(v.viewport().width(), 0).x() / SC
            m = math.floor(a / step) * step
            while m <= b:
                x = v.mapFromScene(QPointF(m * SC, 0)).x(); big = abs(m % major) < 1e-6
                p.drawLine(x, 20 - (12 if big else 5), x, 20)
                if big: p.drawText(x + 2, 9, "%g" % m)
                m += step
        else:
            a, b = v.mapToScene(0, 0).y() / SC, v.mapToScene(0, v.viewport().height()).y() / SC
            m = math.floor(a / step) * step
            while m <= b:
                y = v.mapFromScene(QPointF(0, m * SC)).y(); big = abs(m % major) < 1e-6
                p.drawLine(20 - (12 if big else 5), y, 20, y)
                if big: p.save(); p.translate(9, y - 2); p.rotate(-90); p.drawText(0, 0, "%g" % m); p.restore()
                m += step
    def _mm(self, ev):
        v = self.studio.cur_view(); gp = ev.globalPosition().toPoint(); vp = v.viewport().mapFromGlobal(gp); s = v.mapToScene(vp)
        return (s.y() / SC if self.orient == "h" else s.x() / SC), v.viewport().rect().contains(vp)
    def mousePressEvent(self, e): self._drag = True
    def mouseMoveEvent(self, e):
        if not self._drag or self.studio.cur_page() is None: return
        mm, _ = self._mm(e); self.studio.preview_guide = (self.orient, mm, self.studio.cur_page()); self.studio.cur_scene().update()
    def mouseReleaseEvent(self, e):
        if not self._drag: return
        self._drag = False; mm, inside = self._mm(e); self.studio.preview_guide = None
        if inside: self.studio.add_guide(self.orient, mm)
        self.studio.cur_scene().update()

class LayerList(QListWidget):
    """layer panel list; dragging selected rows onto a page thumbnail moves them there (Ctrl = copy)"""
    def __init__(self, studio):
        super().__init__(); self.studio = studio; self.setSelectionMode(QAbstractItemView.ExtendedSelection); self.setDragEnabled(True)
    def mimeData(self, items):
        m = QMimeData(); m.setData(LAYER_MIME, ",".join(i.data(Qt.UserRole) for i in items).encode("utf-8")); return m
    def startDrag(self, actions):
        d = QDrag(self); d.setMimeData(self.mimeData(self.selectedItems())); d.exec(Qt.MoveAction | Qt.CopyAction, Qt.MoveAction)

class PageList(QListWidget):
    def __init__(self, studio):
        super().__init__(); self.studio = studio; self.setViewMode(QListWidget.IconMode); self.setIconSize(QSize(96, 128)); self.setMovement(QListWidget.Static)
        self.setResizeMode(QListWidget.Adjust); self.setSpacing(6); self.setAcceptDrops(True); self.setDragDropMode(QAbstractItemView.DropOnly)
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(LAYER_MIME): e.acceptProposedAction()
    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(LAYER_MIME): e.acceptProposedAction()
    def dropEvent(self, e):
        it = self.itemAt(e.position().toPoint())
        if it is None or not e.mimeData().hasFormat(LAYER_MIME): return
        ids = bytes(e.mimeData().data(LAYER_MIME)).decode("utf-8").split(",")
        self.studio.send_layers_to_page(ids, self.row(it), copy_=bool(e.modifiers() & Qt.ControlModifier)); e.acceptProposedAction()

# =====================================================================
#  PAGE SETUP DIALOG
# =====================================================================
class PageSetupDialog(QDialog):
    def __init__(self, parent, page=None, title="Page setup"):
        super().__init__(parent); self.setWindowTitle(title); self.bg = tuple(page.bg) if page else (255, 255, 255)
        self.paper = QComboBox(); self.paper.addItems(rc.PAPER_NAMES); self.orient = QComboBox(); self.orient.addItems(["Portrait", "Landscape"])
        self.cw = QDoubleSpinBox(); self.ch = QDoubleSpinBox()
        for s in (self.cw, self.ch): s.setRange(20, 3000); s.setDecimals(1); s.setSuffix(" mm")
        self.margin = QDoubleSpinBox(); self.margin.setRange(0, 100); self.margin.setSuffix(" mm"); self.margin.setDecimals(1)
        self.name = QLineEdit(page.name if page else "Page")
        self.bgb = QPushButton("Background colour"); self.bgb.clicked.connect(self._pick); self._paint()
        self.all_pages = QCheckBox("Apply size to ALL pages"); self.all_pages.setVisible(page is not None)
        if page:
            self.paper.setCurrentText(page.paper if page.paper in rc.PAPER_NAMES else "Custom"); self.orient.setCurrentText(page.orientation)
            self.cw.setValue(page.w_mm); self.ch.setValue(page.h_mm); self.margin.setValue(page.margin_mm)
        else: self.paper.setCurrentText("A4"); self.margin.setValue(5.0); self._fill()
        self.paper.currentTextChanged.connect(self._fill); self.orient.currentTextChanged.connect(self._fill)
        f = QFormLayout(self)
        for a, b in (("Name", self.name), ("Paper", self.paper), ("Orientation", self.orient), ("Custom width", self.cw), ("Custom height", self.ch), ("Margins", self.margin)): f.addRow(a, b)
        f.addRow(self.bgb); f.addRow(self.all_pages)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); f.addRow(bb)
    def _fill(self, *_):
        if self.paper.currentText() != "Custom":
            w, h = rc.page_size_mm(self.paper.currentText(), self.orient.currentText()); self.cw.setValue(w); self.ch.setValue(h)
        else:
            w, h = self.cw.value(), self.ch.value()
            if (self.orient.currentText() == "Landscape") != (w > h): self.cw.setValue(h); self.ch.setValue(w)
        self.cw.setEnabled(self.paper.currentText() == "Custom"); self.ch.setEnabled(self.paper.currentText() == "Custom")
    def _paint(self): self.bgb.setStyleSheet("background: rgb(%d,%d,%d); padding:4px;" % self.bg)
    def _pick(self):
        c = QColorDialog.getColor(QColor(*self.bg), self)
        if c.isValid(): self.bg = (c.red(), c.green(), c.blue()); self._paint()
    def values(self):
        w, h = self.cw.value(), self.ch.value(); return dict(name=self.name.text().strip() or "Page", paper=self.paper.currentText(), orientation=self.orient.currentText(),
                                                           w=w, h=h, margin=self.margin.value(), bg=self.bg, all=self.all_pages.isChecked())

def parse_range(text, n):
    """'1-3,5' -> [0,1,2,4] (1-based input); empty = all"""
    text = (text or "").strip()
    if not text: return list(range(n))
    out = []
    for part in re.split(r"[,\s]+", text):
        if not part: continue
        m = re.match(r"^(\d+)(?:-(\d+))?$", part)
        if not m: raise ValueError("Bad page range: " + part)
        a = int(m.group(1)); b = int(m.group(2) or a)
        for i in range(min(a, b), max(a, b) + 1):
            if 1 <= i <= n and i - 1 not in out: out.append(i - 1)
    if not out: raise ValueError("No such page")
    return out

# =====================================================================
#  MAIN WINDOW
# =====================================================================
class LayoutStudio(QMainWindow):
    def __init__(self, host=None):
        super().__init__(); self.host = host; self.setWindowTitle("Ripon Layout Studio"); self.resize(1500, 920)
        self.project = rc.Project("Untitled"); self.path = None; self.dirty = False; self.embed = True
        self.cache = ImageCache(); self.scenes = {}; self.views = {}; self.undo_stack = []; self.redo_stack = []; self.clip = []
        self.show_grid = False; self.snap_grid = False; self.grid_mm = 5.0; self.show_margins = True; self.keep_aspect = True; self.preview_guide = None
        self._gesture_pre = None; self._sync = False; self._last_push = (None, 0.0); self._view_state = {}; self._tasks = []; self._adj_cache = {}
        self.autosaver = rc.Autosaver(RECOVERY_DIR); self.cache_dir = CACHE_DIR
        self._build_ui(); self._build_menus(); self.rebuild_all()
        self.thumb_timer = QTimer(self); self.thumb_timer.setSingleShot(True); self.thumb_timer.setInterval(350); self.thumb_timer.timeout.connect(self.refresh_thumbs)
        self.auto_timer = QTimer(self); self.auto_timer.setInterval(30000); self.auto_timer.timeout.connect(self._autosave); self.auto_timer.start()
        QTimer.singleShot(400, self._offer_recovery)

    # ---------------- UI ----------------
    def _build_ui(self):
        self.tabs = QTabWidget(); self.tabs.setTabsClosable(False); self.tabs.setMovable(True); self.tabs.currentChanged.connect(self._tab_changed); self.tabs.tabBar().tabMoved.connect(self._tab_moved)
        self.rh, self.rv = Ruler(self, "h"), Ruler(self, "v"); box = QWidget(); g = QGridLayout(box); g.setContentsMargins(0, 0, 0, 0); g.setSpacing(0)
        corner = QWidget(); corner.setFixedSize(20, 20); corner.setStyleSheet("background:#3c3c42;")
        self.view_holder = QWidget(); self.vh = QVBoxLayout(self.view_holder); self.vh.setContentsMargins(0, 0, 0, 0)
        g.addWidget(corner, 0, 0); g.addWidget(self.rh, 0, 1); g.addWidget(self.rv, 1, 0); g.addWidget(self.tabs, 1, 1); g.setRowStretch(1, 1); g.setColumnStretch(1, 1)
        self.setCentralWidget(box)
        # pages dock
        self.page_list = PageList(self); self.page_list.currentRowChanged.connect(lambda r: self.tabs.setCurrentIndex(r) if 0 <= r < self.tabs.count() else None)
        pw = QWidget(); pl = QVBoxLayout(pw); pl.addWidget(self.page_list)
        row = QHBoxLayout()
        for t, fn in (("+", self.page_new), ("Copy", self.page_duplicate), ("Rename", self.page_rename), ("◀", lambda: self.page_move(-1)), ("▶", lambda: self.page_move(1)), ("Del", self.page_delete)):
            b = QPushButton(t); b.clicked.connect(lambda _=False, f=fn: f()); b.setMaximumWidth(56); row.addWidget(b)
        pl.addLayout(row); self.dock_pages = self._dock("Pages", pw, Qt.LeftDockWidgetArea)
        # layers dock
        self.layer_list = LayerList(self); self.layer_list.itemSelectionChanged.connect(self._layer_list_selected); self.layer_list.itemChanged.connect(self._layer_item_changed)
        lw = QWidget(); ll = QVBoxLayout(lw); ll.addWidget(self.layer_list); r1 = QHBoxLayout(); r2 = QHBoxLayout()
        for t, fn in (("Duplicate", self.duplicate_selected), ("Delete", self.delete_selected), ("Lock", self.toggle_lock)): b = QPushButton(t); b.clicked.connect(lambda _=False, f=fn: f()); r1.addWidget(b)
        for t, fn in (("▲ Up", lambda: self.reorder(1)), ("▼ Down", lambda: self.reorder(-1)), ("Rename", self.rename_layer)): b = QPushButton(t); b.clicked.connect(lambda _=False, f=fn: f()); r2.addWidget(b)
        ll.addLayout(r1); ll.addLayout(r2); ll.addWidget(QLabel("Drag layers onto a page thumbnail to move them (Ctrl = copy)."))
        self.dock_layers = self._dock("Layers", lw, Qt.RightDockWidgetArea)
        # properties dock
        self.sp = {}; form = QFormLayout(); pw2 = QWidget(); pw2.setLayout(form)
        self.unit = QComboBox(); self.unit.addItems(["mm", "inch"]); self.unit.currentTextChanged.connect(self._unit_changed); form.addRow("Units", self.unit)
        for k, t in (("x", "X"), ("y", "Y"), ("w", "Width"), ("h", "Height")):
            s = QDoubleSpinBox(); s.setRange(-5000, 5000); s.setDecimals(2); s.setKeyboardTracking(False); s.valueChanged.connect(self._prop_edited); self.sp[k] = s; form.addRow(t, s)
        s = QDoubleSpinBox(); s.setRange(-360, 360); s.setDecimals(1); s.setSuffix("°"); s.setKeyboardTracking(False); s.valueChanged.connect(self._prop_edited); self.sp["rot"] = s; form.addRow("Rotation", s)
        s = QSlider(Qt.Horizontal); s.setRange(0, 100); s.valueChanged.connect(self._prop_edited); self.sp["op"] = s; form.addRow("Opacity", s)
        for k, t in (("bri", "Brightness"), ("con", "Contrast")):
            sl = QSlider(Qt.Horizontal); sl.setRange(-100, 100); sl.valueChanged.connect(self._prop_edited); self.sp[k] = sl; form.addRow(t, sl)
        self.c_gray = QCheckBox("Black && white (grey)"); self.c_gray.toggled.connect(self._prop_edited); form.addRow(self.c_gray)
        self.c_aspect = QCheckBox("Keep aspect ratio when resizing"); self.c_aspect.setChecked(True); self.c_aspect.toggled.connect(lambda v: setattr(self, "keep_aspect", v)); form.addRow(self.c_aspect)
        self.c_flip_h = QCheckBox("Flip horizontal"); self.c_flip_v = QCheckBox("Flip vertical"); self.c_flip_h.toggled.connect(self._prop_edited); self.c_flip_v.toggled.connect(self._prop_edited)
        form.addRow(self.c_flip_h); form.addRow(self.c_flip_v)
        self.lbl_px = QLabel(""); self.lbl_px.setWordWrap(True); form.addRow(self.lbl_px)
        self.dock_props = self._dock("Properties", pw2, Qt.RightDockWidgetArea)
        # tools dock (tabs from ripon_studio_panels.py)
        self.tools_tabs = QTabWidget()
        try:
            import ripon_studio_panels as panels
            panels.install_panels(self, self.tools_tabs)
        except Exception as e:
            log.error("panels failed: %s\n%s", e, traceback.format_exc()); self.tools_tabs.addTab(QLabel("Tool panels could not load:\n%s" % e), "Error")
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self.tools_tabs)
        self.dock_tools = self._dock("Photo Duet · ID Card · Passport · Print", sc, Qt.RightDockWidgetArea); self.dock_tools.setMinimumWidth(380)
        self.tabifyDockWidget(self.dock_layers, self.dock_props)
        self.setStatusBar(QStatusBar()); self.lbl_cursor = QLabel(""); self.lbl_zoom = QLabel(""); self.statusBar().addPermanentWidget(self.lbl_cursor); self.statusBar().addPermanentWidget(self.lbl_zoom)
        self.dock_layers.raise_()
    def _dock(self, title, widget, area):
        d = QDockWidget(title, self); d.setObjectName(title); d.setWidget(widget); self.addDockWidget(area, d); return d

    def _build_menus(self):
        mb = self.menuBar()
        def add(menu, text, fn, key=None):
            a = QAction(text, self); a.triggered.connect(lambda _=False, f=fn: f())
            if key: a.setShortcut(QKeySequence(key))
            menu.addAction(a); return a
        m = mb.addMenu("&File")
        add(m, "New project", self.project_new, "Ctrl+N"); add(m, "Open project...", self.project_open, "Ctrl+O"); add(m, "Save project", self.project_save, "Ctrl+S"); add(m, "Save project as...", self.project_save_as, "Ctrl+Shift+S")
        self.a_embed = add(m, "Embed original images in project file", self._toggle_embed); self.a_embed.setCheckable(True); self.a_embed.setChecked(True); m.addSeparator()
        add(m, "Import images...", self.import_dialog, "Ctrl+I")
        if self.host is not None: add(m, "Import current photo from main window", self.import_from_host)
        m.addSeparator(); add(m, "Export JPG...", lambda: self.export_dialog("jpg")); add(m, "Export PNG...", lambda: self.export_dialog("png")); add(m, "Export PDF (all / selected pages)...", lambda: self.export_dialog("pdf"), "Ctrl+E"); m.addSeparator()
        add(m, "Print preview...", lambda: self.print_dialog(True)); add(m, "Print...", lambda: self.print_dialog(False), "Ctrl+P"); m.addSeparator(); add(m, "Close Layout Studio", self.close)
        m = mb.addMenu("&Edit")
        add(m, "Undo", self.undo, "Ctrl+Z"); add(m, "Redo", self.redo, "Ctrl+Y"); m.addSeparator()
        add(m, "Copy", self.copy_sel, "Ctrl+C"); add(m, "Paste", self.paste, "Ctrl+V"); add(m, "Duplicate", self.duplicate_selected, "Ctrl+D"); add(m, "Delete", self.delete_selected, "Del"); m.addSeparator()
        add(m, "Select all (this page)", self.select_all, "Ctrl+A"); add(m, "Deselect", self.deselect, "Ctrl+Shift+A"); add(m, "Group", self.group_sel, "Ctrl+G"); add(m, "Ungroup", self.ungroup_sel, "Ctrl+Shift+G")
        m = mb.addMenu("&Arrange"); self.menu_arrange = m
        for t, fn in (("Bring to front", lambda: self.reorder(99)), ("Bring forward", lambda: self.reorder(1)), ("Send backward", lambda: self.reorder(-1)), ("Send to back", lambda: self.reorder(-99))): add(m, t, fn)
        m.addSeparator(); am = m.addMenu("Align (to page, or to selection if several)")
        for t, k in (("Left", "l"), ("Centre horizontally", "cx"), ("Right", "r"), ("Top", "t"), ("Centre vertically", "cy"), ("Bottom", "b")): add(am, t, lambda k=k: self.align(k))
        dm = m.addMenu("Distribute (3+ layers)"); add(dm, "Horizontally", lambda: self.distribute("h")); add(dm, "Vertically", lambda: self.distribute("v"))
        sm = m.addMenu("Make same size as first selected"); add(sm, "Width", lambda: self.same_size("w")); add(sm, "Height", lambda: self.same_size("h")); add(sm, "Both", lambda: self.same_size("wh"))
        add(m, "Scale selection...", self.scale_selection); add(m, "Fit to page (keep ratio)", self.fit_to_page); add(m, "Lock / unlock", self.toggle_lock); add(m, "Show / hide", self.toggle_visible); m.addSeparator()
        self.menu_move = m.addMenu("Move to page"); self.menu_copy = m.addMenu("Copy to page"); self.menu_move.aboutToShow.connect(lambda: self._fill_page_menu(self.menu_move, False)); self.menu_copy.aboutToShow.connect(lambda: self._fill_page_menu(self.menu_copy, True))
        m = mb.addMenu("&View")
        add(m, "Zoom in", lambda: self.cur_view().scale_by(1.25) if self.cur_view() else None, "Ctrl++"); add(m, "Zoom out", lambda: self.cur_view().scale_by(0.8) if self.cur_view() else None, "Ctrl+-")
        add(m, "Fit page in window", self.zoom_fit, "Ctrl+0"); add(m, "Actual print size on screen (100%)", self.zoom_actual, "Ctrl+1"); m.addSeparator()
        for t, attr in (("Show grid", "show_grid"), ("Snap to grid", "snap_grid"), ("Show margins", "show_margins")):
            a = add(m, t, lambda attr=attr: self._toggle(attr)); a.setCheckable(True); a.setChecked(getattr(self, attr))
        add(m, "Grid size...", self.set_grid); add(m, "Show / hide rulers", self.toggle_rulers); m.addSeparator()
        add(m, "Add guide at position...", self.add_guide_dialog); add(m, "Remove guides (this page)", self.clear_guides)
        m = mb.addMenu("&Page")
        add(m, "New page...", self.page_new); add(m, "Duplicate page", self.page_duplicate); add(m, "Rename page...", self.page_rename); add(m, "Delete page", self.page_delete)
        add(m, "Move page left", lambda: self.page_move(-1)); add(m, "Move page right", lambda: self.page_move(1)); m.addSeparator(); add(m, "Page setup (size, margins, background)...", self.page_setup)
        m = mb.addMenu("&Tools")
        add(m, "Crop (rectangle, ratio presets)...", lambda: self.open_crop("rect")); add(m, "Perspective crop (4 corners)...", lambda: self.open_crop("persp")); add(m, "Auto crop selected documents (batch)", self.batch_auto_crop); m.addSeparator()
        add(m, "Document restoration (clean up scans)...", self.open_restore); add(m, "Revert selected to original file", self.revert_selected); m.addSeparator(); add(m, "Document models / OCR setup...", self.open_models)

    # ---------------- helpers ----------------
    def cur_index(self): return max(0, self.tabs.currentIndex())
    def cur_page(self): return self.project.pages[self.cur_index()] if self.project.pages else None
    def cur_scene(self): p = self.cur_page(); return self.scenes.get(p.id) if p else None
    def cur_view(self): p = self.cur_page(); return self.views.get(p.id) if p else None
    def page_row_at(self, global_pt):
        vp = self.page_list.viewport(); lp = vp.mapFromGlobal(global_pt)
        if not vp.rect().contains(lp): return -1
        it = self.page_list.itemAt(lp); return self.page_list.row(it) if it is not None else -1
    def display_image(self, L):
        """cached picture for the canvas incl. the non-destructive brightness / contrast / grey adjustments (export applies the same maths)"""
        q = self.cache.qimage(L.path); a = L.adjust or {}
        if q is None or not (a.get("brightness") or a.get("contrast") or a.get("gray")): return q
        key = (L.path, a.get("brightness", 0), a.get("contrast", 0), bool(a.get("gray")), q.width()); hit = self._adj_cache.get(key)
        if hit is None:
            if len(self._adj_cache) > 8: self._adj_cache.clear()
            hit = pil_to_qimage(rc._apply_adjust(qimage_to_pil(q), a)).convertToFormat(QImage.Format_ARGB32_Premultiplied); self._adj_cache[key] = hit
        return hit
    def add_images_as_pages(self, paths, paper="A4"):
        """one NEW page per image (documents / scans): page orientation follows the picture, picture fitted inside the margins"""
        paths = [p for p in paths if self.cache.qimage(p) is not None]
        if not paths: return
        self.push_undo(); first = None
        for p in paths:
            q = self.cache.qimage(p); ori = "Landscape" if q.width() > q.height() else "Portrait"; w, h = rc.page_size_mm(paper, ori)
            idx = self.add_page_silent(w, h, os.path.splitext(os.path.basename(p))[0][:24], paper, ori, 5.0); pg = self.project.pages[idx]; m = pg.margin_mm
            k = min((w - 2 * m) / q.width(), (h - 2 * m) / q.height()); lw, lh = q.width() * k, q.height() * k
            pg.layers.append(rc.Layer(pg.name, p, (w - lw) / 2.0, (h - lh) / 2.0, lw, lh)); first = idx if first is None else first
        self.rebuild_all(select=first); self.say("%d page(s) added" % len(paths))
    def say(self, text, ms=6000): self.statusBar().showMessage(text, ms)
    def status_cursor(self, x, y): self.lbl_cursor.setText("X %.1f mm  Y %.1f mm   (%.2f, %.2f in)" % (x, y, x / 25.4, y / 25.4))
    def status_zoom(self):
        v = self.cur_view()
        if v: self.lbl_zoom.setText("  View %d%%  (print size unchanged)" % round(v.transform().m11() * SC * 25.4 / max(1, self.logicalDpiX()) * 100))
    def update_rulers(self): self.rh.update(); self.rv.update()
    def toggle_rulers(self): vis = not self.rh.isVisible(); self.rh.setVisible(vis); self.rv.setVisible(vis)
    def _toggle(self, attr): setattr(self, attr, not getattr(self, attr)); s = self.cur_scene(); s and s.update()
    def _toggle_embed(self): self.embed = self.a_embed.isChecked()
    def selected_items(self): s = self.cur_scene(); return [i for i in s.selectedItems() if isinstance(i, LayerItem)] if s else []
    def selected_layers(self): return [i.layer for i in self.selected_items()]
    def mark_dirty(self): self.dirty = True; self.setWindowTitle("Ripon Layout Studio - %s*" % (os.path.basename(self.path) if self.path else self.project.name)); self.thumb_timer.start()
    def pil_full(self, layer): return pil_open(layer.path) if layer.path and os.path.isfile(layer.path) else None
    def pil_preview(self, layer): return self.cache.pil(layer.path)

    # ---------------- undo / redo (whole-project snapshots: simple and safe) ----------------
    def _snap(self): return json.dumps(self.project.to_dict())
    def push_undo(self, key=None, merge_s=0.8):
        now = time.time()
        if key and self._last_push[0] == key and now - self._last_push[1] < merge_s: self._last_push = (key, now); self.mark_dirty(); return
        self.undo_stack.append(self._snap()); self.undo_stack = self.undo_stack[-60:]; self.redo_stack.clear(); self._last_push = (key, now); self.mark_dirty()
    def begin_gesture(self): self._gesture_pre = self._snap()
    def end_gesture(self):
        if self._gesture_pre is None: return
        pre, self._gesture_pre = self._gesture_pre, None
        self.commit_items()
        if self._snap() != pre:
            self.undo_stack.append(pre); self.undo_stack = self.undo_stack[-60:]; self.redo_stack.clear(); self.mark_dirty(); self._refresh_side()
        sc = self.cur_scene(); sc and sc.update_sel_handle()
    def _restore(self, snap):
        idx = self.cur_index(); self.project = rc.Project.from_dict(json.loads(snap)); self.rebuild_all(select=idx); self.mark_dirty()
    def undo(self):
        if not self.undo_stack: return self.say("Nothing to undo")
        self.redo_stack.append(self._snap()); self._restore(self.undo_stack.pop())
    def redo(self):
        if not self.redo_stack: return self.say("Nothing to redo")
        self.undo_stack.append(self._snap()); self._restore(self.redo_stack.pop())

    # ---------------- building views ----------------
    def rebuild_all(self, select=None):
        for pg in self.project.pages:
            v = self.views.get(pg.id)
            if v is not None: self._view_state[pg.id] = (v.transform(), v.horizontalScrollBar().value(), v.verticalScrollBar().value())
        old = list(self.views.values()) + list(self.scenes.values())
        self.tabs.blockSignals(True); self.tabs.clear(); self.scenes.clear(); self.views.clear()
        for o in old: o.deleteLater()
        for pg in self.project.pages:
            sc = PageScene(self, pg); v = CanvasView(self, sc); self.scenes[pg.id] = sc; self.views[pg.id] = v; self.tabs.addTab(v, pg.name)
            st = self._view_state.get(pg.id)
            if st: v.setTransform(st[0]); v.horizontalScrollBar().setValue(st[1]); v.verticalScrollBar().setValue(st[2])
            else: QTimer.singleShot(0, lambda v=v, sc=sc, pg=pg: self._fit(v, pg))
        self.tabs.blockSignals(False)
        i = min(len(self.project.pages) - 1, select if select is not None else 0); self.tabs.setCurrentIndex(max(0, i)); self._tab_changed(max(0, i))
        self.refresh_thumbs(); self.update_rulers(); self.setWindowTitle("Ripon Layout Studio - %s%s" % (os.path.basename(self.path) if self.path else self.project.name, "*" if self.dirty else ""))
    def refresh_page_scene(self):
        s = self.cur_scene()
        if s: s.rebuild(); self._refresh_side(); self.thumb_timer.start()
    def _fit(self, v, pg):
        v.fitInView(QRectF(-8 * SC, -8 * SC, (pg.w_mm + 16) * SC, (pg.h_mm + 16) * SC), Qt.KeepAspectRatio); self.update_rulers(); self.status_zoom()
    def zoom_fit(self): v, p = self.cur_view(), self.cur_page(); v and self._fit(v, p)
    def zoom_actual(self):
        v = self.cur_view()
        if v: v.resetTransform(); k = (self.logicalDpiX() / 25.4) / SC; v.scale(k, k); self.update_rulers(); self.status_zoom()
    def _tab_changed(self, i):
        self.page_list.blockSignals(True); self.page_list.setCurrentRow(i); self.page_list.blockSignals(False)
        self._refresh_side(); self.update_rulers(); self.status_zoom()
        if hasattr(self, "panels_page_changed"): self.panels_page_changed()
    def _tab_moved(self, frm, to):
        pg = self.project.pages.pop(frm); self.project.pages.insert(to, pg); self.push_undo_after_move(); self.refresh_thumbs()
    def push_undo_after_move(self): self.mark_dirty()
    def refresh_thumbs(self):
        cur = self.page_list.currentRow(); self.page_list.blockSignals(True); self.page_list.clear()
        for i, pg in enumerate(self.project.pages):
            try:
                im = rc.render_page(pg, 24, loader=self.pil_preview); im.thumbnail((96, 128)); qi = pil_to_qimage(im); ic = QIcon(QPixmap.fromImage(qi))
            except Exception as e: log.warning("thumb: %s", e); ic = QIcon()
            it = QListWidgetItem(ic, "%d. %s" % (i + 1, pg.name)); it.setTextAlignment(Qt.AlignHCenter); self.page_list.addItem(it)
        self.page_list.setCurrentRow(cur if cur >= 0 else self.cur_index()); self.page_list.blockSignals(False)
        for i, pg in enumerate(self.project.pages):
            if i < self.tabs.count() and self.tabs.tabText(i) != pg.name: self.tabs.setTabText(i, pg.name)

    # ---------------- side panels ----------------
    def _refresh_side(self):
        pg = self.cur_page(); self._sync = True; sel_ids = {l.id for l in self.selected_layers()}; self.layer_list.clear()
        if pg:
            for L in reversed(pg.layers):
                it = QListWidgetItem(("🔒 " if L.locked else "") + L.name + ("   [group]" if L.group else "")); it.setData(Qt.UserRole, L.id)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Checked if L.visible else Qt.Unchecked); self.layer_list.addItem(it)
                if L.id in sel_ids: it.setSelected(True)
        self._sync = False; self.props_from_selection()
    def scene_selection_changed(self, scene):
        if scene is not self.cur_scene() or self._sync: return
        self._sync = True
        # groups: selecting one member selects the whole group
        ids = {i.layer.group for i in scene.selectedItems() if isinstance(i, LayerItem) and i.layer.group}
        if ids:
            for it in scene.items_by_id.values():
                if it.layer.group in ids and not it.isSelected(): it.setSelected(True)
        sel = {l.id for l in self.selected_layers()}
        for r in range(self.layer_list.count()):
            it = self.layer_list.item(r); it.setSelected(it.data(Qt.UserRole) in sel)
        self._sync = False; self.props_from_selection()
    def _layer_list_selected(self):
        if self._sync: return
        s = self.cur_scene()
        if not s: return
        ids = {i.data(Qt.UserRole) for i in self.layer_list.selectedItems()}; self._sync = True; s.blockSignals(True)
        for lid, it in s.items_by_id.items(): it.setSelected(lid in ids)
        s.blockSignals(False)
        for it in s.items_by_id.values(): it.update_handles()
        self._sync = False; self.props_from_selection()
    def _layer_item_changed(self, item):
        if self._sync: return
        pg = self.cur_page(); lid = item.data(Qt.UserRole); L = next((l for l in pg.layers if l.id == lid), None)
        if L is None: return
        vis = item.checkState() == Qt.Checked
        if vis != L.visible: self.push_undo(); L.visible = vis; self.cur_scene().items_by_id[L.id].sync(); self.thumb_timer.start()
    def _unit_factor(self): return 1.0 if self.unit.currentText() == "mm" else 1.0 / 25.4
    def _unit_changed(self, *_):
        for k in ("x", "y", "w", "h"): self.sp[k].setDecimals(2 if self.unit.currentText() == "mm" else 3)
        self.props_from_selection()
    def props_from_selection(self):
        ls = self.selected_layers(); one = len(ls) == 1; self._sync = True
        for w in list(self.sp.values()) + [self.c_flip_h, self.c_flip_v, self.c_gray]: w.setEnabled(one)
        if one:
            L = ls[0]; f = self._unit_factor()
            for k, v in (("x", L.x_mm), ("y", L.y_mm), ("w", L.w_mm), ("h", L.h_mm)): self.sp[k].setValue(v * f)
            self.sp["rot"].setValue(L.rot); self.sp["op"].setValue(int(round(L.opacity * 100))); self.c_flip_h.setChecked(L.flip_h); self.c_flip_v.setChecked(L.flip_v)
            self.sp["bri"].setValue(int(L.adjust.get("brightness", 0))); self.sp["con"].setValue(int(L.adjust.get("contrast", 0))); self.c_gray.setChecked(bool(L.adjust.get("gray")))
            q = self.cache.qimage(L.path)
            try:
                src = QImageReader(L.path).size() if L.path else None
                eff = (max(src.width(), src.height()) / (max(L.w_mm, L.h_mm) / 25.4)) if src and src.isValid() else 0
            except Exception: eff = 0
            self.lbl_px.setText("Effective resolution at this size: ~%d DPI%s" % (eff, "  ⚠ low for print" if 0 < eff < 150 else "") if eff else "")
        else: self.lbl_px.setText("%d layers selected" % len(ls) if ls else "")
        self._sync = False
    def _prop_edited(self, *_):
        if self._sync: return
        ls = self.selected_layers()
        if len(ls) != 1: return
        L = ls[0]; it = self.cur_scene().items_by_id[L.id]; f = self._unit_factor(); self.push_undo("props")
        nw, nh = self.sp["w"].value() / f, self.sp["h"].value() / f
        if self.keep_aspect and (abs(nw - L.w_mm) > 1e-6) and abs(nh - L.h_mm) < 1e-6: nh = nw * L.h_mm / L.w_mm
        elif self.keep_aspect and (abs(nh - L.h_mm) > 1e-6) and abs(nw - L.w_mm) < 1e-6: nw = nh * L.w_mm / L.h_mm
        L.x_mm, L.y_mm = self.sp["x"].value() / f, self.sp["y"].value() / f; L.w_mm, L.h_mm = max(1.0, nw), max(1.0, nh)
        L.rot = self.sp["rot"].value(); L.opacity = self.sp["op"].value() / 100.0; L.flip_h, L.flip_v = self.c_flip_h.isChecked(), self.c_flip_v.isChecked()
        L.adjust["brightness"], L.adjust["contrast"], L.adjust["gray"] = self.sp["bri"].value(), self.sp["con"].value(), self.c_gray.isChecked()
        it.sync(); self.props_from_selection()

    # ---------------- snapping ----------------
    def snap_pos(self, item, pos):
        L = item.layer; pg = self.cur_page(); tol = 1.5
        if pg is None: return pos
        x, y = pos.x() / SC, pos.y() / SC; w, h = L.w_mm, L.h_mm; m = pg.margin_mm; sx = sy = False
        for cand in [0, m, pg.w_mm / 2.0] + list(pg.guides_v):
            for off in (0, w / 2.0, w):
                if abs(x + off - cand) < tol: x = cand - off; sx = True; break
            if sx: break
        for cand in [pg.w_mm, pg.w_mm - m]:
            if not sx and abs(x + w - cand) < tol: x = cand - w; sx = True
        for cand in [0, m, pg.h_mm / 2.0] + list(pg.guides_h):
            for off in (0, h / 2.0, h):
                if abs(y + off - cand) < tol: y = cand - off; sy = True; break
            if sy: break
        for cand in [pg.h_mm, pg.h_mm - m]:
            if not sy and abs(y + h - cand) < tol: y = cand - h; sy = True
        if self.snap_grid and self.grid_mm > 0:
            g = self.grid_mm
            if not sx: x = round(x / g) * g
            if not sy: y = round(y / g) * g
        return QPointF(x * SC, y * SC)
    def commit_items(self):
        s = self.cur_scene()
        if not s: return
        for it in s.items_by_id.values(): it.pull()
    def nudge(self, dx, dy):
        ls = self.selected_items()
        if not ls: return
        self.push_undo("nudge", 0.6)
        for it in ls:
            if not it.layer.locked: it.layer.x_mm += dx; it.layer.y_mm += dy; it.sync()
        self.props_from_selection()

    # ---------------- adding / removing layers ----------------
    def import_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import images (many at once)", "", IMAGE_FILTER)
        if paths: self.add_images(paths)
    def import_from_host(self):
        try:
            im = getattr(self.host, "photo", None) or self.host.composite()
            p = unique_path(os.path.join(self.cache_dir, "main_photo.png")); im.convert("RGB").save(p); self.add_images([p])
        except Exception as e: QMessageBox.information(self, "Layout Studio", "No photo is open in the main window yet.\n\n%s" % e)
    def default_size_mm(self, path, dpi=300):
        """real size = pixels / 300 dpi; shrunk to fit ~80% of the page if too big (aspect kept)"""
        q = self.cache.qimage(path)
        if q is None: return 50.0, 50.0
        rd = QImageReader(path); s = rd.size(); big = max(s.width(), s.height()) if s.isValid() else max(q.width(), q.height())
        ratio = q.width() / float(q.height()); mx = big / float(dpi) * 25.4
        w, h = (mx, mx / ratio) if ratio >= 1 else (mx * ratio, mx); pg = self.cur_page()
        k = min(1.0, 0.8 * pg.w_mm / w, 0.8 * pg.h_mm / h); return w * k, h * k
    def add_images(self, paths, at_mm=None, page_index=None):
        if not paths: return []
        self.push_undo(); pg = self.project.pages[self.cur_index() if page_index is None else page_index]; new = []; off = 0.0
        for p in paths:
            if self.cache.qimage(p) is None: self.say("Cannot open: %s" % os.path.basename(p)); continue
            w, h = self.default_size_mm(p)
            x, y = (at_mm[0] - w / 2.0, at_mm[1] - h / 2.0) if at_mm else ((pg.w_mm - w) / 2.0 + off, (pg.h_mm - h) / 2.0 + off)
            L = rc.Layer(os.path.splitext(os.path.basename(p))[0], p, x + off if at_mm else x, y + off if at_mm else y, w, h); pg.layers.append(L); new.append(L); off += 6.0
        self.refresh_page_scene(); s = self.cur_scene()
        if s:
            s.blockSignals(True)
            for L in new:
                if L.id in s.items_by_id: s.items_by_id[L.id].setSelected(True)
            s.blockSignals(False); self.scene_selection_changed(s)
        self.say("%d image(s) added" % len(new)); return new
    def add_layers_to_page(self, layers, page_index):
        """used by Duet / ID / Passport panels: layers already positioned in mm"""
        self.project.pages[page_index].layers.extend(layers)
    def delete_selected(self):
        ls = [l for l in self.selected_layers() if not l.locked]
        if not ls: return
        self.push_undo(); ids = {l.id for l in ls}; pg = self.cur_page(); pg.layers = [l for l in pg.layers if l.id not in ids]; self.refresh_page_scene()
    def duplicate_selected(self):
        ls = self.selected_layers()
        if not ls: return
        self.push_undo(); pg = self.cur_page(); new = []
        for L in ls: n = L.duplicate(5, 5); n.group = ""; pg.layers.append(n); new.append(n)
        self.refresh_page_scene(); self._select_ids({n.id for n in new})
    def _select_ids(self, ids):
        s = self.cur_scene()
        if not s: return
        s.blockSignals(True)
        for lid, it in s.items_by_id.items(): it.setSelected(lid in ids)
        s.blockSignals(False); self.scene_selection_changed(s)
    def select_all(self):
        s = self.cur_scene()
        if s:
            for it in s.items_by_id.values(): it.setSelected(True)
    def deselect(self):
        s = self.cur_scene(); s and s.clearSelection()
    def copy_sel(self):
        ls = self.selected_layers()
        if ls: self.clip = [l.to_dict() for l in ls]; self.say("%d layer(s) copied (full resolution kept: the copy points at the same file)" % len(ls))
    def paste(self):
        if not self.clip: return
        self.push_undo(); pg = self.cur_page(); new = []
        for d in self.clip:
            n = rc.Layer.from_dict(d); n.id = rc._id(); n.x_mm += 5; n.y_mm += 5; n.group = ""; pg.layers.append(n); new.append(n)
        self.clip = [dict(d, x_mm=d["x_mm"] + 5, y_mm=d["y_mm"] + 5) for d in self.clip]; self.refresh_page_scene(); self._select_ids({n.id for n in new})
    def group_sel(self):
        ls = self.selected_layers()
        if len(ls) < 2: return self.say("Select two or more layers to group")
        self.push_undo(); g = rc._id()
        for L in ls: L.group = g
        self._refresh_side()
    def ungroup_sel(self):
        ls = self.selected_layers()
        if not ls: return
        self.push_undo()
        for L in ls: L.group = ""
        self._refresh_side()
    def toggle_lock(self):
        ls = self.selected_layers()
        if not ls: return
        self.push_undo(); v = not all(l.locked for l in ls)
        for L in ls: L.locked = v; self.cur_scene().items_by_id[L.id].sync()
        self._refresh_side()
    def toggle_visible(self):
        ls = self.selected_layers()
        if not ls: return
        self.push_undo(); v = not all(l.visible for l in ls)
        for L in ls: L.visible = v; self.cur_scene().items_by_id[L.id].sync()
        self._refresh_side(); self.thumb_timer.start()
    def rename_layer(self):
        ls = self.selected_layers()
        if len(ls) != 1: return
        t, ok = QInputDialog.getText(self, "Rename layer", "Name:", text=ls[0].name)
        if ok and t.strip(): self.push_undo(); ls[0].name = t.strip(); self._refresh_side()
    def reorder(self, d):
        ls = self.selected_layers(); pg = self.cur_page()
        if not ls: return
        self.push_undo(); ids = {l.id for l in ls}; arr = pg.layers
        if d >= 50:   arr = [l for l in arr if l.id not in ids] + [l for l in arr if l.id in ids]
        elif d <= -50: arr = [l for l in arr if l.id in ids] + [l for l in arr if l.id not in ids]
        else:
            for _ in range(abs(d)):
                rng = range(len(arr) - 2, -1, -1) if d > 0 else range(1, len(arr))
                for i in rng:
                    j = i + 1 if d > 0 else i - 1
                    if arr[i].id in ids and arr[j].id not in ids: arr[i], arr[j] = arr[j], arr[i]
        pg.layers = arr; self.refresh_page_scene(); self._select_ids(ids)

    # ---------------- arrange ----------------
    def _bounds(self, L): return rc.rect_bounds_rotated(L.x_mm, L.y_mm, L.w_mm, L.h_mm, L.rot)
    def align(self, k):
        ls = [l for l in self.selected_layers() if not l.locked]
        if not ls: return
        pg = self.cur_page(); self.push_undo()
        bs = [self._bounds(l) for l in ls]
        if len(ls) == 1: ref = (0, 0, pg.w_mm, pg.h_mm)
        else: ref = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
        for L, b in zip(ls, bs):
            if k == "l": L.x_mm += ref[0] - b[0]
            elif k == "r": L.x_mm += ref[2] - b[2]
            elif k == "cx": L.x_mm += (ref[0] + ref[2]) / 2 - (b[0] + b[2]) / 2
            elif k == "t": L.y_mm += ref[1] - b[1]
            elif k == "b": L.y_mm += ref[3] - b[3]
            elif k == "cy": L.y_mm += (ref[1] + ref[3]) / 2 - (b[1] + b[3]) / 2
        self.refresh_page_scene(); self._select_ids({l.id for l in ls})
    def distribute(self, axis):
        ls = [l for l in self.selected_layers() if not l.locked]
        if len(ls) < 3: return self.say("Select 3 or more layers to distribute")
        self.push_undo(); i0, i1 = (0, 2) if axis == "h" else (1, 3); ls.sort(key=lambda l: self._bounds(l)[i0])
        a, b = self._bounds(ls[0]), self._bounds(ls[-1]); total = (b[i1] - a[i0]) - sum(self._bounds(l)[i1] - self._bounds(l)[i0] for l in ls); gap = total / (len(ls) - 1); pos = a[i1] + gap
        for L in ls[1:-1]:
            bb = self._bounds(L); d = pos - bb[i0]
            if axis == "h": L.x_mm += d
            else: L.y_mm += d
            pos += (bb[i1] - bb[i0]) + gap
        self.refresh_page_scene(); self._select_ids({l.id for l in ls})
    def same_size(self, which):
        ls = self.selected_layers()
        if len(ls) < 2: return
        self.push_undo(); r = ls[0]
        for L in ls[1:]:
            if "w" in which: L.w_mm = r.w_mm
            if "h" in which: L.h_mm = r.h_mm
        self.refresh_page_scene(); self._select_ids({l.id for l in ls})
    def scale_selection(self):
        ls = self.selected_layers()
        if not ls: return
        v, ok = QInputDialog.getDouble(self, "Scale selection", "Scale in % (aspect kept, about the selection centre):", 100.0, 5.0, 1000.0, 1)
        if not ok: return
        self.push_undo(); k = v / 100.0; bs = [self._bounds(l) for l in ls]
        cx, cy = (min(b[0] for b in bs) + max(b[2] for b in bs)) / 2, (min(b[1] for b in bs) + max(b[3] for b in bs)) / 2
        for L in ls:
            lx, ly = L.x_mm + L.w_mm / 2, L.y_mm + L.h_mm / 2; nx, ny = cx + (lx - cx) * k, cy + (ly - cy) * k
            L.w_mm *= k; L.h_mm *= k; L.x_mm, L.y_mm = nx - L.w_mm / 2, ny - L.h_mm / 2
        self.refresh_page_scene(); self._select_ids({l.id for l in ls})
    def fit_to_page(self):
        ls = self.selected_layers(); pg = self.cur_page()
        if not ls: return
        self.push_undo()
        for L in ls:
            k = min((pg.w_mm - 2 * pg.margin_mm) / L.w_mm, (pg.h_mm - 2 * pg.margin_mm) / L.h_mm); L.w_mm *= k; L.h_mm *= k; L.rot = 0.0
            L.x_mm, L.y_mm = (pg.w_mm - L.w_mm) / 2, (pg.h_mm - L.h_mm) / 2
        self.refresh_page_scene(); self._select_ids({l.id for l in ls})
    def set_grid(self):
        v, ok = QInputDialog.getDouble(self, "Grid", "Grid spacing (mm):", self.grid_mm, 0.5, 100.0, 1)
        if ok: self.grid_mm = v; s = self.cur_scene(); s and s.update()
    def add_guide(self, orient, mm):
        pg = self.cur_page(); self.push_undo(); (pg.guides_h if orient == "h" else pg.guides_v).append(round(mm, 2)); s = self.cur_scene(); s and s.update()
    def add_guide_dialog(self):
        o, ok = QInputDialog.getItem(self, "Guide", "Direction:", ["Horizontal (Y position)", "Vertical (X position)"], 0, False)
        if not ok: return
        v, ok = QInputDialog.getDouble(self, "Guide", "Position (mm):", 10.0, -1000, 5000, 2)
        if ok: self.add_guide("h" if o.startswith("H") else "v", v)
    def clear_guides(self):
        pg = self.cur_page(); self.push_undo(); pg.guides_h = []; pg.guides_v = []; s = self.cur_scene(); s and s.update()

    # ---------------- pages ----------------
    def _new_page_from(self, v):
        pg = rc.Page(v["name"], v["w"], v["h"], v["paper"], v["orientation"]); pg.margin_mm = v["margin"]; pg.bg = tuple(v["bg"]); return pg
    def page_new(self):
        cur = self.cur_page(); d = PageSetupDialog(self, None, "New page")
        if cur: d.paper.setCurrentText(cur.paper if cur.paper in rc.PAPER_NAMES else "Custom"); d.orient.setCurrentText(cur.orientation); d.cw.setValue(cur.w_mm); d.ch.setValue(cur.h_mm); d.margin.setValue(cur.margin_mm)
        d.name.setText("Page %d" % (len(self.project.pages) + 1))
        if d.exec() != QDialog.Accepted: return
        self.push_undo(); self.project.pages.append(self._new_page_from(d.values())); self.rebuild_all(select=len(self.project.pages) - 1)
    def add_page_silent(self, w, h, name="Page", paper="Custom", orientation="Portrait", margin=5.0):
        """used by panels (spill-over pages); caller already did push_undo"""
        pg = rc.Page(name, w, h, paper, orientation); pg.margin_mm = margin; self.project.pages.append(pg); return len(self.project.pages) - 1
    def page_duplicate(self):
        pg = self.cur_page(); d = pg.to_dict(); n = rc.Page.from_dict(d); n.id = rc._id(); n.name = pg.name + " copy"
        for L in n.layers: L.id = rc._id()
        self.push_undo(); self.project.pages.insert(self.cur_index() + 1, n); self.rebuild_all(select=self.cur_index() + 1)
    def page_rename(self):
        pg = self.cur_page(); t, ok = QInputDialog.getText(self, "Rename page", "Name:", text=pg.name)
        if ok and t.strip(): self.push_undo(); pg.name = t.strip(); self.refresh_thumbs()
    def page_delete(self):
        if len(self.project.pages) <= 1: return QMessageBox.information(self, "Layout Studio", "A project needs at least one page.")
        pg = self.cur_page()
        if QMessageBox.question(self, "Delete page", "Delete '%s' and its %d layer(s)?\n(Your image files are not deleted.)" % (pg.name, len(pg.layers))) != QMessageBox.Yes: return
        i = self.cur_index(); self.push_undo(); self.project.pages.pop(i); self.rebuild_all(select=max(0, i - 1))
    def page_move(self, d):
        i = self.cur_index(); j = i + d
        if not (0 <= j < len(self.project.pages)): return
        self.push_undo(); self.project.pages[i], self.project.pages[j] = self.project.pages[j], self.project.pages[i]; self.rebuild_all(select=j)
    def page_setup(self):
        pg = self.cur_page(); d = PageSetupDialog(self, pg)
        if d.exec() != QDialog.Accepted: return
        v = d.values(); self.push_undo(); targets = self.project.pages if v["all"] else [pg]
        for t in targets: t.w_mm, t.h_mm, t.paper, t.orientation = v["w"], v["h"], v["paper"], v["orientation"]; t.margin_mm = v["margin"]; t.bg = tuple(v["bg"])
        pg.name = v["name"]; self.rebuild_all(select=self.cur_index())
    def _fill_page_menu(self, menu, copy_):
        menu.clear()
        for i, pg in enumerate(self.project.pages):
            if i == self.cur_index() and not copy_: continue
            a = QAction("%d. %s" % (i + 1, pg.name), self); a.triggered.connect(lambda _=False, i=i: self.send_layers_to_page([l.id for l in self.selected_layers()], i, copy_)); menu.addAction(a)
    def send_layers_to_page(self, ids, target, copy_=False):
        src = self.cur_page(); dst = self.project.pages[target]; ls = [l for l in src.layers if l.id in set(ids)]
        if not ls: return
        if dst is src and not copy_: return
        self.push_undo(); new = []
        for L in ls:
            n = rc.Layer.from_dict(L.to_dict()) if copy_ else L
            if copy_: n.id = rc._id(); n.group = ""
            b = rc.rect_bounds_rotated(n.x_mm, n.y_mm, n.w_mm, n.h_mm, n.rot)           # size, ratio and file (= resolution) are untouched; only the position is kept on the page
            if b[2] > dst.w_mm: n.x_mm -= min(b[2] - dst.w_mm, max(0.0, b[0]))
            if b[3] > dst.h_mm: n.y_mm -= min(b[3] - dst.h_mm, max(0.0, b[1]))
            dst.layers.append(n); new.append(n)
        if not copy_: src.layers = [l for l in src.layers if l.id not in {x.id for x in ls}]
        self.rebuild_all(select=self.cur_index()); self.say("%d layer(s) %s '%s'" % (len(ls), "copied to" if copy_ else "moved to", dst.name))

    # ---------------- tools (dialogs live in ripon_studio_tools.py) ----------------
    def _tools(self):
        import ripon_studio_tools as t; return t
    def open_crop(self, mode):
        ls = self.selected_layers()
        if len(ls) != 1: return QMessageBox.information(self, "Crop", "Select exactly one image layer first.")
        try: self._tools().CropDialog(self, ls[0], mode).exec()
        except Exception as e: self._tool_error("Crop", e)
    def batch_auto_crop(self):
        ls = self.selected_layers()
        if not ls: return QMessageBox.information(self, "Auto crop", "Select the document layers first (Ctrl+A for all).")
        try: self._tools().batch_auto_crop(self, ls)
        except Exception as e: self._tool_error("Auto crop", e)
    def open_restore(self):
        ls = self.selected_layers()
        if not ls: return QMessageBox.information(self, "Restoration", "Select one or more scanned document layers first.")
        try: self._tools().RestoreDialog(self, ls).exec()
        except Exception as e: self._tool_error("Restoration", e)
    def open_models(self):
        try: self._tools().ModelsDialog(self).exec()
        except Exception as e: self._tool_error("Models", e)
    def _tool_error(self, name, e):
        log.error("%s failed: %s\n%s", name, e, traceback.format_exc()); QMessageBox.warning(self, name, "%s could not open:\n%s" % (name, e))
    def revert_selected(self):
        ls = [l for l in self.selected_layers() if l.adjust.get("orig_path")]
        if not ls: return self.say("The selected layers have no earlier original to go back to.")
        self.push_undo()
        for L in ls:
            L.path = L.adjust.pop("orig_path"); ow = L.adjust.pop("orig_crop", None); L.crop = tuple(ow) if ow else None
        self.cache.clear(); self.refresh_page_scene()
    def replace_layer_image(self, L, new_path, keep_width=True):
        """swap the picture of a layer for a NEW derived file (crop / restore); the original path is remembered, the file itself untouched"""
        if "orig_path" not in L.adjust: L.adjust["orig_path"] = L.path; L.adjust["orig_crop"] = list(L.crop) if L.crop else None
        q = QImageReader(new_path).size()
        if q.isValid() and q.width() > 0:
            r = q.height() / float(q.width()); c = (L.x_mm + L.w_mm / 2, L.y_mm + L.h_mm / 2)
            if keep_width: L.h_mm = L.w_mm * r
            else: L.w_mm = L.h_mm / r
            L.x_mm, L.y_mm = c[0] - L.w_mm / 2, c[1] - L.h_mm / 2
        L.path = new_path; L.crop = None

    # ---------------- project IO ----------------
    def _ask_save_changes(self):
        if not self.dirty: return True
        r = QMessageBox.question(self, "Unsaved changes", "Save changes to this project first?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Cancel: return False
        if r == QMessageBox.Save: return self.project_save()
        return True
    def project_new(self):
        if not self._ask_save_changes(): return
        self.project = rc.Project("Untitled"); self.path = None; self.dirty = False; self.undo_stack.clear(); self.redo_stack.clear(); self._view_state.clear(); self.cache.clear(); self.autosaver.clear(); self.rebuild_all()
    def project_save(self):
        if not self.path: return self.project_save_as()
        return self._save_to(self.path)
    def project_save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save project", self.path or "layout.rproj", "Ripon project (*.rproj)")
        if not p: return False
        if not p.lower().endswith(".rproj"): p += ".rproj"
        return self._save_to(p)
    def _save_to(self, p):
        try:
            self.commit_items(); self.project.name = os.path.splitext(os.path.basename(p))[0]; rc.save_project(self.project, p, embed=self.embed)
        except Exception as e:
            log.error("save failed: %s\n%s", e, traceback.format_exc()); QMessageBox.warning(self, "Save failed", "The project could not be saved:\n%s" % e); return False
        self.path = p; self.dirty = False; self.autosaver.clear(); self.setWindowTitle("Ripon Layout Studio - %s" % os.path.basename(p)); self.say("Saved: " + p); return True
    def project_open(self):
        if not self._ask_save_changes(): return
        p, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Ripon project (*.rproj)")
        if p: self._open_path(p)
    def _open_path(self, p):
        try: proj, missing = rc.open_project(p, os.path.join(self.cache_dir, "proj_" + os.path.splitext(os.path.basename(p))[0]))
        except Exception as e:
            log.error("open failed: %s\n%s", e, traceback.format_exc()); return QMessageBox.warning(self, "Open failed", "This project file could not be opened:\n%s" % e)
        if missing:
            r = QMessageBox.question(self, "Missing images", "%d image file(s) were not found.\nChoose a folder to search for them (relink)?" % len(missing))
            if r == QMessageBox.Yes:
                d = QFileDialog.getExistingDirectory(self, "Folder that contains the original images")
                if d: n = rc.relink(missing, d); QMessageBox.information(self, "Relink", "%d of %d file(s) found. The rest are shown as 'Missing file'." % (n, len(missing)))
        self.project = proj; self.path = p; self.dirty = False; self.undo_stack.clear(); self.redo_stack.clear(); self._view_state.clear(); self.cache.clear(); self.rebuild_all()
    def _autosave(self):
        if self.dirty:
            try: self.commit_items(); self.autosaver.save(self.project)
            except Exception as e: log.warning("autosave failed: %s", e)
    def _offer_recovery(self):
        if self.autosaver.pending():
            if QMessageBox.question(self, "Recover work", "Layout Studio did not close normally last time.\nRecover the autosaved layout?") == QMessageBox.Yes:
                try: self.project = self.autosaver.load(); self.path = None; self.dirty = True; self.rebuild_all()
                except Exception as e: QMessageBox.warning(self, "Recover", "Could not recover: %s" % e)
            else: self.autosaver.clear()
    def closeEvent(self, e):
        if not self._ask_save_changes(): return e.ignore()
        self.auto_timer.stop(); self.autosaver.clear(); e.accept()

    # ---------------- export ----------------
    def ask_pages_and_dpi(self, title, with_dpi=True):
        d = QDialog(self); d.setWindowTitle(title); f = QFormLayout(d); n = len(self.project.pages)
        scope = QComboBox(); scope.addItems(["All pages", "Current page only", "Page range"]); rng = QLineEdit(); rng.setPlaceholderText("e.g. 1-3,5")
        dpi = QSpinBox(); dpi.setRange(72, 1200); dpi.setValue(int(self.project.dpi)); q = QSpinBox(); q.setRange(50, 100); q.setValue(95)
        f.addRow("Pages", scope); f.addRow("Range", rng)
        if with_dpi: f.addRow("Output resolution (DPI)", dpi); f.addRow("JPEG quality", q); f.addRow(QLabel("Physical size is exact: pixels = size in inches × DPI.\nDPI changes pixel count, not real detail (no fake sharpness)."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
        if d.exec() != QDialog.Accepted: return None
        try: idx = [self.cur_index()] if scope.currentIndex() == 1 else (parse_range(rng.text(), n) if scope.currentIndex() == 2 else list(range(n)))
        except ValueError as e: QMessageBox.warning(self, title, str(e)); return None
        self.project.dpi = dpi.value(); return idx, dpi.value(), q.value()
    def check_out_of_page(self, idx):
        bad = []
        for i in idx:
            pg = self.project.pages[i]; b = rc.out_of_page(pg.layers, pg.w_mm, pg.h_mm)
            if b: bad.append("%s: %s" % (pg.name, ", ".join(b[:6]) + (" ..." if len(b) > 6 else "")))
        if bad: return QMessageBox.question(self, "Outside the page", "These layers go outside the page and will be cut off:\n\n%s\n\nContinue anyway?" % "\n".join(bad)) == QMessageBox.Yes
        return True
    def missing_files(self, idx):
        return sorted({l.name for i in idx for l in self.project.pages[i].layers if l.visible and not (l.path and os.path.isfile(l.path))})
    def export_dialog(self, fmt):
        self.commit_items(); r = self.ask_pages_and_dpi("Export " + fmt.upper())
        if r is None: return
        idx, dpi, qual = r
        if not self.check_out_of_page(idx): return
        miss = self.missing_files(idx)
        if miss and QMessageBox.question(self, "Missing images", "These images are missing and will be blank:\n%s\n\nContinue?" % "\n".join(miss[:10])) != QMessageBox.Yes: return
        pages = [rc.Page.from_dict(self.project.pages[i].to_dict()) for i in idx]      # snapshot, so editing during export is safe
        if fmt == "pdf":
            p, _ = QFileDialog.getSaveFileName(self, "Save PDF", (self.project.name or "layout") + ".pdf", "PDF (*.pdf)")
            if not p: return
            def job(task):
                tmp = tempfile.mkdtemp(prefix="rl_", dir=self.cache_dir); files = []; cacheL = {}
                for n, pg in enumerate(pages):
                    task.progress.emit(n, len(pages), "Rendering %s" % pg.name)
                    if task.cancelled(): return None
                    im = rc.render_page(pg, dpi, loader=lambda L: self._loader(L, cacheL)); fp = os.path.join(tmp, "p%03d.jpg" % n); rc.save_image(im, fp, dpi, qual); files.append(fp); del im
                imgs = [Image.open(f) for f in files]; self._pdf_lazy(imgs, p, dpi)
                for i in imgs: i.close()
                return p
            run_task(self, "Exporting PDF", job, lambda res: res and QMessageBox.information(self, "PDF", "Saved %d page(s):\n%s\n(page size and order kept, %d DPI)" % (len(pages), res, dpi)))
        else:
            d = QFileDialog.getExistingDirectory(self, "Save pages into folder")
            if not d: return
            def job(task):
                out = []; cacheL = {}
                for n, pg in enumerate(pages):
                    task.progress.emit(n, len(pages), "Rendering %s" % pg.name)
                    if task.cancelled(): return out
                    im = rc.render_page(pg, dpi, loader=lambda L: self._loader(L, cacheL))
                    out.append(rc.save_image(im, unique_path(os.path.join(d, "%s_p%02d.%s" % (re.sub(r"[^\w\-]+", "_", self.project.name), n + 1, fmt))), dpi, qual)); del im
                return out
            run_task(self, "Exporting " + fmt.upper(), job, lambda res: QMessageBox.information(self, "Export", "Saved %d file(s) (%d DPI):\n%s" % (len(res), dpi, "\n".join(res[:8]))))
    @staticmethod
    def _pdf_lazy(imgs, path, dpi):
        imgs[0].save(path, "PDF", resolution=float(dpi), save_all=True, append_images=imgs[1:])
    def _loader(self, L, store):
        if L.path not in store:
            if len(store) > 6: store.clear()                              # bounded memory when many different images are used
            store[L.path] = self.pil_full(L)
        return store[L.path]

    # ---------------- print ----------------
    def print_dialog(self, preview):
        self.commit_items(); r = self.ask_pages_and_dpi("Print", with_dpi=True)
        if r is None: return
        idx, dpi, _ = r
        if not self.check_out_of_page(idx): return
        printer = QPrinter(QPrinter.HighResolution)
        pg0 = self.project.pages[idx[0]]; self._set_printer_page(printer, pg0)
        if preview:
            dlg = QPrintPreviewDialog(printer, self); dlg.setWindowTitle("Print preview - real size (%s)" % pg0.paper)
            dlg.paintRequested.connect(lambda pr: self._paint_printer(pr, idx, dpi)); dlg.resize(1100, 800); dlg.exec(); return
        dlg = QPrintDialog(printer, self); dlg.setWindowTitle("Print")
        if dlg.exec() != QDialog.Accepted: return
        self._warn_printable_area(printer, idx)
        self._paint_printer(printer, idx, dpi)
    def _set_printer_page(self, printer, pg):
        w, h = pg.w_mm, pg.h_mm; ori = QPageLayout.Landscape if w > h else QPageLayout.Portrait
        printer.setPageLayout(QPageLayout(QPageSize(QSizeF(min(w, h), max(w, h)), QPageSize.Millimeter), ori, QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter))
        printer.setFullPage(True)
    def _warn_printable_area(self, printer, idx):
        try:
            mm = printer.pageLayout().minimumMargins(); need = max(mm.left(), mm.top(), mm.right(), mm.bottom())
        except Exception: return
        risky = []
        for i in idx:
            pg = self.project.pages[i]
            for L in pg.layers:
                b = rc.rect_bounds_rotated(L.x_mm, L.y_mm, L.w_mm, L.h_mm, L.rot)
                if L.visible and need > 0.05 and (b[0] < need or b[1] < need or b[2] > pg.w_mm - need or b[3] > pg.h_mm - need): risky.append(pg.name)
                break
        if risky and need > 0.05:
            QMessageBox.information(self, "Printer margins", "This printer cannot print the outer ~%.1f mm of the paper (not borderless).\nImages touching the page edge may be clipped. Pages: %s\n\nBorderless only works on printers/drivers that support it - choose that in the printer's own settings." % (need, ", ".join(sorted(set(risky)))))
    def _paint_printer(self, printer, idx, dpi):
        """draw each page at its exact physical size: full-page mode, zero margins, target rect = mm / 25.4 * printer DPI (no 'fit to page' scaling)"""
        painter = QPainter(); first = True; cacheL = {}
        self._set_printer_page(printer, self.project.pages[idx[0]])
        if not painter.begin(printer): return QMessageBox.warning(self, "Print", "Could not start the printer.")
        try:
            res = printer.resolution()
            for n, i in enumerate(idx):
                pg = self.project.pages[i]
                if n > 0: self._set_printer_page(printer, pg); printer.newPage()
                im = rc.render_page(pg, dpi, loader=lambda L: self._loader(L, cacheL)); qi = pil_to_qimage(im)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                painter.drawImage(QRectF(0, 0, pg.w_mm / 25.4 * res, pg.h_mm / 25.4 * res), qi)
        finally: painter.end()

# =====================================================================
#  ENTRY POINTS
# =====================================================================
def open_studio(host=None):
    """called from the main program. Keeps one window alive on the host."""
    w = getattr(host, "_layout_studio", None) if host is not None else None
    if w is None or not w.isVisible():
        w = LayoutStudio(host)
        if host is not None: host._layout_studio = w
    w.show(); w.raise_(); w.activateWindow(); return w

def main():
    app = QApplication(sys.argv); w = LayoutStudio(); w.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()
