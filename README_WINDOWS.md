# Ripon Computer – Layout Studio upgrade: Windows 10/11 guide

## 1. Files (put ALL in one folder, e.g. C:\RiponComputer\)
Photostudio(1).py · ripon_ai_studio.py (unchanged) · ripon_core.py · ripon_layout_studio.py · ripon_studio_base.py · ripon_studio_tools.py · ripon_studio_panels.py

## 2. Install
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
Run:  `py "Photostudio(1).py"`   →  menu **Studio ▸ Layout Studio** (Ctrl+Shift+L), or **Studio ▸ Send this photo to Layout Studio**.
Standalone Layout Studio (without the photo program): `py ripon_layout_studio.py`

## 3. Where things are stored (nothing in your existing data is touched)
%LOCALAPPDATA%\RiponComputer\  → settings.json, presets.json, shop.db, jobs\ (unchanged)  +  NEW: layout_presets.json, layout_cache\ (cropped/restored copies), recovery\layout_autosave.json, models\document\ (document models)
Original image files are never overwritten. Crops / restorations are new files; Tools ▸ "Revert selected to original file" goes back.

## 4. Using it (short)
* **Import** many photos (Ctrl+I) or drag files onto the canvas. Each image = one layer. Drag to move; yellow handle = rotate (Shift = 15° steps); square handle = resize (Shift = free ratio). Arrow keys nudge (Shift 10 mm, Ctrl 0.1 mm). Space+drag or middle-drag pans, Ctrl+wheel zooms (print size never changes).
* **Pages**: left panel thumbnails; Page menu: new / duplicate / rename / delete / reorder / setup (A3 A4 A5 A6 Letter Legal Custom, portrait/landscape, margins, background). Drag layers from the Layers list onto a page thumbnail to move them (Ctrl = copy), or use Arrange ▸ Move/Copy to page.
* **Bank Photo Duet** tab: default 1.9 × 1.5 in = 570 × 450 px @300 DPI (edit size, gap, margins, number of duets, side/stack/auto, same photo twice or two photos, save presets). "Build layout" creates normal editable layers.
* **ID Card** tab: card size (mm/inch), front + back, gap, stack/side, many cards per page. **Passport Sheet** tab: size, copies, spacing, margins, "copies = max that fit", spills to more pages.
* **Tools ▸ Crop / Perspective crop** (4 draggable corners, auto edge detection, straighten), **Auto crop selected (batch)**, **Document restoration** (before/after slider, difference map, batch).
* **Print & Export**: JPG / PNG / multi-page PDF with exact DPI tag; Print preview and Print at real size. A warning appears if anything is outside the page.
* **Save project** (.rproj) keeps all pages/layers/positions (+ embedded copies of the originals; if a file is missing on reopen you can relink). Autosave every 30 s; recovery offered after a crash.

## 5. Document models / OCR
Tools ▸ "Document models / OCR setup". All crop/deskew/restoration features use OpenCV and need **no download**.
No model download links are bundled (they could not be verified offline). To add a model put an entry in `%LOCALAPPDATA%\RiponComputer\models\doc_models.json` (button "Write example" creates a template with url, file, sha256, licence). Downloads then have progress, Cancel, Retry (resume), checksum check and a clear message for 404 / offline.
OCR (optional, view-only): install Tesseract-OCR (UB-Mannheim Windows build, tick Bengali) + `pip install pytesseract`.
GPU: `pip uninstall onnxruntime` then `pip install onnxruntime-directml` (any GPU) or `onnxruntime-gpu` (NVIDIA). Without a GPU everything runs on CPU.

## 6. Printing notes
Page is sent at full size with zero margins and no scaling: choose the same paper size in the printer dialog and "Actual size / 100%". Most printers cannot print the outer few mm – the program warns about it. Borderless works only if your printer/driver supports it.
