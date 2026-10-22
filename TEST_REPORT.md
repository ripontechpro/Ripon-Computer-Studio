# Test report (honest)

## Actually run here (Linux sandbox, Python 3.12) – ALL PASS
* `python test_core.py` – **76 tests** on ripon_core.py: paper sizes (A3/A4/A5/A6/Letter/Legal/Custom), 1.9×1.5 in = 570×450 px, duet / ID card / passport layouts (no overlap, inside margins, gaps exact), out-of-page warning (incl. rotated layers), project save/reopen (embedded originals, relink, autosave recovery), exact-size rendering, JPG/PNG DPI metadata, multi-page PDF (A4 page size/order), original file untouched, edge detection (corners within 4 px on a synthetic perspective photo), perspective warp, deskew (−3° recovered), restoration (uneven light flattened, background whitened, noise reduced, content retention ≥ 0.99, red seal kept, B&W, gray, faint text, dust vs. dots near text), downloader (ok, SHA-256, no re-download, 404 message, resume after interrupted connection, bad checksum rejected, cancel keeps .part, offline message).
* `python test_qt_free.py` – 13 tests: all Qt modules **import** (with a fake Qt), page-range parser, non-destructive cover-crop, panel→layer conversion (rotated, contain), end-to-end: a duet photo renders at exactly **570 × 450 px @300 DPI**.
* `python -m py_compile` on all files; undefined-name static check.

## NOT run (no PySide6 / Windows / printer / GPU / Tesseract in the sandbox) – please test on your PC
Application start-up; every widget, mouse drag, handles (move/resize/rotate), multi-select, group/ungroup, rulers/guides/snap, page tabs & thumbnails, drag layers to another page, undo/redo in the UI, Crop & Perspective dialog interaction, Restoration dialog (preview/batch threads), Models dialog, Print Preview and real printing, borderless printing, GPU acceleration, OCR, existing AI Studio / photo workflow after the menu patch (only 2 menu lines + 2 methods were added to Photostudio(1).py; nothing removed).
No AI document model was downloaded or run (no verified URLs offline): the built-in OpenCV methods are used; nothing claims otherwise.

## Added in the final step (also NOT run on real Qt)
* Brightness / contrast / grey sliders per layer (canvas preview + export use the same maths; core part tested)
* Group scale handle (orange square) for multi-selection
* Drop a dragged layer on another page's thumbnail (Ctrl = copy)
* Document Studio button: 'Send pages to Layout Studio (one page each)'

## Known limits
* Only one page canvas is visible at a time; moving layers between pages = drop on the thumbnail, Layers list drag, or Arrange menu.
* Dewarping of curved pages (book pages) is not implemented (needs a model); flat perspective + deskew only.
