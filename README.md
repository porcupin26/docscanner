# docscanner

Turns phone photos of documents into clean, flat, print-ready scans — the core
of apps like Adobe Scan, Microsoft Lens or CamScanner — in pure Rust (no OpenCV).

## Install

Download a package for your distribution from the
[latest release](https://github.com/porcupin26/docscanner/releases/latest) (x86_64 and aarch64):

| Distribution | File | Install |
|---|---|---|
| Debian, Ubuntu, Mint, Pop!_OS | `docscanner_*.deb` | `sudo apt install ./docscanner_*.deb` |
| Fedora, RHEL, openSUSE | `docscanner-*.rpm` | `sudo dnf install ./docscanner-*.rpm` (openSUSE: `sudo zypper install ./docscanner-*.rpm`) |
| Arch, Manjaro, EndeavourOS | `docscanner-*.pkg.tar.zst` | `sudo pacman -U docscanner-*.pkg.tar.zst` (x86_64; `packaging/arch/PKGBUILD` builds it anywhere) |
| Any distribution | `DocScanner-*.AppImage` | `chmod +x DocScanner-*.AppImage && ./DocScanner-*.AppImage` (append `cli …` for the command line) |
| Any distribution | `docscanner-*-linux.tar.gz` | unpack and run `docscanner-gui` or `docscanner` |

The packages install `docscanner` (command line) and `docscanner-gui` (desktop app, also in your app menu).
Scanning needs SANE (`sane-utils` / `sane-backends`), and text recognition needs Tesseract (see below).

From source: `cargo install --git https://github.com/porcupin26/docscanner`.

## Pipeline

1. **Load** – any JPEG/PNG/TIFF/WebP/BMP, EXIF orientation applied.
2. **Detect the page** – multi-channel Canny edges plus three segmentation strategies
   (closed edge contour, brightness, "whiteness") each propose an outline → convex hull →
   largest inscribed quadrilateral → sides snapped to edge pixels with robust sub-pixel
   line fitting → candidates scored by edge support along the perimeter.
3. **Flatten** – homography warp with bilinear sampling; the true aspect ratio is
   recovered from the camera geometry (Zhang & He), so tilted shots aren't stretched.
4. **Remove shadows** – local paper colour estimated on a coarse grid and divided out:
   blank paper becomes pure white, colour casts and hand shadows disappear.
5. **Filter** – `auto` (colour or grey by content), `color` (magic colour), `gray`,
   `bw` (Sauvola adaptive threshold), `original`.
6. **Deskew** – projection-profile text-angle estimate (automatic for flatbed/no-edge images).
7. **OCR (optional, fully local)** – Tesseract 5 finds text lines with exact positions (for the
   searchable layer); in *best* mode a vision LLM served by Ollama (e.g. `gemma4:12b`,
   `qwen3.5:9b`) writes the Markdown transcription — tables, columns, handwriting.
8. **Export** – multi-page PDF (JPEG pages, B&W pages as 1-bit Flate bitmaps, invisible Unicode
   text layer when OCR ran), images, or Markdown text.

Pages are processed in parallel.

## Desktop app

```sh
cargo run --release --bin docscanner-gui [photos...]
```

- **Scan…** (Ctrl+N): pick a SANE scanner (flatbed or document feeder), mode, DPI and paper size; pages
  appear in the list as they finish scanning. Requires `scanimage` (sane-utils).
- Drop photos (or folders) onto the window, or **Open…** (Ctrl+O).
- Each page is auto-detected; drag the corner handles to adjust (a magnifier
  appears while dragging). **Auto-detect** restores the detection, **Full image** disables cropping.
- Pick a filter per page (or *apply to all*), rotate, toggle *Straighten text* — the
  result pane updates live.
- Reorder/remove pages in the sidebar (↑/↓ to navigate, Del to remove).
- **Export PDF…** (Ctrl+E), **Images…** or **Text…** with page size, DPI and JPEG quality from the toolbar.
- **OCR (local)**: the *🔤 Text* tab transcribes a page on your machine. Tick *Searchable (OCR)*
  to embed an invisible text layer so the PDF can be searched and copied from. ⚙ picks the mode
  (Best = Tesseract + a local vision model via Ollama, Fast = Tesseract only), the model and the
  languages.

File dialogs use the XDG desktop portal (falls back to zenity); no GTK needed.
Build only the CLI with `cargo build --release --no-default-features`.

## Command line

```sh
cargo build --release

# Multi-page PDF from photos (or a directory of photos, ordered by name)
docscanner IMG_001.jpg IMG_002.jpg -o contract.pdf

# Black & white receipt, A4 pages, 200 dpi
docscanner receipts/ -o receipts.pdf -f bw --page-size a4 --dpi 200

# Per-page images into a folder, plus detection overlays for troubleshooting
docscanner photos/ -o scans/ --debug debug/

# Searchable PDF + Markdown transcript with local OCR (Tesseract + Ollama vision model)
docscanner scans/ -o contract.pdf --ocr --text contract.md
docscanner scans/ -o contract.pdf --ocr fast --ocr-lang pol+eng      # Tesseract only

# Detection missed? Give the corners yourself (source pixel coordinates, any order)
docscanner whiteboard.jpg -o board.png --corners "120,80 1900,95 1950,2600 90,2580"
```

Run `docscanner --help` for all options (`--no-crop`, `--rotate`, `--deskew`, `--quality`).

## OCR requirements

```sh
sudo apt install tesseract-ocr tesseract-ocr-pol   # + other tesseract-ocr-<lang> packs
ollama pull gemma4:12b                              # any Ollama model with vision works
```

Settings are saved in `~/.config/docscanner/ocr.conf`.
