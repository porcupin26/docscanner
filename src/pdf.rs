//! Minimal multi-page PDF writer: one full-page image per page.
//! Colour/grey pages are embedded as JPEG (DCTDecode), black & white pages as
//! 1-bit Flate-compressed bitmaps, which keeps text-only scans tiny and crisp.
//! OCR results become an invisible text layer so the PDF is searchable and
//! copyable; text uses a Unicode (Identity-H) font so any script survives.

use std::io::Write;

use anyhow::Result;
use clap::ValueEnum;
use flate2::{write::ZlibEncoder, Compression};
use image::{codecs::jpeg::JpegEncoder, ExtendedColorType};

use crate::enhance::PageImage;
use crate::ocr::OcrPage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PageSize {
    /// Page matches the scan's aspect ratio (long side = A4 height)
    Fit,
    A4,
    Letter,
}

const A4_IN: (f64, f64) = (8.27, 11.69);
const LETTER_IN: (f64, f64) = (8.5, 11.0);

impl PageSize {
    /// Physical page (width, height) in inches for an image of the given pixel size.
    pub fn page_inches(self, img_w: u32, img_h: u32) -> (f64, f64) {
        let landscape = img_w > img_h;
        let orient = |(a, b): (f64, f64)| if landscape { (b, a) } else { (a, b) };
        match self {
            PageSize::A4 => orient(A4_IN),
            PageSize::Letter => orient(LETTER_IN),
            PageSize::Fit => {
                let s = A4_IN.1 / img_w.max(img_h) as f64;
                (img_w as f64 * s, img_h as f64 * s)
            }
        }
    }

    /// Scale factor (<= 1) bringing the image down to `dpi` on this page.
    pub fn downscale_factor(self, img_w: u32, img_h: u32, dpi: f64) -> f64 {
        let (pw, ph) = self.page_inches(img_w, img_h);
        (pw * dpi / img_w as f64).min(ph * dpi / img_h as f64).min(1.0)
    }
}

pub fn encode_jpeg(page: &PageImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
    match page {
        PageImage::Color(i) => enc.encode(i.as_raw(), i.width(), i.height(), ExtendedColorType::Rgb8)?,
        PageImage::Gray(i) | PageImage::Bilevel(i) => enc.encode(i.as_raw(), i.width(), i.height(), ExtendedColorType::L8)?,
    }
    Ok(buf)
}

struct ImageObject {
    dict: String,
    data: Vec<u8>,
}

fn image_object(page: &PageImage, quality: u8) -> Result<ImageObject> {
    let (w, h) = page.dimensions();
    Ok(match page {
        PageImage::Bilevel(img) => {
            let row_bytes = (w as usize).div_ceil(8);
            let mut bits = vec![0u8; row_bytes * h as usize];
            for (x, y, p) in img.enumerate_pixels() {
                if p[0] > 127 {
                    bits[y as usize * row_bytes + x as usize / 8] |= 0x80 >> (x % 8);
                }
            }
            let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
            z.write_all(&bits)?;
            ImageObject {
                dict: format!("/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /FlateDecode /Width {w} /Height {h}"),
                data: z.finish()?,
            }
        }
        _ => {
            let cs = if matches!(page, PageImage::Color(_)) { "DeviceRGB" } else { "DeviceGray" };
            ImageObject {
                dict: format!("/ColorSpace /{cs} /BitsPerComponent 8 /Filter /DCTDecode /Width {w} /Height {h}"),
                data: encode_jpeg(page, quality)?,
            }
        }
    })
}

/// Invisible text (render mode 3) laid over the page image, one run per OCR line,
/// stretched horizontally to span the line's bounding box.
fn text_layer(ocr: &OcrPage, dx: f64, dy: f64, dw: f64, dh: f64) -> String {
    let mut out = String::from("BT 3 Tr\n");
    for line in &ocr.lines {
        let [x0, y0, x1, y1] = line.bbox.map(|v| v as f64);
        let units: Vec<u16> = line.text.encode_utf16().collect();
        if units.is_empty() {
            continue;
        }
        let h = (y1 - y0) * dh;
        let size = (h * 0.85).max(1.0);
        // The font's glyphs are all 0.5 em wide.
        let natural = units.len() as f64 * 0.5 * size;
        let stretch = ((x1 - x0) * dw / natural * 100.0).clamp(1.0, 10_000.0);
        let (x, y) = (dx + x0 * dw, dy + (1.0 - y1) * dh + h * 0.2);
        let hex: String = units.iter().map(|u| format!("{u:04X}")).collect();
        out += &format!("/F1 {size:.2} Tf {stretch:.2} Tz 1 0 0 1 {x:.2} {y:.2} Tm <{hex}> Tj\n");
    }
    out + "ET"
}

/// CMap mapping every 2-byte code to the same Unicode code unit.
fn identity_to_unicode() -> String {
    let mut s = String::from(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n",
    );
    let ranges: Vec<u32> = (0..256).collect();
    for chunk in ranges.chunks(100) {
        s += &format!("{} beginbfrange\n", chunk.len());
        for hi in chunk {
            s += &format!("<{hi:02X}00> <{hi:02X}FF> <{hi:02X}00>\n");
        }
        s += "endbfrange\n";
    }
    s + "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
}

/// Serialises the pages into a PDF document. `ocr` may be empty or hold a
/// transcription per page; pages with one get a searchable text layer.
pub fn write_pdf(pages: &[PageImage], ocr: &[Option<OcrPage>], size: PageSize, quality: u8) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets: Vec<usize> = Vec::new();
    let obj = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, head: &str, stream: Option<&[u8]>| {
        offsets.push(out.len());
        let id = offsets.len();
        out.extend_from_slice(format!("{id} 0 obj\n{head}\n").as_bytes());
        if let Some(s) = stream {
            out.extend_from_slice(b"stream\n");
            out.extend_from_slice(s);
            out.extend_from_slice(b"\nendstream\n");
        }
        out.extend_from_slice(b"endobj\n");
    };

    let n = pages.len();
    let has_text = ocr.iter().any(|o| o.as_ref().is_some_and(|o| !o.lines.is_empty()));
    // Object layout: 1 catalog, 2 page tree, 3 info, then (page, contents, image) per page,
    // then the shared OCR font (Type0, CIDFont, descriptor, ToUnicode).
    let font = 4 + 3 * n;
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 4 + 3 * i)).collect();
    obj(&mut out, &mut offsets, "<< /Type /Catalog /Pages 2 0 R >>", None);
    obj(&mut out, &mut offsets, &format!("<< /Type /Pages /Kids [{}] /Count {n} >>", kids.join(" ")), None);
    obj(&mut out, &mut offsets, "<< /Producer (docscanner) /Creator (docscanner) >>", None);

    for (i, page) in pages.iter().enumerate() {
        let (w, h) = page.dimensions();
        let (pw_in, ph_in) = size.page_inches(w, h);
        let (pw, ph) = (pw_in * 72.0, ph_in * 72.0);
        let s = (pw / w as f64).min(ph / h as f64);
        let (dw, dh) = (w as f64 * s, h as f64 * s);
        let (dx, dy) = ((pw - dw) / 2.0, (ph - dh) / 2.0);
        let mut content = format!("q {dw:.3} 0 0 {dh:.3} {dx:.3} {dy:.3} cm /Im0 Do Q");
        if let Some(Some(text)) = ocr.get(i) {
            content += "\n";
            content += &text_layer(text, dx, dy, dw, dh);
        }
        let fonts = if has_text { format!(" /Font << /F1 {font} 0 R >>") } else { String::new() };
        let base = 4 + 3 * i;
        obj(
            &mut out,
            &mut offsets,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.3} {ph:.3}] /Resources << /XObject << /Im0 {} 0 R >>{fonts} >> /Contents {} 0 R >>",
                base + 2,
                base + 1
            ),
            None,
        );
        obj(&mut out, &mut offsets, &format!("<< /Length {} >>", content.len()), Some(content.as_bytes()));
        let img = image_object(page, quality)?;
        obj(
            &mut out,
            &mut offsets,
            &format!("<< /Type /XObject /Subtype /Image {} /Length {} >>", img.dict, img.data.len()),
            Some(&img.data),
        );
    }

    if has_text {
        obj(
            &mut out,
            &mut offsets,
            &format!(
                "<< /Type /Font /Subtype /Type0 /BaseFont /GlyphLessFont /Encoding /Identity-H /DescendantFonts [{} 0 R] /ToUnicode {} 0 R >>",
                font + 1,
                font + 3
            ),
            None,
        );
        obj(
            &mut out,
            &mut offsets,
            &format!(
                "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /GlyphLessFont /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor {} 0 R /DW 500 /CIDToGIDMap /Identity >>",
                font + 2
            ),
            None,
        );
        obj(
            &mut out,
            &mut offsets,
            "<< /Type /FontDescriptor /FontName /GlyphLessFont /Flags 5 /FontBBox [0 0 500 1000] /ItalicAngle 0 /Ascent 1000 /Descent 0 /CapHeight 1000 /StemV 80 >>",
            None,
        );
        let cmap = identity_to_unicode();
        obj(&mut out, &mut offsets, &format!("<< /Length {} >>", cmap.len()), Some(cmap.as_bytes()));
    }

    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes());
    for o in &offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R /Info 3 0 R >>\nstartxref\n{xref}\n%%EOF\n", offsets.len() + 1).as_bytes(),
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::OcrLine;
    use image::{GrayImage, Luma};

    #[test]
    fn searchable_text_round_trips_through_pdftotext() {
        let page = PageImage::Gray(GrayImage::from_pixel(850, 1100, Luma([255])));
        let ocr = OcrPage {
            markdown: String::new(),
            engine: String::new(),
            warning: None,
            lines: vec![
                OcrLine { text: "Faktura VAT nr 12/2026".into(), bbox: [0.1, 0.05, 0.6, 0.08] },
                OcrLine { text: "Zażółć gęślą jaźń — 1 234,56 zł".into(), bbox: [0.1, 0.2, 0.9, 0.23] },
            ],
        };
        let bytes = write_pdf(&[page], &[Some(ocr)], PageSize::Letter, 85).unwrap();
        let path = std::env::temp_dir().join(format!("docscanner-text-{}.pdf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let Ok(out) = std::process::Command::new("pdftotext").args([path.to_str().unwrap(), "-"]).output() else {
            return; // poppler not installed
        };
        if std::env::var_os("KEEP_PDF").is_none() { let _ = std::fs::remove_file(&path); }
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("Faktura VAT nr 12/2026"), "{text}");
        assert!(text.contains("Zażółć gęślą jaźń — 1 234,56 zł"), "{text}");
    }
}
