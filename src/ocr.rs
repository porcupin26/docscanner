//! Local, open-source OCR.
//!
//! * **Tesseract** (`tesseract` CLI, LSTM engine) finds every text line with a precise
//!   bounding box — that drives the PDF's invisible, searchable text layer.
//! * In **Best** mode a vision language model served by **Ollama** on this machine also
//!   transcribes the page into Markdown, which handles tables, multi-column layouts
//!   and handwriting far better than Tesseract's plain text.
//!
//! Nothing leaves the computer.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use image::{imageops, DynamicImage, ImageFormat};
use serde_json::{json, Value};

use crate::enhance::PageImage;
use crate::pdf::encode_jpeg;

pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
/// Long edge of the image handed to the vision model.
const VLM_EDGE: u32 = 1600;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OcrMode {
    /// Tesseract only: about a second per page
    Fast,
    /// Tesseract for positions + a local vision model for the text (tables, handwriting)
    Best,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcrConfig {
    pub mode: OcrMode,
    /// Tesseract language codes joined with `+`, e.g. `pol+eng`.
    pub languages: String,
    pub ollama_url: String,
    /// Ollama vision model for Best mode; `None` picks the first one installed.
    pub model: Option<String>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        let langs = tesseract_languages().unwrap_or_default();
        let languages = if langs.is_empty() {
            "eng".into()
        } else {
            // Polish first when present: it also covers plain Latin text well.
            let mut l = langs;
            l.sort_by_key(|c| (c != "pol", c != "eng", c.clone()));
            l.join("+")
        };
        Self { mode: OcrMode::Best, languages, ollama_url: DEFAULT_OLLAMA_URL.into(), model: None }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcrLine {
    pub text: String,
    /// `[x0, y0, x1, y1]` as fractions of the page width/height, origin top-left.
    pub bbox: [f32; 4],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OcrPage {
    pub markdown: String,
    pub lines: Vec<OcrLine>,
    /// What produced `markdown`, e.g. "gemma4:12b" or "Tesseract".
    pub engine: String,
    /// Set when Best mode had to fall back to Tesseract text.
    pub warning: Option<String>,
}

fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("docscanner").join("ocr.conf"))
}

pub fn load_config() -> OcrConfig {
    let mut c = OcrConfig::default();
    let Some(text) = config_path().and_then(|p| std::fs::read_to_string(p).ok()) else { return c };
    for line in text.lines() {
        match line.split_once('=') {
            Some(("mode", "fast")) => c.mode = OcrMode::Fast,
            Some(("mode", "best")) => c.mode = OcrMode::Best,
            Some(("languages", v)) if !v.is_empty() => c.languages = v.into(),
            Some(("ollama_url", v)) if !v.is_empty() => c.ollama_url = v.into(),
            Some(("model", v)) if !v.is_empty() => c.model = Some(v.into()),
            _ => {}
        }
    }
    c
}

pub fn save_config(c: &OcrConfig) {
    let Some(path) = config_path() else { return };
    let mode = if c.mode == OcrMode::Fast { "fast" } else { "best" };
    let text = format!(
        "mode={mode}\nlanguages={}\nollama_url={}\nmodel={}\n",
        c.languages,
        c.ollama_url,
        c.model.as_deref().unwrap_or("")
    );
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let _ = std::fs::write(path, text);
}

/// Tesseract version string, or an error with install instructions.
pub fn tesseract_version() -> Result<String> {
    let out = Command::new("tesseract")
        .arg("--version")
        .output()
        .context("Tesseract is not installed (sudo apt install tesseract-ocr tesseract-ocr-pol)")?;
    let text = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
    Ok(text.lines().next().unwrap_or("tesseract").trim().to_string())
}

/// Installed Tesseract language packs (excluding the orientation-only `osd`).
pub fn tesseract_languages() -> Result<Vec<String>> {
    let out = Command::new("tesseract").arg("--list-langs").output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text.lines().skip(1).map(str::trim).filter(|l| !l.is_empty() && *l != "osd").map(String::from).collect())
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(timeout).build()
}

/// Locally installed Ollama models that accept images (remote `:cloud` models are skipped).
pub fn vision_models(url: &str) -> Result<Vec<String>> {
    let a = agent(Duration::from_secs(10));
    let tags: Value = a
        .get(&format!("{url}/api/tags"))
        .call()
        .with_context(|| format!("Ollama is not reachable at {url}"))?
        .into_json()?;
    let mut models = Vec::new();
    for m in tags["models"].as_array().map(|a| a.as_slice()).unwrap_or_default() {
        let Some(name) = m["name"].as_str() else { continue };
        if name.ends_with(":cloud") || name.ends_with("-cloud") || m.get("remote_host").is_some() {
            continue;
        }
        let info: Value = match a.post(&format!("{url}/api/show")).send_json(json!({ "model": name })) {
            Ok(r) => r.into_json().unwrap_or_default(),
            Err(_) => continue,
        };
        let vision = info["capabilities"].as_array().is_some_and(|c| c.iter().any(|v| v == "vision"));
        if vision {
            models.push(name.to_string());
        }
    }
    Ok(models)
}

/// Groups Tesseract's word-level TSV into lines (boxes normalised to the image size)
/// and paragraphs of plain text.
fn parse_tsv(tsv: &str, img_w: f32, img_h: f32) -> (Vec<OcrLine>, String) {
    struct Acc {
        key: (u32, u32, u32),
        words: Vec<String>,
        bbox: [f32; 4],
    }
    let mut lines: Vec<Acc> = Vec::new();
    for row in tsv.lines().skip(1) {
        let f: Vec<&str> = row.split('\t').collect();
        if f.len() < 12 || f[0] != "5" {
            continue;
        }
        let text = f[11].trim();
        let conf: f32 = f[10].parse().unwrap_or(-1.0);
        if text.is_empty() || conf < 0.0 {
            continue;
        }
        let n = |i: usize| f[i].parse::<f32>().unwrap_or(0.0);
        let key = (n(2) as u32, n(3) as u32, n(4) as u32);
        let b = [n(6), n(7), n(6) + n(8), n(7) + n(9)];
        match lines.last_mut() {
            Some(l) if l.key == key => {
                l.words.push(text.to_string());
                l.bbox = [l.bbox[0].min(b[0]), l.bbox[1].min(b[1]), l.bbox[2].max(b[2]), l.bbox[3].max(b[3])];
            }
            _ => lines.push(Acc { key, words: vec![text.to_string()], bbox: b }),
        }
    }
    let mut text = String::new();
    let mut prev_par = None;
    for l in &lines {
        let par = (l.key.0, l.key.1);
        if prev_par.is_some() {
            text += if prev_par == Some(par) { "\n" } else { "\n\n" };
        }
        text += &l.words.join(" ");
        prev_par = Some(par);
    }
    let out = lines
        .into_iter()
        .map(|l| OcrLine {
            text: l.words.join(" "),
            bbox: [l.bbox[0] / img_w, l.bbox[1] / img_h, l.bbox[2] / img_w, l.bbox[3] / img_h].map(|v| v.clamp(0.0, 1.0)),
        })
        .collect();
    (out, text)
}

const VLM_PROMPT: &str = "\
Transcribe this scanned document page exactly, as Markdown.
- Keep the original language, spelling, diacritics, punctuation and numbers. Do not translate, \
summarise, correct or complete anything.
- Follow the natural reading order. Use headings and lists where the layout shows them and \
Markdown tables for tabular content.
- Write [illegible] for text you cannot read.
Output only the transcription, with no comments before or after it.";

/// The GPU serves one page at a time; queueing here keeps requests from timing out.
static VLM_LOCK: Mutex<()> = Mutex::new(());

pub struct OcrEngine {
    cfg: OcrConfig,
}

impl OcrEngine {
    pub fn new(cfg: OcrConfig) -> Self {
        Self { cfg }
    }

    pub fn recognize(&self, page: &PageImage) -> Result<OcrPage> {
        let img = match page {
            PageImage::Color(i) => DynamicImage::ImageRgb8(i.clone()),
            PageImage::Gray(i) | PageImage::Bilevel(i) => DynamicImage::ImageLuma8(i.clone()),
        };
        let (lines, plain) = self.tesseract(&img)?;
        let mut result = OcrPage { markdown: plain, lines, engine: "Tesseract".into(), warning: None };
        if self.cfg.mode == OcrMode::Best {
            match self.vision_model(&img) {
                Ok((model, md)) => {
                    result.markdown = md;
                    result.engine = model;
                }
                Err(e) => result.warning = Some(format!("AI model unavailable, used Tesseract text: {e:#}")),
            }
        }
        Ok(result)
    }

    fn tesseract(&self, img: &DynamicImage) -> Result<(Vec<OcrLine>, String)> {
        let mut png = Vec::new();
        img.to_luma8().write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png)?;
        let mut child = Command::new("tesseract")
            .args(["stdin", "stdout", "-l", &self.cfg.languages, "--psm", "3", "tsv"])
            // Pages already run in parallel; keep each Tesseract on few threads.
            .env("OMP_THREAD_LIMIT", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Tesseract is not installed (sudo apt install tesseract-ocr tesseract-ocr-pol)")?;
        let mut stdin = child.stdin.take().unwrap();
        let writer = std::thread::spawn(move || stdin.write_all(&png));
        let out = child.wait_with_output()?;
        let _ = writer.join();
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            bail!("Tesseract failed: {}", err.lines().last().unwrap_or("unknown error"));
        }
        Ok(parse_tsv(&String::from_utf8_lossy(&out.stdout), img.width() as f32, img.height() as f32))
    }

    fn vision_model(&self, img: &DynamicImage) -> Result<(String, String)> {
        let url = &self.cfg.ollama_url;
        let model = match &self.cfg.model {
            Some(m) => m.clone(),
            None => vision_models(url)?.into_iter().next().context("no vision model installed in Ollama (try: ollama pull gemma4:12b)")?,
        };
        let small = img.resize(VLM_EDGE, VLM_EDGE, imageops::FilterType::CatmullRom);
        let jpeg = encode_jpeg(
            &match small {
                DynamicImage::ImageLuma8(g) => PageImage::Gray(g),
                other => PageImage::Color(other.to_rgb8()),
            },
            92,
        )?;
        let mut body = json!({
            "model": model,
            "stream": false,
            "think": false,
            "options": { "temperature": 0, "num_ctx": 8192, "num_predict": 4096 },
            "messages": [{ "role": "user", "content": VLM_PROMPT, "images": [base64(&jpeg)] }]
        });
        let _gpu = VLM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let a = agent(Duration::from_secs(900));
        let send = |body: &Value| a.post(&format!("{url}/api/chat")).send_json(body.clone());
        let resp = match send(&body) {
            // Models without a thinking switch reject the `think` field.
            Err(ureq::Error::Status(400, _)) => {
                body.as_object_mut().unwrap().remove("think");
                send(&body)
            }
            r => r,
        };
        let v: Value = match resp {
            Ok(r) => r.into_json()?,
            Err(ureq::Error::Status(code, r)) => {
                let msg = r.into_json::<Value>().ok().and_then(|v| v["error"].as_str().map(String::from)).unwrap_or_default();
                bail!("Ollama error {code}: {msg}");
            }
            Err(e) => bail!("Ollama is not reachable at {url}: {e}"),
        };
        let md = clean_markdown(v["message"]["content"].as_str().unwrap_or_default());
        if md.is_empty() {
            bail!("{model} returned no text");
        }
        Ok((model, md))
    }
}

/// Strips a wrapping ```markdown fence that some models add.
fn clean_markdown(s: &str) -> String {
    let t = s.trim();
    let t = t.strip_prefix("```markdown").or_else(|| t.strip_prefix("```md")).or_else(|| t.strip_prefix("```")).map_or(t, |r| {
        r.trim_end().strip_suffix("```").unwrap_or(r)
    });
    t.trim().to_string()
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// All pages' transcriptions as one Markdown document.
pub fn markdown_document(ocr: &[Option<OcrPage>]) -> String {
    ocr.iter()
        .enumerate()
        .map(|(i, o)| match o {
            Some(o) => format!("<!-- page {} -->\n\n{}\n", i + 1, o.markdown),
            None => format!("<!-- page {}: text recognition failed -->\n", i + 1),
        })
        .collect::<Vec<_>>()
        .join("\n---\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648() {
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn groups_tesseract_words_into_lines() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
            1\t1\t0\t0\t0\t0\t0\t0\t1000\t2000\t-1\t\n\
            5\t1\t1\t1\t1\t1\t100\t50\t80\t30\t96\tFaktura\n\
            5\t1\t1\t1\t1\t2\t200\t52\t60\t30\t95\tVAT\n\
            5\t1\t1\t1\t2\t1\t100\t100\t120\t30\t91\tZażółć\n\
            5\t1\t2\t1\t1\t1\t100\t300\t50\t30\t90\tKraków\n";
        let (lines, text) = parse_tsv(tsv, 1000.0, 2000.0);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].text, "Faktura VAT");
        assert_eq!(lines[0].bbox, [0.1, 0.025, 0.26, 0.041]);
        assert_eq!(text, "Faktura VAT\nZażółć\n\nKraków");
    }

    #[test]
    fn strips_code_fences() {
        assert_eq!(clean_markdown("```markdown\n# Hi\n```"), "# Hi");
        assert_eq!(clean_markdown("# Hi"), "# Hi");
    }
}
