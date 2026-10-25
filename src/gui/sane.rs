//! Scanner access through SANE's `scanimage` front-end (package `sane-utils`).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    pub name: String,
    pub label: String,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceOptions {
    pub sources: Vec<String>,
    pub modes: Vec<String>,
    pub resolutions: Vec<u32>,
    /// Largest scan area (width, height) in mm, when the device lets us set it.
    pub max_area: Option<(f64, f64)>,
}

/// Paper sizes offered for the scan area, in mm.
pub const PAPER_SIZES: &[(&str, f64, f64)] = &[("Letter", 215.9, 279.4), ("A4", 210.0, 297.0), ("Legal", 215.9, 355.6)];
pub const FULL_AREA: &str = "Full scan area";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanSettings {
    pub device: String,
    pub source: Option<String>,
    pub mode: Option<String>,
    pub resolution: Option<u32>,
    pub paper: Option<String>,
}

impl ScanSettings {
    pub fn uses_feeder(&self) -> bool {
        self.source.as_deref().is_some_and(|s| {
            let s = s.to_ascii_lowercase();
            s.contains("adf") || s.contains("feeder")
        })
    }
}

/// Handle to a running scan; `cancel` kills the `scanimage` process.
pub type ScanHandle = Arc<Mutex<Option<Child>>>;

pub fn cancel(handle: &ScanHandle) {
    if let Some(child) = handle.lock().unwrap().as_mut() {
        let _ = child.kill();
    }
}

fn backend(name: &str) -> &str {
    name.split(':').next().unwrap_or("")
}

/// Network-native drivers first, webcams (v4l) last.
fn rank(name: &str) -> u8 {
    match backend(name) {
        "escl" | "airscan" => 0,
        "v4l" => 3,
        _ if name.contains("/net/") => 2,
        _ => 1,
    }
}

fn last_error(stderr: &str) -> String {
    stderr
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty() && !l.contains("Output format is not set") && !l.contains("rounded value"))
        .map(|l| l.trim_start_matches("scanimage: ").to_string())
        .unwrap_or_else(|| "scanner reported an error".into())
}

pub fn list_devices() -> Result<Vec<Device>, String> {
    let out = Command::new("scanimage")
        .args(["-f", "%d\t%v\t%m\t%t%n"])
        .output()
        .map_err(|e| format!("cannot run scanimage ({e}) — install the sane-utils package"))?;
    let mut devices: Vec<Device> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            let (name, vendor, model, kind) = (f.next()?, f.next()?, f.next()?, f.next().unwrap_or(""));
            let vendor = if vendor == "Noname" { "" } else { vendor };
            let label = format!("{vendor} {} — {kind}, {}", model.replace('_', " "), backend(name));
            Some(Device { name: name.to_string(), label: label.trim().to_string() })
        })
        .collect();
    devices.sort_by_key(|d| rank(&d.name));
    Ok(devices)
}

/// Parses the value list of an option line such as `--source Flatbed|ADF [Flatbed]`.
fn option_values(line: &str, name: &str) -> Option<Vec<String>> {
    let rest = line.trim().strip_prefix(name)?.strip_prefix(' ')?;
    if rest.contains("[inactive]") {
        return None;
    }
    let values = rest.rfind(" [").map_or(rest, |i| &rest[..i]);
    Some(values.split('|').map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect())
}

fn parse_resolutions(values: &[String]) -> Vec<u32> {
    let joined = values.join("|");
    let joined = joined.trim_end_matches("dpi");
    if let Some((lo, hi)) = joined.split_once("..") {
        // Range form, e.g. "75..1200dpi (in steps of 1)": offer the usual stops.
        let num = |s: &str| s.trim().split(|c: char| !c.is_ascii_digit()).next().and_then(|n| n.parse::<u32>().ok());
        let (lo, hi) = (num(lo).unwrap_or(75), num(hi).unwrap_or(600));
        return [75, 100, 150, 200, 300, 400, 600, 1200].into_iter().filter(|r| (lo..=hi).contains(r)).collect();
    }
    joined.split('|').filter_map(|v| v.trim().trim_end_matches("dpi").parse().ok()).collect()
}

/// Range maximum of a geometry option line such as `-y 0..381mm [381]`.
fn range_max(line: &str, name: &str) -> Option<f64> {
    let rest = line.trim().strip_prefix(name)?.strip_prefix(' ')?;
    let (_, hi) = rest.split_once("..")?;
    hi.trim_start().split(|c: char| !(c.is_ascii_digit() || c == '.')).next()?.parse().ok()
}

/// Options depend on the source (feeders often allow fewer resolutions), so pass it when known.
pub fn query_options(device: &str, source: Option<&str>) -> Result<DeviceOptions, String> {
    let mut cmd = Command::new("scanimage");
    cmd.args(["-d", device]);
    if let Some(s) = source {
        cmd.args(["--source", s]);
    }
    let out = cmd
        .arg("-A")
        .output()
        .map_err(|e| format!("cannot run scanimage: {e}"))?;
    if !out.status.success() {
        return Err(last_error(&String::from_utf8_lossy(&out.stderr)));
    }
    let mut opts = DeviceOptions::default();
    let (mut max_x, mut max_y) = (None, None);
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        max_x = max_x.or_else(|| range_max(line, "-x"));
        max_y = max_y.or_else(|| range_max(line, "-y"));
        if let Some(v) = option_values(line, "--source") {
            opts.sources = v;
        } else if let Some(v) = option_values(line, "--mode") {
            opts.modes = v;
        } else if let Some(v) = option_values(line, "--resolution") {
            opts.resolutions = parse_resolutions(&v);
        }
    }
    opts.max_area = max_x.zip(max_y);
    Ok(opts)
}

/// Scan-area size in mm for the chosen paper, clamped to what the device can do.
pub fn paper_area(settings: &ScanSettings, opts: &DeviceOptions) -> Option<(f64, f64)> {
    let (mx, my) = opts.max_area?;
    let name = settings.paper.as_deref()?;
    let &(_, w, h) = PAPER_SIZES.iter().find(|(n, _, _)| *n == name)?;
    Some((w.min(mx), h.min(my)))
}

/// Scans into `dir` as `scan-NNN.png`, numbering from `first_page`. A feeder source
/// scans until it runs empty; a flatbed scans one page. `on_page` gets each finished
/// file as soon as it is written. Returns the number of pages scanned.
pub fn scan(
    settings: &ScanSettings,
    area: Option<(f64, f64)>,
    dir: &Path,
    first_page: u32,
    handle: &ScanHandle,
    mut on_page: impl FnMut(PathBuf),
) -> Result<u32, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut cmd = Command::new("scanimage");
    cmd.args(["-d", &settings.device]);
    if let Some(s) = &settings.source {
        cmd.args(["--source", s]);
    }
    if let Some(m) = &settings.mode {
        cmd.args(["--mode", m]);
    }
    if let Some(r) = settings.resolution {
        cmd.arg(format!("--resolution={r}"));
    }
    if let Some((w, h)) = area {
        cmd.args(["-l", "0", "-t", "0", "-x", &format!("{w}"), "-y", &format!("{h}")]);
    }
    cmd.arg("--format=png")
        .arg(format!("--batch={}", dir.join("scan-%03d.png").display()))
        .arg(format!("--batch-start={first_page}"));
    if !settings.uses_feeder() {
        cmd.arg("--batch-count=1");
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run scanimage ({e}) — install the sane-utils package"))?;
    let stderr = child.stderr.take().expect("piped");
    *handle.lock().unwrap() = Some(child);

    let page_path = |n: u32| dir.join(format!("scan-{n:03}.png"));
    let (mut log, mut pages) = (String::new(), 0u32);
    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
        if let Some(n) = line.strip_prefix("Scanned page ").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<u32>().ok()) {
            // scanimage may still be renaming its temp file; give it a moment.
            let path = page_path(n);
            let start = Instant::now();
            while !path.exists() && start.elapsed() < Duration::from_secs(3) {
                std::thread::sleep(Duration::from_millis(50));
            }
            if path.exists() {
                pages += 1;
                on_page(path);
            }
        }
        log.push_str(&line);
        log.push('\n');
    }
    let status = handle.lock().unwrap().take().map(|mut c| c.wait());
    let killed = matches!(&status, Some(Ok(s)) if s.code().is_none());
    if pages > 0 {
        Ok(pages)
    } else if killed {
        Err("Scan cancelled".into())
    } else {
        Err(last_error(&log))
    }
}

fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("docscanner").join("scanner.conf"))
}

pub fn load_settings() -> Option<ScanSettings> {
    let text = std::fs::read_to_string(config_path()?).ok()?;
    let mut s = ScanSettings::default();
    for line in text.lines() {
        match line.split_once('=') {
            Some(("device", v)) => s.device = v.to_string(),
            Some(("source", v)) => s.source = Some(v.to_string()),
            Some(("mode", v)) => s.mode = Some(v.to_string()),
            Some(("resolution", v)) => s.resolution = v.parse().ok(),
            Some(("paper", v)) => s.paper = Some(v.to_string()),
            _ => {}
        }
    }
    (!s.device.is_empty()).then_some(s)
}

pub fn save_settings(s: &ScanSettings) {
    let Some(path) = config_path() else { return };
    let mut text = format!("device={}\n", s.device);
    if let Some(v) = &s.source {
        text += &format!("source={v}\n");
    }
    if let Some(v) = &s.mode {
        text += &format!("mode={v}\n");
    }
    if let Some(v) = s.resolution {
        text += &format!("resolution={v}\n");
    }
    if let Some(v) = &s.paper {
        text += &format!("paper={v}\n");
    }
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let _ = std::fs::write(path, text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_option_lines() {
        assert_eq!(option_values("    --source Flatbed|ADF [Flatbed]", "--source").unwrap(), ["Flatbed", "ADF"]);
        assert_eq!(
            option_values("    --source Flatbed|Automatic Document Feeder [Flatbed]", "--source").unwrap(),
            ["Flatbed", "Automatic Document Feeder"]
        );
        assert!(option_values("    --brightness 0..100 [inactive]", "--brightness").is_none());
        let r = option_values("    --resolution 200|300|600dpi [200]", "--resolution").unwrap();
        assert_eq!(parse_resolutions(&r), [200, 300, 600]);
        let r = option_values("    --resolution 75..1200dpi (in steps of 1) [75]", "--resolution").unwrap();
        assert_eq!(parse_resolutions(&r), [75, 100, 150, 200, 300, 400, 600, 1200]);
        assert_eq!(range_max("    -y 0..381mm [381]", "-y"), Some(381.0));
        assert_eq!(range_max("    -x 25.4..215.9mm [215.9]", "-x"), Some(215.9));
    }
}
