#!/usr/bin/env bash
# Builds DocScanner-<version>-<arch>.AppImage from target/release. Usage: packaging/appimage.sh <version> <out-dir>
# The AppImage starts the desktop app; `DocScanner.AppImage cli <args>` runs the command-line scanner instead.
set -euo pipefail
version="$1"; out="$2"
arch="$(uname -m)"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
app="$work/DocScanner.AppDir"

install -Dm755 "$root/target/release/docscanner" "$root/target/release/docscanner-gui" -t "$app/usr/bin/"
install -Dm644 "$root/packaging/docscanner.desktop" -t "$app/usr/share/applications/"
install -Dm644 "$root/packaging/docscanner.png" -t "$app/usr/share/icons/hicolor/256x256/apps/"
cp "$root/packaging/docscanner.desktop" "$root/packaging/docscanner.png" "$app/"
ln -s docscanner.png "$app/.DirIcon"
cat > "$app/AppRun" <<'RUN'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
if [ "${1:-}" = "cli" ]; then shift; exec "$here/usr/bin/docscanner" "$@"; fi
exec "$here/usr/bin/docscanner-gui" "$@"
RUN
chmod 755 "$app/AppRun"

tool="$work/appimagetool"
curl -fsSL -o "$tool" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$arch.AppImage"
chmod +x "$tool"
mkdir -p "$out"
ARCH="$arch" APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream "$app" "$out/DocScanner-$version-$arch.AppImage"
