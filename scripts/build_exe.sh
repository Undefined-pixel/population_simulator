#!/usr/bin/env bash
# =============================================================================
# build_exe.sh
# =============================================================================
# Baut eine ausfuehrbare Einzeldatei mit PyInstaller.
#
# Ausfuehren aus dem Projektverzeichnis:
#   bash scripts/build_exe.sh
#
# Die fertige Datei liegt danach in:
#   dist/population_simulator        (Linux / macOS)
#   dist/population_simulator.exe    (Windows)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
ENTRY="$SRC_DIR/population_simulator/main.py"

echo "============================================"
echo " Populationswachstum-Simulator  Build"
echo "============================================"
echo ""

# --- Abhaengigkeiten pruefen / installieren ----------------------------------
echo "[1/3] Pruefe Abhaengigkeiten..."
pip install --quiet -r "$PROJECT_DIR/requirements.txt"
pip install --quiet pyinstaller
echo "      OK"

# --- Build -------------------------------------------------------------------
echo "[2/3] Baue ausfuehrbare Datei..."
cd "$PROJECT_DIR"

pyinstaller \
    --onefile \
    --windowed \
    --name "population_simulator" \
    --paths "$SRC_DIR" \
    --hidden-import "scipy.integrate" \
    --hidden-import "matplotlib.backends.backend_tkagg" \
    --clean \
    --noconfirm \
    "$ENTRY"

echo "      OK"

# --- Ergebnis ----------------------------------------------------------------
echo "[3/3] Fertig!"
echo ""
echo "  Ausfuehrbare Datei:"
if [[ -f "$PROJECT_DIR/dist/population_simulator.exe" ]]; then
    echo "    dist/population_simulator.exe"
else
    echo "    dist/population_simulator"
fi
echo ""
echo "  Starten:"
echo "    ./dist/population_simulator"
echo "============================================"
