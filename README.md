# Populationswachstum-Simulator

Animierter Simulator fuer Populationswachstum mit grafischer Benutzeroberflaeche.

## Modelle

| Modell | Gleichung (kontinuierlich) |
|---|---|
| Logistisches Wachstum | `dN/dt = r * N * (1 - N/K)` |
| Allee-Effekt | `dN/dt = r * N * (N/A - 1) * (1 - N/K)` |

Beide Modelle sind auch als diskrete Differenzengleichung verfuegbar.

## Projektstruktur

```
population_simulator/
├── src/
│   └── population_simulator/
│       ├── __init__.py      # Paket-Definition und Version
│       ├── config.py        # Farben und Konstanten
│       ├── models.py        # Simulationslogik und Klassifizierung
│       ├── gui.py           # tkinter/matplotlib GUI
│       └── main.py          # Einstiegspunkt
├── tests/
│   ├── __init__.py
│   └── test_models.py       # Unit-Tests fuer models.py
├── scripts/
│   └── build_exe.sh         # Baut eine ausfuehrbare Datei
├── pyproject.toml           # Projektmetadaten und Abhaengigkeiten
├── requirements.txt         # Abhaengigkeiten fuer pip
└── README.md
```

## Installation

### Voraussetzungen

- Python 3.11 oder neuer
- pip

### Abhaengigkeiten installieren

```bash
pip install -r requirements.txt
```

### Starten

```bash
python -m population_simulator
```

oder direkt:

```bash
python src/population_simulator/main.py
```

## Tests ausfuehren

```bash
pip install pytest
pytest tests/ -v
```

Ausgabe-Beispiel:

```
tests/test_models.py::TestSimulateOutputFormat::test_returns_two_arrays[...] PASSED
tests/test_models.py::TestLogisticGrowth::test_converges_to_K_ode         PASSED
tests/test_models.py::TestAlleeEffect::test_above_A_converges_to_K        PASSED
...
```

## Ausfuehrbare Datei erstellen

Das Build-Script verwendet PyInstaller und erstellt eine einzelne `.exe`-Datei
(Windows) bzw. eine ausfuehrbare Binaerdatei (Linux/macOS).

```bash
bash scripts/build_exe.sh
```

Die fertige Datei liegt danach in `dist/population_simulator`.

## Bedienung

| Element | Beschreibung |
|---|---|
| **Normales Wachstum** | Logistisches Modell – Population waechst bis zur Kapazitaetsgrenze K |
| **Bedrohte Tierart** | Allee-Effekt – Startpopulation unter A fuehrt zum Aussterben |
| **r** | Wachstumsgeschwindigkeit |
| **K** | Maximale Anzahl Tiere (Kapazitaetsgrenze) |
| **A** | Allee-Schwelle (nur im Allee-Modell aktiv) |
| **N0** | Startzahl der Tiere |
| **t** | Beobachtungszeit |
| **Start / Pause / Neu** | Animationssteuerung |
| **<<< / >>>** | Animationsgeschwindigkeit |

### Farbcode

| Farbe | Bedeutung |
|---|---|
| Gruen | Population gesund (50-105 % von K) |
| Gelb | Population niedrig (20-50 % von K) |
| Orange | Population sehr niedrig (< 20 % von K) |
| Rot | Aussterben droht oder bereits eingetreten |
| Tuerkis | Population ueberschwingt K leicht |

## Lizenz

MIT
