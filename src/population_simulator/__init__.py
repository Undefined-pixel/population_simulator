"""
population_simulator
====================
Animierter Simulator fuer Populationswachstum mit GUI.

Module
------
config  - Farben und Konstanten
models  - Simulationsmodelle und Klassifizierungsfunktionen
gui     - tkinter/matplotlib GUI
main    - Einstiegspunkt
"""

__version__ = "1.0.0"
__author__ = "Population Simulator"

from .main import run

__all__ = ["run"]
