"""
main.py
=======
Einstiegspunkt der Anwendung.
"""

from .gui import PopulationSimulatorApp


def run() -> None:
    """Startet den Populationswachstum-Simulator."""
    app = PopulationSimulatorApp()
    app.mainloop()


if __name__ == "__main__":
    run()
