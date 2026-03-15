"""
models.py
=========
Populationswachstum-Modelle und Hilfs-Funktionen.

Modelle
-------
- Logistisches Wachstum (Verhulst)
    Kontinuierlich : dN/dt = r * N * (1 - N/K)
    Diskret        : N[t+1] = N[t] + r*N[t]*(1 - N[t]/K)

- Allee-Effekt
    Kontinuierlich : dN/dt = r * N * (N/A - 1) * (1 - N/K)
    Diskret        : N[t+1] = N[t] + r*N[t]*(N[t]/A - 1)*(1 - N[t]/K)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from .config import ANIM_STEPS, RED, ORANGE, YELLOW, GREEN, TEAL


# ---------------------------------------------------------------------------
# ODE-Definitionen
# ---------------------------------------------------------------------------


def _logistic_ode(N: float, _t: float, r: float, K: float) -> float:
    """Rechte Seite der logistischen ODE."""
    return r * N * (1.0 - N / K)


def _allee_ode(N: float, _t: float, r: float, K: float, A: float) -> float:
    """Rechte Seite der Allee-Effekt-ODE."""
    return r * N * (N / A - 1.0) * (1.0 - N / K)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate(
    model: str,
    mode: str,
    n0: float,
    t_end: float,
    r: float,
    K: float,
    A: float | None,
    steps: int = ANIM_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simuliert das gewaehlte Modell und gibt (t, N) als numpy-Arrays zurueck.

    Parameters
    ----------
    model : 'Logistisch' | 'Allee-Effekt'
    mode  : 'Kontinuierlich' | 'Diskret'
    n0    : Startpopulation  (>= 0)
    t_end : Simulationsende
    r     : Wachstumsrate
    K     : Kapazitaetsgrenze
    A     : Allee-Schwelle  (nur bei Allee-Effekt, sonst None)
    steps : Anzahl Ausgabe-Zeitpunkte

    Returns
    -------
    t : 1-D Array der Zeitpunkte (Laenge = steps)
    N : 1-D Array der Populationsgroesse (Laenge = steps)

    Raises
    ------
    ValueError
        Bei ungueltigen Parameterkombinationen (z. B. A >= K).
    """
    if K <= 0:
        raise ValueError(f"K muss positiv sein, erhalten: {K}")
    if model == "Allee-Effekt":
        if A is None:
            raise ValueError("A muss angegeben werden fuer den Allee-Effekt.")
        if A <= 0:
            raise ValueError(f"A muss positiv sein, erhalten: {A}")
        if A >= K:
            raise ValueError(f"A ({A}) muss kleiner als K ({K}) sein.")

    n0 = max(float(n0), 0.1)

    if mode == "Kontinuierlich":
        return _simulate_ode(model, n0, t_end, r, K, A, steps)
    elif mode == "Diskret":
        return _simulate_discrete(model, n0, t_end, r, K, A, steps)
    else:
        raise ValueError(f"Unbekannter Modus: {mode!r}")


def _simulate_ode(
    model: str,
    n0: float,
    t_end: float,
    r: float,
    K: float,
    A: float | None,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    t_full = np.linspace(0.0, t_end, max(steps * 5, 1500))

    if model == "Logistisch":
        raw = odeint(_logistic_ode, n0, t_full, args=(r, K))
    else:
        assert A is not None
        raw = odeint(_allee_ode, n0, t_full, args=(r, K, A))

    # ODE-Loeser kann bei N -> 0 winzige negative Werte produzieren -> klemmen
    N_full = np.maximum(0.0, raw.flatten())
    idx = np.linspace(0, len(t_full) - 1, min(steps, len(t_full)), dtype=int)
    return t_full[idx], N_full[idx]


def _simulate_discrete(
    model: str,
    n0: float,
    t_end: float,
    r: float,
    K: float,
    A: float | None,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(t_end)
    t_full = np.arange(n_steps + 1, dtype=float)
    N_full = np.zeros(n_steps + 1)
    N_full[0] = n0

    for i in range(n_steps):
        n = N_full[i]
        if model == "Logistisch":
            delta = r * n * (1.0 - n / K)
        else:
            assert A is not None
            delta = r * n * (n / A - 1.0) * (1.0 - n / K)
        N_full[i + 1] = max(0.0, n + delta)

    idx = np.linspace(0, n_steps, min(steps, n_steps + 1), dtype=int)
    return t_full[idx], N_full[idx]


# ---------------------------------------------------------------------------
# Klassifizierung & Farbe
# ---------------------------------------------------------------------------


def tier_farbe(N: float, K: float, A: float | None) -> str:
    """Gibt eine Hex-Farbe zurueck basierend auf dem Populationsgesundheitszustand."""
    if N < 1.0:
        return RED
    if A is not None and N < A:
        return RED
    ratio = N / K
    if ratio < 0.20:
        return ORANGE
    if ratio < 0.50:
        return YELLOW
    if ratio <= 1.05:
        return GREEN
    return TEAL


def status_msg(
    N: float,
    K: float,
    A: float | None,
    prev_N: float,
) -> tuple[str, str]:
    """
    Gibt eine (Text, Farbe)-Tuple zurueck die den aktuellen Populationsstatus beschreibt.

    Parameters
    ----------
    N      : aktuelle Population
    K      : Kapazitaetsgrenze
    A      : Allee-Schwelle (oder None)
    prev_N : Population im vorherigen Zeitschritt
    """
    from .config import BLUE, GREEN2, ORANGE, YELLOW

    if N < 1.0:
        return "Alle Tiere sind ausgestorben!", RED
    if A is not None and N < A:
        return "Zu wenige Tiere - Aussterben droht! Kritische Zone!", RED

    dN = N - prev_N
    ratio = N / K

    if ratio >= 0.97:
        return "Super! Die Population hat ihre maximale Groesse erreicht!", GREEN2
    if dN > K * 0.025:
        return "Wow! Die Population waechst super schnell!", BLUE
    if dN > 0:
        return "Die Population waechst - alles laeuft gut!", GREEN
    if dN < -K * 0.025:
        return "Achtung! Die Population schrumpft stark!", ORANGE
    if dN < 0:
        return "Die Population nimmt ein bisschen ab.", YELLOW
    return "Die Population ist schoen stabil!", GREEN2
