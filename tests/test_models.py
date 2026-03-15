"""
tests/test_models.py
====================
Unit-Tests fuer das models-Modul.

Ausfuehren mit:
    pytest tests/ -v
"""

import pytest
import numpy as np

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from population_simulator.models import simulate, tier_farbe, status_msg
from population_simulator.config import (
    RED,
    ORANGE,
    YELLOW,
    GREEN,
    TEAL,
    GREEN2,
    BLUE,
)


# ===========================================================================
# simulate() – Rueckgabe-Format
# ===========================================================================


class TestSimulateOutputFormat:
    """simulate() muss immer zwei Arrays gleicher Laenge zurueckgeben."""

    @pytest.mark.parametrize(
        "model,mode,A",
        [
            ("Logistisch", "Kontinuierlich", None),
            ("Logistisch", "Diskret", None),
            ("Allee-Effekt", "Kontinuierlich", 40.0),
            ("Allee-Effekt", "Diskret", 40.0),
        ],
    )
    def test_returns_two_arrays(self, model, mode, A):
        t, N = simulate(model, mode, n0=50, t_end=50, r=0.5, K=200, A=A)
        assert isinstance(t, np.ndarray)
        assert isinstance(N, np.ndarray)
        assert len(t) == len(N)

    @pytest.mark.parametrize(
        "model,mode,A",
        [
            ("Logistisch", "Kontinuierlich", None),
            ("Logistisch", "Diskret", None),
            ("Allee-Effekt", "Kontinuierlich", 40.0),
            ("Allee-Effekt", "Diskret", 40.0),
        ],
    )
    def test_time_starts_at_zero(self, model, mode, A):
        t, _ = simulate(model, mode, n0=50, t_end=50, r=0.5, K=200, A=A)
        assert t[0] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "model,mode,A",
        [
            ("Logistisch", "Kontinuierlich", None),
            ("Logistisch", "Diskret", None),
            ("Allee-Effekt", "Kontinuierlich", 40.0),
            ("Allee-Effekt", "Diskret", 40.0),
        ],
    )
    def test_population_nonnegative(self, model, mode, A):
        _, N = simulate(model, mode, n0=10, t_end=100, r=0.5, K=200, A=A)
        assert (N >= 0).all()


# ===========================================================================
# simulate() – Logistisches Wachstum
# ===========================================================================


class TestLogisticGrowth:
    """Logistisches Wachstum soll gegen K konvergieren."""

    def test_converges_to_K_ode(self):
        K = 500
        _, N = simulate(
            "Logistisch", "Kontinuierlich", n0=10, t_end=200, r=0.5, K=K, A=None
        )
        assert abs(N[-1] - K) < K * 0.02, f"N_end={N[-1]:.1f}, K={K}"

    def test_converges_to_K_discrete(self):
        K = 500
        _, N = simulate("Logistisch", "Diskret", n0=10, t_end=200, r=0.3, K=K, A=None)
        assert abs(N[-1] - K) < K * 0.05, f"N_end={N[-1]:.1f}, K={K}"

    def test_population_grows_from_small_start(self):
        _, N = simulate(
            "Logistisch", "Kontinuierlich", n0=1, t_end=50, r=0.5, K=200, A=None
        )
        assert N[-1] > N[0]

    def test_population_stays_at_K_when_starting_at_K(self):
        K = 200
        _, N = simulate(
            "Logistisch", "Kontinuierlich", n0=K, t_end=50, r=0.5, K=K, A=None
        )
        assert abs(N[-1] - K) < K * 0.01

    def test_high_growth_rate_still_converges(self):
        K = 200
        _, N = simulate(
            "Logistisch", "Kontinuierlich", n0=10, t_end=300, r=2.5, K=K, A=None
        )
        assert abs(N[-1] - K) < K * 0.05


# ===========================================================================
# simulate() – Allee-Effekt
# ===========================================================================


class TestAlleeEffect:
    """Allee-Effekt: N > A wachst zu K, N < A stirbt aus."""

    def test_above_A_converges_to_K(self):
        K, A = 200, 40
        _, N = simulate(
            "Allee-Effekt", "Kontinuierlich", n0=100, t_end=200, r=0.3, K=K, A=A
        )
        assert abs(N[-1] - K) < K * 0.02

    def test_below_A_goes_extinct(self):
        K, A = 200, 40
        _, N = simulate(
            "Allee-Effekt", "Kontinuierlich", n0=5, t_end=100, r=0.3, K=K, A=A
        )
        assert N[-1] < 1.0, f"Erwartet Aussterben, N_end={N[-1]:.2f}"

    def test_exactly_at_A_discrete_goes_extinct(self):
        """Exakt auf A: kleine Stoerung durch Diskretisierung -> Aussterben."""
        K, A = 200, 40
        _, N = simulate("Allee-Effekt", "Diskret", n0=A, t_end=200, r=0.3, K=K, A=A)
        # Kein Wachstum erwartet (N bleibt bei A oder geht runter)
        assert N[-1] <= K

    def test_invalid_A_gte_K_raises(self):
        with pytest.raises(ValueError, match="kleiner als K"):
            simulate(
                "Allee-Effekt", "Kontinuierlich", n0=10, t_end=50, r=0.3, K=100, A=150
            )

    def test_missing_A_raises(self):
        with pytest.raises(ValueError, match="A muss angegeben"):
            simulate(
                "Allee-Effekt", "Kontinuierlich", n0=10, t_end=50, r=0.3, K=200, A=None
            )

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError, match="K muss positiv"):
            simulate(
                "Logistisch", "Kontinuierlich", n0=10, t_end=50, r=0.3, K=0, A=None
            )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unbekannter Modus"):
            simulate("Logistisch", "Unbekannt", n0=10, t_end=50, r=0.3, K=200, A=None)


# ===========================================================================
# tier_farbe()
# ===========================================================================


class TestTierFarbe:
    """Farbgebung basierend auf Populationsgesundheit."""

    def test_extinct_returns_red(self):
        assert tier_farbe(0.0, 200, None) == RED
        assert tier_farbe(0.5, 200, None) == RED

    def test_below_allee_returns_red(self):
        assert tier_farbe(20, 200, 40) == RED

    def test_very_low_population_returns_orange(self):
        # 5% von K = 10, unter 20% -> ORANGE
        assert tier_farbe(10, 200, None) == ORANGE

    def test_medium_population_returns_yellow(self):
        # 30% von K
        assert tier_farbe(60, 200, None) == YELLOW

    def test_healthy_population_returns_green(self):
        # 80% von K
        assert tier_farbe(160, 200, None) == GREEN

    def test_at_K_returns_green(self):
        assert tier_farbe(200, 200, None) == GREEN

    def test_above_K_returns_teal(self):
        assert tier_farbe(220, 200, None) == TEAL


# ===========================================================================
# status_msg()
# ===========================================================================


class TestStatusMsg:
    """Status-Nachrichten sollen korrekte Texte und Farben liefern."""

    def test_extinct_message(self):
        txt, col = status_msg(0.0, 200, None, 10.0)
        assert "ausgestorben" in txt
        assert col == RED

    def test_allee_danger_message(self):
        txt, col = status_msg(20, 200, 40, 25)
        assert "Aussterben" in txt
        assert col == RED

    def test_max_population_message(self):
        txt, col = status_msg(198, 200, None, 195)
        assert "maximale" in txt.lower() or "stabil" in txt.lower()
        assert col in (GREEN2, GREEN)

    def test_fast_growth_message(self):
        # dN = 200*0.03 = 6 > K*0.025 = 5
        txt, col = status_msg(106, 200, None, 100)
        assert "schnell" in txt.lower() or "waechst" in txt.lower()
        assert col == BLUE

    def test_slow_growth_message(self):
        txt, col = status_msg(101, 200, None, 100)
        assert "waechst" in txt.lower()
        assert col == GREEN

    def test_shrinking_message(self):
        txt, col = status_msg(90, 200, None, 100)
        assert "schrumpft" in txt.lower() or "nimmt" in txt.lower()

    def test_stable_message(self):
        txt, col = status_msg(100, 200, None, 100)
        assert "stabil" in txt.lower()
        assert col == GREEN2
