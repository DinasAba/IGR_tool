"""
Unit tests for concordance metrics.

Tests cover:
  - top-k intersection: perfect overlap, no overlap, partial, tied values
  - Kendall tau: concordant, discordant, identical, constant/tied vectors
  - Spearman rho: perfect, inverse, identical
  - Cosine similarity: parallel, orthogonal, anti-parallel
  - Pearson: perfect, shifted, constant

Run:
    python -m pytest tests/test_metrics.py -v
"""

import sys
import os
import math

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.concordance import (
    topk_intersection_exact,
    kendall_tau_exact,
    spearman_rho_exact,
    cosine_sim_exact,
    pearson_exact,
)


# ─── Top-k Intersection ────────────────────────────────────────────────


class TestTopKIntersection:
    def test_perfect_overlap(self):
        """Identical vectors -> top-k fully overlaps -> 1.0."""
        v = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.8])
        assert topk_intersection_exact(v, v, k=3) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Disjoint top-k features -> 0.0."""
        v1 = torch.tensor([10.0, 9.0, 0.0, 0.0, 0.0])  # top-2: idx 0,1
        v2 = torch.tensor([0.0, 0.0, 0.0, 9.0, 10.0])   # top-2: idx 3,4
        assert topk_intersection_exact(v1, v2, k=2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Exactly 1 of 2 top features overlaps -> 0.5."""
        v1 = torch.tensor([10.0, 9.0, 1.0, 0.0])  # top-2: {0,1}
        v2 = torch.tensor([10.0, 0.0, 9.0, 0.0])  # top-2: {0,2}
        assert topk_intersection_exact(v1, v2, k=2) == pytest.approx(0.5)

    def test_k_equals_dim(self):
        """When k = dim, all features selected -> always 1.0."""
        v1 = torch.tensor([3.0, 1.0, 2.0])
        v2 = torch.tensor([1.0, 3.0, 2.0])
        assert topk_intersection_exact(v1, v2, k=3) == pytest.approx(1.0)

    def test_k_equals_one(self):
        """k=1: check if single most important feature matches."""
        v1 = torch.tensor([0.1, 0.5, 0.9])  # top-1: idx 2
        v2 = torch.tensor([0.9, 0.5, 0.1])  # top-1: idx 0
        assert topk_intersection_exact(v1, v2, k=1) == pytest.approx(0.0)

    def test_uses_abs(self):
        """Top-k uses abs(attr), so large negative values count."""
        v1 = torch.tensor([-10.0, 1.0, 2.0])  # |v1| top-1: idx 0
        v2 = torch.tensor([10.0, 1.0, 2.0])   # |v2| top-1: idx 0
        assert topk_intersection_exact(v1, v2, k=1) == pytest.approx(1.0)

    def test_tied_values(self):
        """Tied abs values: topk picks arbitrarily; overlap is still valid."""
        v1 = torch.tensor([5.0, 5.0, 5.0, 0.0, 0.0])
        v2 = torch.tensor([5.0, 5.0, 5.0, 0.0, 0.0])
        result = topk_intersection_exact(v1, v2, k=2)
        # Both pick 2 from the same set of 3 tied values -> overlap >= 1/2
        assert 0.5 <= result <= 1.0


# ─── Kendall Tau ────────────────────────────────────────────────────────


class TestKendallTau:
    def test_identical_vectors(self):
        """Identical vectors -> tau = 1.0."""
        v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kendall_tau_exact(v, v, sample_pairs=None)
        assert result == pytest.approx(1.0)

    def test_reversed_vectors(self):
        """Perfectly reversed -> tau = -1.0."""
        v1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        result = kendall_tau_exact(v1, v2, sample_pairs=None)
        assert result == pytest.approx(-1.0)

    def test_concordant_pair(self):
        """Two-element: concordant pair -> tau = 1.0."""
        v1 = torch.tensor([1.0, 2.0])
        v2 = torch.tensor([3.0, 4.0])
        result = kendall_tau_exact(v1, v2, sample_pairs=None)
        assert result == pytest.approx(1.0)

    def test_discordant_pair(self):
        """Two-element: discordant pair -> tau = -1.0."""
        v1 = torch.tensor([2.0, 1.0])
        v2 = torch.tensor([1.0, 2.0])
        result = kendall_tau_exact(v1, v2, sample_pairs=None)
        assert result == pytest.approx(-1.0)

    def test_known_value(self):
        """Known permutation: [1,2,3,4] vs [1,3,2,4] -> tau = 2/3."""
        v1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v2 = torch.tensor([1.0, 3.0, 2.0, 4.0])
        result = kendall_tau_exact(v1, v2, sample_pairs=None)
        # C(6) = {(0,1),(0,2),(0,3),(1,3),(2,3)} = 5 concordant
        # D = {(1,2)} = 1 discordant
        # tau = (5-1)/6 = 4/6 = 2/3
        assert result == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_ties_return_zero_contribution(self):
        """Tied values: sign(0) = 0, so tied pairs contribute 0 to tau.
        Our implementation uses sign-based tau-a: ties contribute 0."""
        v1 = torch.tensor([1.0, 1.0, 3.0])
        v2 = torch.tensor([2.0, 3.0, 1.0])
        result = kendall_tau_exact(v1, v2, sample_pairs=None)
        # Pairs: (0,1): sign(0)*sign(1)=0, (0,2): sign(2)*sign(-1)=-1, (1,2): sign(2)*sign(-2)=-1
        # tau = (0 + -1 + -1)/3 = -2/3
        assert result == pytest.approx(-2.0 / 3.0, abs=1e-6)

    def test_constant_vector(self):
        """All-constant vector: all pairs tied -> tau = 0."""
        v1 = torch.tensor([5.0, 5.0, 5.0, 5.0])
        v2 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = kendall_tau_exact(v1, v2, sample_pairs=None)
        assert result == pytest.approx(0.0)

    def test_subsampled_close_to_exact(self):
        """Subsampled tau should be close to exact for a reasonable number of pairs."""
        torch.manual_seed(42)
        v1 = torch.randn(100)
        v2 = torch.randn(100)
        exact = kendall_tau_exact(v1, v2, sample_pairs=None)
        sampled = kendall_tau_exact(v1, v2, sample_pairs=50000)
        assert sampled == pytest.approx(exact, abs=0.05)

    def test_range(self):
        """Tau should be in [-1, 1]."""
        torch.manual_seed(0)
        for _ in range(10):
            v1 = torch.randn(50)
            v2 = torch.randn(50)
            tau = kendall_tau_exact(v1, v2, sample_pairs=None)
            assert -1.0 <= tau <= 1.0


# ─── Spearman Rho ──────────────────────────────────────────────────────


class TestSpearmanRho:
    def test_identical(self):
        """Identical vectors -> rho = 1.0."""
        v = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0])
        result = spearman_rho_exact(v, v)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_reversed(self):
        """Perfectly reversed -> rho = -1.0."""
        v1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        result = spearman_rho_exact(v1, v2)
        assert result == pytest.approx(-1.0, abs=1e-5)

    def test_known_value(self):
        """Known example: Spearman between monotonic transforms."""
        v1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        result = spearman_rho_exact(v1, v2)
        # Same ranking -> rho = 1.0
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_range(self):
        """Rho should be in [-1, 1]."""
        torch.manual_seed(1)
        for _ in range(10):
            v1 = torch.randn(50)
            v2 = torch.randn(50)
            rho = spearman_rho_exact(v1, v2)
            assert -1.0 - 1e-6 <= rho <= 1.0 + 1e-6


# ─── Cosine Similarity ─────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim_exact(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite(self):
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([-1.0, 0.0, 0.0])
        assert cosine_sim_exact(v1, v2) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal(self):
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        assert cosine_sim_exact(v1, v2) == pytest.approx(0.0, abs=1e-5)

    def test_scaled(self):
        """Cosine is scale-invariant."""
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([100.0, 200.0, 300.0])
        assert cosine_sim_exact(v1, v2) == pytest.approx(1.0, abs=1e-5)


# ─── Pearson ────────────────────────────────────────────────────────────


class TestPearson:
    def test_identical(self):
        v = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert pearson_exact(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_perfect_negative(self):
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([3.0, 2.0, 1.0])
        assert pearson_exact(v1, v2) == pytest.approx(-1.0, abs=1e-5)

    def test_shifted_is_one(self):
        """Pearson is shift-invariant: r(v, v+c) = 1."""
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([101.0, 102.0, 103.0])
        assert pearson_exact(v1, v2) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_vs_pearson_differ(self):
        """Cosine != Pearson when mean != 0 and vectors not centered."""
        v1 = torch.tensor([10.0, 11.0, 12.0])
        v2 = torch.tensor([1.0, 2.0, 3.0])
        cos = cosine_sim_exact(v1, v2)
        pear = pearson_exact(v1, v2)
        # Pearson should be 1.0 (perfect linear), cosine should be less
        assert pear == pytest.approx(1.0, abs=1e-5)
        assert cos < 1.0  # shifted, so cosine is not 1


# ─── Cross-metric consistency ──────────────────────────────────────────


class TestCrossMetricConsistency:
    def test_monotonic_transform_all_high(self):
        """For identical rankings, all rank-based metrics should be high."""
        v1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = v1 ** 2  # same ranking

        tau = kendall_tau_exact(v1, v2, sample_pairs=None)
        rho = spearman_rho_exact(v1, v2)
        cos = cosine_sim_exact(v1, v2)
        pear = pearson_exact(v1, v2)
        topk = topk_intersection_exact(v1, v2, k=3)

        assert tau == pytest.approx(1.0)
        assert rho == pytest.approx(1.0, abs=1e-5)
        assert topk == pytest.approx(1.0)
        assert cos > 0.9
        assert pear > 0.9

    def test_random_vectors_bounded(self):
        """All metrics should be in valid ranges for random data."""
        torch.manual_seed(42)
        v1 = torch.randn(200)
        v2 = torch.randn(200)

        tau = kendall_tau_exact(v1, v2, sample_pairs=None)
        rho = spearman_rho_exact(v1, v2)
        cos = cosine_sim_exact(v1, v2)
        pear = pearson_exact(v1, v2)
        topk = topk_intersection_exact(v1, v2, k=50)

        assert -1.0 <= tau <= 1.0
        assert -1.0 - 1e-6 <= rho <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= cos <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= pear <= 1.0 + 1e-6
        assert 0.0 <= topk <= 1.0
