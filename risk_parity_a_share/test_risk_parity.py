"""Tests for risk parity portfolio calculator.

This module contains unit tests for the risk parity weight calculation
using synthetic covariance matrices to avoid external API dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk_parity_a_share.data_fetcher import (
    DataFetcherError,
    DataQualityError,
    DataQualityReport,
    RiskParityData,
)
from risk_parity_a_share.risk_parity_calculator import (
    RiskParityResult,
    calculate_balance_score,
    calculate_risk_contribution,
    calculate_risk_parity_weights,
)


class TestCalculateRiskParityWeights:
    """Tests for calculate_risk_parity_weights function."""

    def test_weights_sum_to_one(self):
        """Test that calculated weights sum to 1."""
        # Create a simple 3-asset covariance matrix
        cov_matrix = np.array(
            [
                [0.04, 0.02, 0.01],
                [0.02, 0.09, 0.03],
                [0.01, 0.03, 0.16],
            ]
        )

        weights = calculate_risk_parity_weights(cov_matrix)

        assert np.isclose(np.sum(weights), 1.0, atol=1e-10)
        assert len(weights) == 3
        assert all(w >= 0 for w in weights)  # No short selling

    def test_weights_sum_to_one_different_sizes(self):
        """Test weights sum to 1 for different portfolio sizes."""
        for n in [2, 3, 5, 10]:
            # Create a diagonal covariance matrix (uncorrelated assets)
            cov_matrix = np.diag(np.random.uniform(0.01, 0.25, n))

            weights = calculate_risk_parity_weights(cov_matrix)

            assert np.isclose(np.sum(weights), 1.0, atol=1e-10), f"Failed for n={n}"
            assert len(weights) == n

    def test_equal_variance_equal_weights(self):
        """Test that equal variance uncorrelated assets get equal weights."""
        n = 4
        cov_matrix = np.eye(n) * 0.04  # Equal variance, no correlation

        weights = calculate_risk_parity_weights(cov_matrix)

        expected_weight = 1.0 / n
        for w in weights:
            assert np.isclose(w, expected_weight, atol=1e-6)

    def test_higher_variance_gets_lower_weight(self):
        """Test that higher variance assets get lower weights."""
        # Asset 1: low variance, Asset 2: high variance
        cov_matrix = np.array(
            [
                [0.01, 0.0],
                [0.0, 0.25],
            ]
        )

        weights = calculate_risk_parity_weights(cov_matrix)

        # Low variance asset should have higher weight
        assert weights[0] > weights[1]

    def test_invalid_covariance_matrix(self):
        """Test that non-square matrix raises ValueError."""
        cov_matrix = np.array(
            [
                [0.04, 0.02],
                [0.02, 0.09],
                [0.01, 0.03],
            ]
        )

        with pytest.raises(ValueError, match="square"):
            calculate_risk_parity_weights(cov_matrix)


class TestRiskContributionEquality:
    """Tests for risk contribution equality in risk parity portfolios."""

    def test_risk_contributions_sum_to_one(self):
        """Test that risk contributions sum to 1."""
        cov_matrix = np.array(
            [
                [0.04, 0.02, 0.01],
                [0.02, 0.09, 0.03],
                [0.01, 0.03, 0.16],
            ]
        )

        weights = calculate_risk_parity_weights(cov_matrix)
        risk_contributions = calculate_risk_contribution(weights, cov_matrix)

        assert np.isclose(np.sum(risk_contributions), 1.0, atol=1e-10)

    def test_risk_contributions_approximately_equal(self):
        """Test that risk contributions are approximately equal (1/n)."""
        n = 4
        # Create a covariance matrix with varying variances and correlations
        cov_matrix = np.array(
            [
                [0.04, 0.015, 0.01, 0.005],
                [0.015, 0.09, 0.02, 0.01],
                [0.01, 0.02, 0.16, 0.015],
                [0.005, 0.01, 0.015, 0.25],
            ]
        )

        weights = calculate_risk_parity_weights(cov_matrix)
        risk_contributions = calculate_risk_contribution(weights, cov_matrix)

        target_rc = 1.0 / n
        for rc in risk_contributions:
            assert np.isclose(rc, target_rc, atol=0.01)  # Within 1% tolerance

    def test_risk_contributions_approximately_equal_uncorrelated(self):
        """Test risk contribution equality for uncorrelated assets."""
        n = 5
        # Diagonal covariance matrix (uncorrelated)
        variances = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
        cov_matrix = np.diag(variances)

        weights = calculate_risk_parity_weights(cov_matrix)
        risk_contributions = calculate_risk_contribution(weights, cov_matrix)

        target_rc = 1.0 / n
        for rc in risk_contributions:
            assert np.isclose(rc, target_rc, atol=0.001)

    def test_risk_contribution_with_correlation(self):
        """Test risk contribution with correlated assets."""
        # Two highly correlated assets
        cov_matrix = np.array(
            [
                [0.04, 0.035],
                [0.035, 0.04],
            ]
        )

        weights = calculate_risk_parity_weights(cov_matrix)
        risk_contributions = calculate_risk_contribution(weights, cov_matrix)

        # With high correlation and equal variance, weights should be equal
        assert np.isclose(weights[0], weights[1], atol=0.01)
        assert np.isclose(risk_contributions[0], risk_contributions[1], atol=0.01)


class TestDataFetcherImport:
    """Tests for data fetcher module imports and basic functionality."""

    def test_risk_parity_data_import(self):
        """Test that RiskParityData class can be imported."""
        assert RiskParityData is not None

    def test_exceptions_import(self):
        """Test that custom exceptions can be imported."""
        assert DataFetcherError is not None
        assert DataQualityError is not None

    def test_data_quality_report_import(self):
        """Test that DataQualityReport can be imported."""
        report = DataQualityReport()
        assert report is not None
        assert report.is_valid is True

    def test_risk_parity_data_offline_mode(self):
        """Test RiskParityData initialization in offline mode."""
        # Should work without token in offline mode
        data_fetcher = RiskParityData(offline_mode=True)
        assert data_fetcher.offline_mode is True
        assert data_fetcher.token is None or data_fetcher.token == ""

    def test_risk_parity_data_requires_token_online(self):
        """Test that RiskParityData requires token in online mode."""
        import os

        # Temporarily remove token from environment
        original_token = os.environ.pop("TUSHARE_TOKEN", None)
        try:
            with pytest.raises(DataFetcherError, match="token"):
                RiskParityData(offline_mode=False, token=None)
        finally:
            # Restore token if it existed
            if original_token:
                os.environ["TUSHARE_TOKEN"] = original_token


class TestRiskParityResult:
    """Tests for RiskParityResult dataclass."""

    def test_result_creation(self):
        """Test that RiskParityResult can be created."""
        result = RiskParityResult(
            etf_names=["ETF1", "ETF2"],
            etf_codes=["510300", "510500"],
            weights=np.array([0.5, 0.5]),
            risk_contributions=np.array([0.5, 0.5]),
            covariance_matrix=np.array([[0.04, 0.02], [0.02, 0.09]]),
            portfolio_volatility=0.15,
            start_date="20230101",
            end_date="20231231",
        )

        assert len(result.etf_names) == 2
        assert len(result.weights) == 2
        assert result.portfolio_volatility > 0


class TestCalculateBalanceScore:
    """Tests for calculate_balance_score function."""

    def test_perfect_balance_zero_score(self):
        """Test that perfect risk parity gives zero balance score."""
        n = 4
        risk_contributions = np.ones(n) / n

        score = calculate_balance_score(risk_contributions)

        assert np.isclose(score, 0.0, atol=1e-10)

    def test_imbalanced_higher_score(self):
        """Test that imbalanced contributions give higher score."""
        balanced = np.array([0.25, 0.25, 0.25, 0.25])
        imbalanced = np.array([0.4, 0.3, 0.2, 0.1])

        balanced_score = calculate_balance_score(balanced)
        imbalanced_score = calculate_balance_score(imbalanced)

        assert balanced_score < imbalanced_score


class TestSyntheticCovarianceMatrices:
    """Tests using various synthetic covariance matrices."""

    def test_two_asset_case(self):
        """Test simple two-asset portfolio."""
        cov_matrix = np.array(
            [
                [0.04, 0.01],
                [0.01, 0.09],
            ]
        )

        weights = calculate_risk_parity_weights(cov_matrix)
        risk_contributions = calculate_risk_contribution(weights, cov_matrix)

        assert np.isclose(np.sum(weights), 1.0)
        assert np.isclose(np.sum(risk_contributions), 1.0)
        assert np.isclose(risk_contributions[0], risk_contributions[1], atol=0.01)

    def test_five_asset_diverse_volatilities(self):
        """Test five assets with diverse volatilities."""
        # Create covariance matrix with different volatilities
        vols = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
        corr = 0.3
        n = len(vols)

        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = vols[i] ** 2
                else:
                    cov_matrix[i, j] = vols[i] * vols[j] * corr

        weights = calculate_risk_parity_weights(cov_matrix)
        risk_contributions = calculate_risk_contribution(weights, cov_matrix)

        # Verify properties
        assert np.isclose(np.sum(weights), 1.0)
        assert np.isclose(np.sum(risk_contributions), 1.0)

        # Risk contributions should be approximately equal
        target_rc = 1.0 / n
        for rc in risk_contributions:
            assert np.isclose(rc, target_rc, atol=0.02)

    def test_ten_asset_portfolio(self):
        """Test larger ten-asset portfolio."""
        n = 10
        # Random positive definite covariance matrix
        np.random.seed(42)
        A = np.random.randn(n, n)
        cov_matrix = np.dot(A, A.T) * 0.01  # Scale down

        weights = calculate_risk_parity_weights(cov_matrix)
        risk_contributions = calculate_risk_contribution(weights, cov_matrix)

        assert np.isclose(np.sum(weights), 1.0)
        assert np.isclose(np.sum(risk_contributions), 1.0)
        assert len(weights) == n
        assert all(w >= 0 for w in weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
