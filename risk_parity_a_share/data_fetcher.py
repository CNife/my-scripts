"""Data fetcher module for risk parity portfolio optimization.

This module provides data fetching capabilities from Tushare API with caching,
retry mechanisms, and data quality checks for ETF daily price data.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


class DataFetcherError(Exception):
    """Base exception for data fetcher errors."""


class TushareAPIError(DataFetcherError):
    """Exception for Tushare API errors."""


class DataQualityError(DataFetcherError):
    """Exception for data quality issues."""


@dataclass
class DataQualityReport:
    """Data quality check report."""

    missing_dates: list[str] = field(default_factory=list)
    zero_values: dict[str, list[str]] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
    gaps: list[tuple[str, str]] = field(default_factory=list)
    is_valid: bool = True
    message: str = ""


class RiskParityData:
    """Data operations class for risk parity portfolio optimization.

    This class handles ETF data fetching from Tushare API with retry mechanisms,
    caching support, and comprehensive data quality checks.
    """

    def __init__(
        self,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        offline_mode: bool = False,
        rate_limit_delay: float = 0.5,
    ) -> None:
        """Initialize RiskParityData instance.

        Args:
            token: Tushare API token. If None, reads from TUSHARE_TOKEN env var.
            cache_dir: Directory for caching data. If None, uses temp directory.
            offline_mode: If True, only use cached data without API calls.
            rate_limit_delay: Delay between API requests in seconds.
        """
        self.token = token or os.getenv("TUSHARE_TOKEN")
        self.offline_mode = offline_mode
        self.rate_limit_delay = rate_limit_delay
        self._pro = None

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path(os.getenv("TEMP", "/tmp")) / "risk_parity_cache"  # noqa: S108
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate token if not in offline mode
        if not self.offline_mode and not self.token:
            raise DataFetcherError(
                "Tushare token required. Set TUSHARE_TOKEN env var or pass token parameter."
            )

    def _get_pro_api(self):
        """Get or create Tushare Pro API instance."""
        if self._pro is None:
            try:
                import tushare as ts
            except ImportError as e:
                raise DataFetcherError(
                    "tushare package required. Install with: pip install tushare"
                ) from e

            if self.token is None:
                raise DataFetcherError("Tushare token is required for API access")
            self._pro = ts.pro_api(self.token)
        return self._pro

    def _format_etf_code(self, ts_code: str) -> str:
        """Format ETF code with .SH or .SZ suffix.

        Args:
            ts_code: ETF code, may or may not have suffix.

        Returns:
            Formatted ETF code with suffix.
        """
        ts_code = ts_code.strip().upper()

        # Already has suffix
        if ".SH" in ts_code or ".SZ" in ts_code:
            return ts_code

        # Add suffix based on code pattern
        # Shanghai: 50*, 51*, 60*, 68*, 88*, etc.
        # Shenzhen: 00*, 15*, 16*, 30*, 39*, etc.
        if ts_code.startswith(("5", "6", "8")):
            return f"{ts_code}.SH"
        else:
            return f"{ts_code}.SZ"

    def _get_cache_path(self, ts_code: str, start_date: str, end_date: str) -> Path:
        """Get cache file path for given parameters."""
        cache_key = f"{ts_code}_{start_date}_{end_date}.parquet"
        return self.cache_dir / cache_key

    def _load_from_cache(self, cache_path: Path) -> pd.DataFrame | None:
        """Load data from cache if exists."""
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                return None
        return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache."""
        with contextlib.suppress(Exception):
            df.to_parquet(cache_path, index=False)  # Cache failure should not break functionality

    def get_etf_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
        retry_delay: float = 60.0,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch ETF daily data from Tushare API with retry mechanism.

        Args:
            ts_code: ETF code (e.g., '510330' or '510330.SH').
            start_date: Start date in YYYYMMDD format.
            end_date: End date in YYYYMMDD format.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
            use_cache: Whether to use caching.

        Returns:
            DataFrame with ETF daily data.

        Raises:
            DataFetcherError: If data fetching fails after all retries.
            DataQualityError: If data quality checks fail.
        """
        formatted_code = self._format_etf_code(ts_code)
        cache_path = self._get_cache_path(formatted_code, start_date, end_date)

        # Try to load from cache first
        if use_cache:
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df

        if self.offline_mode:
            raise DataFetcherError(
                f"No cached data for {formatted_code} and offline_mode is enabled"
            )

        # Fetch from API with retry
        pro = self._get_pro_api()
        last_error = None

        for attempt in range(max_retries):
            try:
                # Rate limiting
                if attempt > 0:
                    time.sleep(self.rate_limit_delay)

                df = pro.fund_daily(
                    ts_code=formatted_code,
                    start_date=start_date,
                    end_date=end_date,
                )

                if df is None or df.empty:
                    raise TushareAPIError(
                        f"No data returned for {formatted_code} from {start_date} to {end_date}"
                    )

                # Sort by date
                df = df.sort_values("trade_date").reset_index(drop=True)

                # Save to cache
                if use_cache:
                    self._save_to_cache(df, cache_path)

                return df

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

        raise DataFetcherError(
            f"Failed to fetch data for {formatted_code} after {max_retries} attempts: {last_error}"
        )

    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        return_type: str = "log",
    ) -> pd.Series:
        """Calculate returns from price data.

        Args:
            df: DataFrame with price data.
            price_col: Column name for price data.
            return_type: Type of return calculation ('log' or 'simple').

        Returns:
            Series of returns.

        Raises:
            DataFetcherError: If return calculation fails.
        """
        if price_col not in df.columns:
            raise DataFetcherError(f"Price column '{price_col}' not found in data")

        prices = df[price_col].astype(float)

        if return_type == "log":
            returns = pd.Series(np.log(prices / prices.shift(1)), index=prices.index)
        elif return_type == "simple":
            returns = pd.Series(prices.pct_change(), index=prices.index)
        else:
            raise DataFetcherError(f"Invalid return_type: {return_type}. Use 'log' or 'simple'")

        return returns.dropna()

    def calculate_covariance_matrix(
        self,
        returns_dict: dict[str, pd.Series],
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.DataFrame:
        """Calculate covariance matrix from returns.

        Args:
            returns_dict: Dictionary mapping ETF codes to return Series.
            annualize: Whether to annualize the covariance matrix.
            trading_days: Number of trading days per year for annualization.

        Returns:
            Covariance matrix DataFrame.
        """
        if not returns_dict:
            raise DataFetcherError("Empty returns dictionary")

        # Align all returns to common dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if returns_df.empty:
            raise DataFetcherError("No common dates found in returns data")

        cov_matrix = returns_df.cov()

        if annualize:
            cov_matrix = cov_matrix * trading_days

        return cov_matrix

    def check_data_quality(
        self,
        df: pd.DataFrame,
        ts_code: str,
        expected_start: str | None = None,
        expected_end: str | None = None,
    ) -> DataQualityReport:
        """Check data quality for ETF daily data.

        Args:
            df: DataFrame with ETF daily data.
            ts_code: ETF code for reporting.
            expected_start: Expected start date (YYYYMMDD).
            expected_end: Expected end date (YYYYMMDD).

        Returns:
            DataQualityReport with quality check results.
        """
        report = DataQualityReport()

        if df.empty:
            report.is_valid = False
            report.message = "Empty DataFrame"
            return report

        # Check for required columns
        required_cols = ["trade_date", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report.is_valid = False
            report.message = f"Missing required columns: {missing_cols}"
            return report

        # Check for duplicates
        duplicates = df[df.duplicated("trade_date", keep=False)]
        if not duplicates.empty:
            dup_dates = duplicates["trade_date"].tolist()
            report.duplicates = sorted({str(d) for d in dup_dates})
            report.is_valid = False

        # Check for zero values in price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                zero_mask = df[col] == 0
                if zero_mask.any():
                    report.zero_values[col] = df.loc[zero_mask, "trade_date"].tolist()
                    report.is_valid = False

        # Check for missing dates
        if expected_start and expected_end:
            df_dates = set(df["trade_date"].astype(str))
            expected_dates = self._generate_date_range(expected_start, expected_end)
            missing = expected_dates - df_dates
            if missing:
                report.missing_dates = sorted(missing)
                report.is_valid = False

        # Check for gaps in data
        df_sorted = df.sort_values("trade_date")
        dates = pd.to_datetime(df_sorted["trade_date"])
        gaps = []
        for i in range(1, len(dates)):
            gap = (dates.iloc[i] - dates.iloc[i - 1]).days
            if gap > 5:  # More than 5 days gap (excluding weekends)
                gaps.append((str(dates.iloc[i - 1].date()), str(dates.iloc[i].date())))
        report.gaps = gaps

        if not report.message:
            report.message = (
                "Data quality check passed" if report.is_valid else "Data quality issues found"
            )

        return report

    def clean_data(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        remove_zero_prices: bool = True,
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """Clean ETF daily data.

        Args:
            df: DataFrame with ETF daily data.
            remove_duplicates: Whether to remove duplicate dates.
            remove_zero_prices: Whether to remove rows with zero prices.
            fill_missing: Whether to fill missing dates (not implemented).

        Returns:
            Cleaned DataFrame.
        """
        df_clean = df.copy()

        # Remove duplicates (keep first)
        if remove_duplicates and "trade_date" in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=["trade_date"], keep="first")

        # Remove rows with zero prices
        if remove_zero_prices:
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in df_clean.columns:
                    df_clean = df_clean[df_clean[col] != 0]

        # Sort by date
        if "trade_date" in df_clean.columns:
            df_clean = df_clean.sort_values("trade_date", ascending=True).reset_index(drop=True)

        return pd.DataFrame(df_clean)

    def _generate_date_range(self, start_date: str, end_date: str) -> set[str]:
        """Generate set of business dates between start and end.

        Note: This is a simplified version that excludes weekends.
        For accurate trading calendar, use Tushare trade_cal interface.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq="B")  # Business days
        return set(dates.strftime("%Y%m%d"))

    def fetch_multiple_etfs(
        self,
        ts_codes: Sequence[str],
        start_date: str,
        end_date: str,
        price_col: str = "close",
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple ETFs.

        Args:
            ts_codes: List of ETF codes.
            start_date: Start date in YYYYMMDD format.
            end_date: End date in YYYYMMDD format.
            price_col: Price column to extract.
            **kwargs: Additional arguments for get_etf_daily.

        Returns:
            Dictionary mapping ETF codes to DataFrames.
        """
        results = {}
        for code in ts_codes:
            try:
                df = self.get_etf_daily(code, start_date, end_date, **kwargs)
                results[code] = df
            except DataFetcherError as e:
                results[code] = pd.DataFrame()  # Empty DataFrame for failed fetches
                print(f"Warning: Failed to fetch {code}: {e}")
        return results

    def get_price_matrix(
        self,
        ts_codes: Sequence[str],
        start_date: str,
        end_date: str,
        price_col: str = "close",
        **kwargs,
    ) -> pd.DataFrame:
        """Get price matrix for multiple ETFs aligned by date.

        Args:
            ts_codes: List of ETF codes.
            start_date: Start date in YYYYMMDD format.
            end_date: End date in YYYYMMDD format.
            price_col: Price column to extract.
            **kwargs: Additional arguments for get_etf_daily.

        Returns:
            DataFrame with dates as index and ETF codes as columns.
        """
        data_dict = self.fetch_multiple_etfs(ts_codes, start_date, end_date, **kwargs)

        price_data = {}
        for code, df in data_dict.items():
            if not df.empty and price_col in df.columns:
                formatted_code = self._format_etf_code(code)
                price_series = df.set_index("trade_date")[price_col]
                price_series.index = pd.to_datetime(price_series.index)
                price_data[formatted_code] = price_series

        if not price_data:
            raise DataFetcherError("No valid price data retrieved")

        price_matrix = pd.DataFrame(price_data)
        price_matrix = price_matrix.sort_index()

        return price_matrix
