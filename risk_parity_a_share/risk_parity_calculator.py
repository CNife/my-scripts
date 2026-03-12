"""Risk Parity Portfolio Calculator for A-Share ETFs.

This script calculates risk parity weights for a portfolio of ETFs,
where each asset contributes equally to the total portfolio risk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from scipy.optimize import minimize

from risk_parity_a_share.data_fetcher import RiskParityData

if TYPE_CHECKING:
    pass

console = Console()
app = typer.Typer(help="A股ETF风险平价计算器")

# Default ETF configuration for A-share market
DEFAULT_ETF_CONFIG = {
    "etfs": [
        {"name": "沪深300", "code": "510300", "description": "大盘蓝筹"},
        {"name": "中证500", "code": "510500", "description": "中盘成长"},
        {"name": "创业板指", "code": "159915", "description": "小盘成长"},
        {"name": "红利ETF", "code": "510880", "description": "高股息策略"},
        {"name": "黄金ETF", "code": "518880", "description": "黄金避险"},
        {"name": "国债ETF", "code": "511010", "description": "债券避险"},
    ]
}


@dataclass
class RiskParityResult:
    """Risk parity calculation result."""

    etf_names: list[str]
    etf_codes: list[str]
    weights: np.ndarray
    risk_contributions: np.ndarray
    covariance_matrix: np.ndarray
    portfolio_volatility: float
    start_date: str
    end_date: str


@app.command()
def main(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="ETF配置文件路径(JSON格式)"),
    ] = None,
    start_date: Annotated[
        str,
        typer.Option("--start-date", "-s", help="数据起始日期(YYYYMMDD格式)"),
    ] = "20230101",
    end_date: Annotated[
        str,
        typer.Option("--end-date", "-e", help="数据结束日期(YYYYMMDD格式)"),
    ] = "20241231",
    offline: Annotated[
        bool,
        typer.Option("--offline", "-o", help="离线模式，仅使用缓存数据"),
    ] = False,
) -> None:
    """
    计算A股ETF组合的风险平价权重

    风险平价策略使每个资产对组合总风险的贡献相等，
    实现真正的风险分散化配置。
    """
    # Load ETF configuration
    etf_config = load_etf_config(config) if config else DEFAULT_ETF_CONFIG

    etf_names = [etf["name"] for etf in etf_config["etfs"]]
    etf_codes = [etf["code"] for etf in etf_config["etfs"]]

    console.print("[bold]===== 风险平价计算 =====[/bold]")
    console.print(f"数据区间: [cyan]{start_date}[/cyan] 至 [cyan]{end_date}[/cyan]")
    console.print(f"离线模式: [cyan]{'是' if offline else '否'}[/cyan]")
    console.print(f"ETF数量: [cyan]{len(etf_codes)}[/cyan]")

    try:
        # Initialize data fetcher
        data_fetcher = RiskParityData(offline_mode=offline)

        # Fetch price data for all ETFs
        console.print("\n[yellow]正在获取ETF数据...[/yellow]")
        price_matrix = data_fetcher.get_price_matrix(
            ts_codes=etf_codes,
            start_date=start_date,
            end_date=end_date,
        )

        # Calculate returns
        returns_df = price_matrix.pct_change().dropna()

        # Calculate covariance matrix (annualized)
        cov_matrix = returns_df.cov() * 252  # 252 trading days per year

        # Calculate risk parity weights
        console.print("[yellow]正在计算风险平价权重...[/yellow]")
        weights = calculate_risk_parity_weights(cov_matrix.values)

        # Calculate risk contributions
        risk_contributions = calculate_risk_contribution(weights, cov_matrix.values)

        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)

        # Create result
        result = RiskParityResult(
            etf_names=etf_names,
            etf_codes=etf_codes,
            weights=weights,
            risk_contributions=risk_contributions,
            covariance_matrix=cov_matrix.values,
            portfolio_volatility=portfolio_vol,
            start_date=start_date,
            end_date=end_date,
        )

        # Print results
        print_results(result)

    except Exception as e:
        console.print(f"[red]计算失败: {e}[/red]")
        raise typer.Exit(1) from e


def load_etf_config(config_path: Path) -> dict:
    """Load ETF configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file.

    Returns:
        Dictionary with ETF configuration.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If config format is invalid.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "etfs" not in data or not isinstance(data["etfs"], list):
        raise ValueError("配置文件必须包含 'etfs' 列表")

    for etf in data["etfs"]:
        if "name" not in etf or "code" not in etf:
            raise ValueError("每个ETF配置必须包含 'name' 和 'code' 字段")

    return data


def calculate_risk_parity_weights(
    cov_matrix: np.ndarray,
    initial_weights: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> np.ndarray:
    """Calculate risk parity weights using optimization.

    The risk parity objective is to minimize the sum of squared differences
    between each asset's risk contribution and the target (1/n).

    Objective: min Σ(RCᵢ - 1/n)²
    where RCᵢ = (wᵢ * (Σw)ᵢ) / (wᵀΣw) is the risk contribution of asset i

    Args:
        cov_matrix: Covariance matrix of asset returns (n x n).
        initial_weights: Initial guess for weights. If None, uses equal weights.
        tol: Tolerance for optimization convergence.
        max_iter: Maximum number of iterations.

    Returns:
        Optimal risk parity weights (sums to 1).

    Raises:
        ValueError: If covariance matrix is invalid.
        RuntimeError: If optimization fails to converge.
    """
    n = cov_matrix.shape[0]

    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")

    if initial_weights is None:
        initial_weights = np.ones(n) / n

    # Objective function: sum of squared differences from equal risk contribution
    def risk_parity_objective(weights: np.ndarray) -> float:
        """Minimize Σ(RCᵢ - 1/n)²."""
        portfolio_var = weights @ cov_matrix @ weights
        if portfolio_var <= 0:
            return 1e10

        # Marginal risk contribution
        marginal_rc = cov_matrix @ weights

        # Risk contribution of each asset
        rc = weights * marginal_rc / portfolio_var

        # Target: equal risk contribution (1/n for each asset)
        target_rc = 1.0 / n

        # Sum of squared deviations
        return float(np.sum((rc - target_rc) ** 2))

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},  # weights sum to 1
    ]

    # Bounds: 0 <= weight <= 1 (no short selling, no leverage)
    bounds = [(0.0, 1.0) for _ in range(n)]

    # Optimize using SLSQP method
    result = minimize(
        fun=risk_parity_objective,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options={"maxiter": max_iter, "ftol": tol},
    )

    if not result.success:
        raise RuntimeError(f"风险平价优化失败: {result.message}")

    weights = result.x

    # Normalize to ensure exact sum of 1
    weights = weights / np.sum(weights)

    return weights


def calculate_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate risk contribution of each asset.

    Risk contribution RCᵢ = wᵢ * (Σw)ᵢ / (wᵀΣw)
    where:
    - wᵢ is the weight of asset i
    - (Σw)ᵢ is the i-th element of covariance matrix times weights
    - wᵀΣw is the portfolio variance

    Args:
        weights: Portfolio weights (sums to 1).
        cov_matrix: Covariance matrix of asset returns.

    Returns:
        Risk contribution of each asset (sums to 1).
    """
    portfolio_var = weights @ cov_matrix @ weights

    if portfolio_var <= 0:
        raise ValueError("Portfolio variance must be positive")

    # Marginal risk contribution
    marginal_rc = cov_matrix @ weights

    # Risk contribution
    rc = weights * marginal_rc / portfolio_var

    return rc


def print_results(result: RiskParityResult) -> None:
    """Print risk parity calculation results using Rich tables.

    Args:
        result: RiskParityResult containing calculation results.
    """
    console.print("\n[bold]===== 风险平价配置结果 =====[/bold]")

    # Weights table
    weights_table = Table(title="风险平价权重分配")
    weights_table.add_column("ETF名称", justify="left", style="cyan")
    weights_table.add_column("ETF代码", justify="center")
    weights_table.add_column("权重", justify="right", style="green")
    weights_table.add_column("风险贡献", justify="right", style="yellow")
    weights_table.add_column("目标风险贡献", justify="right")
    weights_table.add_column("偏差", justify="right")

    n = len(result.etf_names)
    target_rc = 1.0 / n

    for i, (name, code) in enumerate(zip(result.etf_names, result.etf_codes, strict=True)):
        weight = result.weights[i]
        rc = result.risk_contributions[i]
        deviation = rc - target_rc

        weights_table.add_row(
            name,
            code,
            f"{weight * 100:.2f}%",
            f"{rc * 100:.2f}%",
            f"{target_rc * 100:.2f}%",
            f"{deviation * 100:+.4f}%",
        )

    # Add total row
    weights_table.add_row(
        "[bold]合计[/bold]",
        "-",
        f"[bold]{result.weights.sum() * 100:.2f}%[/bold]",
        f"[bold]{result.risk_contributions.sum() * 100:.2f}%[/bold]",
        f"[bold]{target_rc * n * 100:.2f}%[/bold]",
        "-",
    )

    console.print(weights_table)

    # Portfolio metrics
    console.print("\n[bold]===== 组合风险指标 =====[/bold]")
    console.print(f"组合年化波动率: [yellow]{result.portfolio_volatility * 100:.2f}%[/yellow]")
    console.print(
        f"风险贡献均衡度: [yellow]{calculate_balance_score(result.risk_contributions):.6f}[/yellow]"
    )

    # Risk contribution chart (text-based)
    console.print("\n[bold]===== 风险贡献可视化 =====[/bold]")
    max_bar_width = 40
    for name, rc in zip(result.etf_names, result.risk_contributions, strict=True):
        bar_width = int(rc * max_bar_width * n)  # Scale so target is max_bar_width / n
        bar = "█" * bar_width
        console.print(f"{name:8s} |{bar:<{max_bar_width}s}| {rc * 100:5.2f}%")


def calculate_balance_score(risk_contributions: np.ndarray) -> float:
    """Calculate risk contribution balance score.

    Lower score means better balance (closer to equal risk contribution).
    Score of 0 means perfect risk parity.

    Args:
        risk_contributions: Risk contribution of each asset.

    Returns:
        Balance score (sum of squared deviations from equal).
    """
    n = len(risk_contributions)
    target = 1.0 / n
    return float(np.sum((risk_contributions - target) ** 2))


def format_percent(value: float) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    app()
