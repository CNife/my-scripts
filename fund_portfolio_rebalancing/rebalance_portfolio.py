from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from scipy.optimize import minimize
import typer


console = Console()
app = typer.Typer(help="Fund portfolio rebalancing helper.")


@dataclass
class AllocationParams:
    asset_names: List[str]
    current_values: np.ndarray
    target_weights: np.ndarray
    target_total_value: float
    additional_capital: int  # integer total additional capital
    only_buy: bool


@dataclass
class AllocationResult:
    asset_names: List[str]
    current_values: np.ndarray
    additional_allocations: np.ndarray
    new_values: np.ndarray
    target_weights: np.ndarray
    new_weights: np.ndarray
    weight_diff: np.ndarray
    weight_diff_sq: np.ndarray
    euclidean_distance: float
    additional_capital: int
    current_total_value: float
    target_total_value: float
    only_buy: bool


def round_to_int_with_fixed_sum(values: np.ndarray, total_int: int) -> np.ndarray:
    """Round a float vector to integers while preserving the exact integer sum."""
    floor = np.floor(values).astype(int)
    remainder = int(total_int - floor.sum())
    if remainder <= 0:
        return floor

    frac = values - floor
    idx = np.argsort(-frac)
    floor[idx[:remainder]] += 1
    return floor


def optimize_integer_allocation(
    continuous_allocations: np.ndarray,
    total_int: int,
    current_values: np.ndarray,
    target_weights: np.ndarray,
    target_total_value: float,
) -> np.ndarray:
    """Search for a good integer allocation near the continuous optimum."""

    def objective_int(integer_allocations: np.ndarray) -> float:
        new_values = current_values + integer_allocations
        new_weights = new_values / target_total_value
        weight_diff = new_weights - target_weights
        return float(np.sum(weight_diff**2))

    allocations_int = round_to_int_with_fixed_sum(continuous_allocations, total_int)
    best_value = objective_int(allocations_int)

    n = len(allocations_int)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if allocations_int[i] == 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                candidate = allocations_int.copy()
                candidate[i] -= 1
                candidate[j] += 1
                value = objective_int(candidate)
                if value < best_value - 1e-12:
                    allocations_int = candidate
                    best_value = value
                    improved = True
    return allocations_int


def compute_optimal_allocation(params: AllocationParams) -> AllocationResult:
    """Run continuous optimization and integer adjustment for portfolio rebalancing."""
    asset_names = params.asset_names
    current_values = params.current_values
    target_weights = params.target_weights
    target_total_value = params.target_total_value
    additional_capital = params.additional_capital
    only_buy = params.only_buy

    n_assets = len(asset_names)
    if not (len(current_values) == len(target_weights) == n_assets):
        raise ValueError("asset_names, current_values and target_weights must have the same length.")

    current_total_value = float(current_values.sum())

    # Continuous objective
    def objective(allocations: np.ndarray) -> float:
        new_values = current_values + allocations
        new_weights = new_values / target_total_value
        weight_diff = new_weights - target_weights
        return float(np.sum(weight_diff**2))

    # Constraints
    def constraint_sum(allocations: np.ndarray) -> float:
        return float(np.sum(allocations) - additional_capital)

    constraints = [{"type": "eq", "fun": constraint_sum}]
    if only_buy:
        # x_i >= 0  -> min(allocations) >= 0
        constraints.append({"type": "ineq", "fun": lambda allocations: allocations})

    x0 = np.ones(n_assets) * (additional_capital / n_assets if n_assets else 0.0)

    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        constraints=constraints,
        tol=1e-9,
        options={"maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    integer_allocations = optimize_integer_allocation(
        continuous_allocations=result.x,
        total_int=additional_capital,
        current_values=current_values,
        target_weights=target_weights,
        target_total_value=target_total_value,
    )

    new_values = current_values + integer_allocations
    new_weights = new_values / target_total_value
    weight_diff = new_weights - target_weights
    weight_diff_sq = weight_diff**2
    euclidean_distance = float(np.sqrt(np.sum(weight_diff_sq)))

    return AllocationResult(
        asset_names=asset_names,
        current_values=current_values,
        additional_allocations=integer_allocations,
        new_values=new_values,
        target_weights=target_weights,
        new_weights=new_weights,
        weight_diff=weight_diff,
        weight_diff_sq=weight_diff_sq,
        euclidean_distance=euclidean_distance,
        additional_capital=additional_capital,
        current_total_value=current_total_value,
        target_total_value=target_total_value,
        only_buy=only_buy,
    )


def format_currency_2(value: float) -> str:
    return f"{value:,.2f}"


def format_currency_int(value: float | int) -> str:
    return f"{int(value):,d}"


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_percent_signed(value: float) -> str:
    return f"{value * 100:+.2f}%"


def load_portfolio_config(config_path: Path) -> tuple[List[str], np.ndarray]:
    """Load portfolio configuration from JSON file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assets = data.get("assets")
    if not assets or not isinstance(assets, list):
        raise ValueError("Config file must contain an 'assets' list.")

    asset_names: List[str] = []
    target_weights: List[float] = []
    for asset in assets:
        name = asset.get("name")
        weight = asset.get("target_weight")
        if name is None or weight is None:
            raise ValueError("Each asset must have 'name' and 'target_weight'.")
        asset_names.append(str(name))
        target_weights.append(float(weight))

    target_weights_array = np.array(target_weights, dtype=float)
    weight_sum = float(target_weights_array.sum())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"Sum of target_weight must be 1.0, got {weight_sum:.6f}")

    return asset_names, target_weights_array


def prompt_current_values(asset_names: List[str]) -> np.ndarray:
    """Interactively ask user to input current value for each asset."""
    values: List[float] = []
    for name in asset_names:
        while True:
            raw = Prompt.ask(f"请输入标的当前市值（元）: [bold]{name}[/]")
            try:
                value = float(raw)
            except ValueError:
                console.print("[red]输入无效，请输入数字[/red]")
                continue
            if value < 0:
                console.print("[red]市值不能为负，请重新输入[/red]")
                continue
            values.append(value)
            break
    return np.array(values, dtype=float)


def print_allocation_result(result: AllocationResult) -> None:
    """Render allocation result using Rich."""
    console.print("[bold]===== 基础数据 =====[/bold]")
    mode_text = "只买不卖 (only buy)" if result.only_buy else "允许卖出 (allow sell)"
    console.print(f"模式: [cyan]{mode_text}[/cyan]")
    console.print(f"现有总市值: [green]{format_currency_2(result.current_total_value)}[/green] 元")
    console.print(f"追加资金（整数）: [green]{format_currency_int(result.additional_capital)}[/green] 元")
    console.print(f"变动后总市值（整数追加）: [green]{format_currency_2(result.target_total_value)}[/green] 元")

    table = Table(title="资金分配结果")
    table.add_column("资产", justify="left")
    table.add_column("当前市值", justify="right")
    table.add_column("追加金额", justify="right")
    table.add_column("变动后市值", justify="right")
    table.add_column("目标权重", justify="right")
    table.add_column("变动后权重", justify="right")
    table.add_column("权重偏差", justify="right")

    for name, cur, add, new, target_w, new_w, diff in zip(
        result.asset_names,
        result.current_values,
        result.additional_allocations,
        result.new_values,
        result.target_weights,
        result.new_weights,
        result.weight_diff,
    ):
        table.add_row(
            str(name),
            format_currency_2(float(cur)),
            format_currency_int(int(add)),
            format_currency_2(float(new)),
            format_percent(float(target_w)),
            format_percent(float(new_w)),
            format_percent_signed(float(diff)),
        )

    console.print()
    console.print(table)

    console.print()
    console.print("[bold]===== 关键指标 =====[/bold]")
    sum_additional = int(result.additional_allocations.sum())
    console.print(f"追加金额总和: [yellow]{format_currency_int(sum_additional)}[/yellow] 元")
    console.print(f"欧几里得距离: [yellow]{result.euclidean_distance:.6f}[/yellow]")
    console.print(f"比例偏差平方和: [yellow]{float(result.weight_diff_sq.sum()):.6f}[/yellow]")


@app.command()
def main(
    config: Path = typer.Option(
        Path("portfolio.json"),
        "--config",
        "-c",
        help="Path to portfolio configuration JSON file.",
    ),
    target_total: Optional[float] = typer.Option(
        None,
        "--target-total",
        help="Target total portfolio value. Cannot be used together with --additional-cash.",
    ),
    additional_cash: Optional[float] = typer.Option(
        None,
        "--additional-cash",
        help="Total additional cash to invest (can be negative for net sell). "
        "Cannot be used together with --target-total.",
    ),
    current_values: Optional[List[float]] = typer.Option(
        None,
        "--current-values",
        help="Current values of each asset, in the same order as in the config file. "
        "If not provided, you will be prompted to input them interactively.",
    ),
    only_buy: bool = typer.Option(
        False,
        "--only-buy/--allow-sell",
        help="Enable only-buy mode (no negative allocations). Default is allow-sell.",
    ),
) -> None:
    """
    Rebalance a fund portfolio according to target weights.
    """
    if (target_total is None and additional_cash is None) or (target_total is not None and additional_cash is not None):
        console.print("[red]必须且只能提供 --target-total 或 --additional-cash 其中一个参数。[/red]")
        raise typer.Exit(1)

    try:
        asset_names, target_weights = load_portfolio_config(config)
    except (OSError, ValueError) as exc:
        console.print(f"[red]配置文件加载失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    if current_values is not None:
        if len(current_values) != len(asset_names):
            console.print(
                f"[red]--current-values 数量 ({len(current_values)}) "
                f"与配置文件中资产数量 ({len(asset_names)}) 不一致。[/red]"
            )
            raise typer.Exit(1)
        current_values_array = np.array(current_values, dtype=float)
    else:
        current_values_array = prompt_current_values(asset_names)

    current_total_value = float(current_values_array.sum())

    if target_total is not None:
        target_total_value = float(target_total)
        additional_capital_float = target_total_value - current_total_value
    else:
        # 此时 additional_cash 一定非空，类型检查器仍认为可能为 None，这里显式断言
        assert additional_cash is not None
        additional_capital_float = float(additional_cash)
        target_total_value = current_total_value + additional_capital_float

    additional_capital_int = int(np.rint(additional_capital_float))
    target_total_value_int = current_total_value + additional_capital_int

    params = AllocationParams(
        asset_names=asset_names,
        current_values=current_values_array,
        target_weights=target_weights,
        target_total_value=target_total_value_int,
        additional_capital=additional_capital_int,
        only_buy=only_buy,
    )

    try:
        result = compute_optimal_allocation(params)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]优化计算失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    print_allocation_result(result)


if __name__ == "__main__":
    app()
