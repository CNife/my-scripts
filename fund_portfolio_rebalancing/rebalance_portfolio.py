import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from scipy.optimize import minimize

console = Console()
app = typer.Typer(help="基金投资组合再平衡助手")


@dataclass
class AllocationParams:
    """投资组合分配参数"""

    asset_names: list[str]
    current_values: np.ndarray
    target_weights: np.ndarray
    target_total_value: float
    additional_capital: int  # 整数追加资金总额
    mode: str  # 交易模式: "all"(允许买卖), "buy"(只买不卖), "sell"(只卖不买)


@dataclass
class AllocationResult:
    """投资组合分配结果"""

    asset_names: list[str]
    current_values: np.ndarray
    additional_allocations: np.ndarray
    new_values: np.ndarray
    target_weights: np.ndarray
    new_weights: np.ndarray
    weight_diff: np.ndarray
    additional_capital: int
    target_total_value: float
    mode: str  # 交易模式: "all"(允许买卖), "buy"(只买不卖), "sell"(只卖不买)
    fund_codes: list[str]  # 基金代码列表
    fund_navs: np.ndarray | None = None  # 基金单位净值
    current_shares: np.ndarray | None = None  # 当前份额
    new_shares: np.ndarray | None = None  # 变动后份额
    share_changes: np.ndarray | None = None  # 份额变动（正数为买入，负数为卖出）

    @property
    def weight_diff_sq(self) -> np.ndarray:
        return self.weight_diff**2

    @property
    def euclidean_distance(self) -> float:
        return float(np.sqrt(np.sum(self.weight_diff_sq)))

    @property
    def current_total_value(self) -> float:
        return float(self.current_values.sum())


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", "-c", help="投资组合配置JSON文件路径")],
    target_total: Annotated[
        float | None,
        typer.Option(
            "--target-total", "-t", help="目标投资组合总市值。不能与 --additional-cash 同时使用"
        ),
    ] = None,
    additional_cash: Annotated[
        float | None,
        typer.Option(
            "--additional-cash",
            "-a",
            help="追加投资的总金额（可为负数表示净卖出）。不能与 --target-total 同时使用",
        ),
    ] = None,
    current_values: Annotated[
        list[float] | None,
        typer.Option(
            "--current-values",
            "-v",
            help="各资产当前市值，顺序需与配置文件一致。如未提供，将交互式询问输入",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="交易模式：all(允许买卖), buy(只买不卖), sell(只卖不卖)"),
    ] = "all",
) -> None:
    """
    根据目标权重对基金投资组合进行再平衡
    """
    if (target_total is None and additional_cash is None) or (
        target_total is not None and additional_cash is not None
    ):
        console.print(
            "[red]必须且只能提供 --target-total 或 --additional-cash 其中一个参数。[/red]"
        )
        raise typer.Exit(1)

    # 验证 mode 参数
    valid_modes = ["all", "buy", "sell"]
    if mode not in valid_modes:
        console.print(
            f"[red]无效的 --mode 参数: {mode}，必须是 {', '.join(valid_modes)} 之一。[/red]"
        )
        raise typer.Exit(1)

    try:
        asset_names, target_weights, fund_codes = load_portfolio_config(config)
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
        # 此时 additional_cash 一定非空，因为前面已经做过互斥检查
        additional_capital_float = float(additional_cash or 0)
        target_total_value = current_total_value + additional_capital_float

    additional_capital_int = int(np.rint(additional_capital_float))
    target_total_value_int = current_total_value + additional_capital_int

    params = AllocationParams(
        asset_names=asset_names,
        current_values=current_values_array,
        target_weights=target_weights,
        target_total_value=target_total_value_int,
        additional_capital=additional_capital_int,
        mode=mode,
    )

    try:
        result = compute_optimal_allocation(params, fund_codes)
    except Exception as exc:
        console.print(f"[red]优化计算失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    print_allocation_result(result)


def load_portfolio_config(config_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """从JSON文件加载投资组合配置"""
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assets = data.get("assets")
    if not assets or not isinstance(assets, list):
        raise ValueError("配置文件必须包含 'assets' 列表")

    try:
        asset_names = [str(asset["name"]) for asset in assets]
        fund_codes = [str(asset["code"]) for asset in assets]
        target_weights = [float(asset["target_weight"]) for asset in assets]
    except KeyError as e:
        raise ValueError(f"资产配置缺少必要字段: {e}") from e

    target_weights_array = np.array(target_weights, dtype=float)
    weight_sum = float(target_weights_array.sum())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"目标权重总和必须为 1.0，实际为 {weight_sum:.6f}")

    return asset_names, target_weights_array, fund_codes


def prompt_current_values(asset_names: list[str]) -> np.ndarray:
    """交互式询问用户输入每个资产的当前市值"""
    values: list[float] = []
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


def compute_optimal_allocation(params: AllocationParams, fund_codes: list[str]) -> AllocationResult:
    """运行连续优化和整数调整，进行投资组合再平衡"""
    asset_names = params.asset_names
    current_values = params.current_values
    target_weights = params.target_weights
    target_total_value = params.target_total_value
    additional_capital = params.additional_capital
    mode = params.mode

    n_assets = len(asset_names)
    if not (len(current_values) == len(target_weights) == n_assets):
        raise ValueError("asset_names、current_values 和 target_weights 的长度必须相同")

    # 连续优化目标函数
    def objective(allocations: np.ndarray) -> float:
        new_values = current_values + allocations
        new_weights = new_values / target_total_value
        weight_diff = new_weights - target_weights
        return float(np.sum(weight_diff**2))

    # 约束条件辅助函数
    def constraint_buy(allocations: np.ndarray) -> float:
        """只买不卖模式约束：所有分配额必须非负"""
        return float(np.min(allocations))

    def constraint_sell(allocations: np.ndarray) -> float:
        """只卖不买模式约束：所有分配额必须非正"""
        return float(-np.max(allocations))

    # 约束条件
    def constraint_sum(allocations: np.ndarray) -> float:
        return float(np.sum(allocations) - additional_capital)

    constraints = [{"type": "eq", "fun": constraint_sum}]
    if mode == "buy":
        constraints.append({"type": "ineq", "fun": constraint_buy})
    elif mode == "sell":
        constraints.append({"type": "ineq", "fun": constraint_sell})

    initial_guess = np.ones(n_assets) * (additional_capital / n_assets if n_assets else 0.0)

    result = minimize(
        fun=objective,
        x0=initial_guess,
        method="SLSQP",
        constraints=constraints,
        tol=1e-9,
        options={"maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"优化失败: {result.message}")

    integer_allocations = optimize_integer_allocation(
        continuous_allocations=result.x,
        total_int=additional_capital,
        current_values=current_values,
        target_weights=target_weights,
        target_total_value=target_total_value,
        mode=mode,
    )

    new_values = current_values + integer_allocations
    new_weights = new_values / target_total_value
    weight_diff = new_weights - target_weights

    # 检查是否需要卖出，仅在需要时获取基金净值并计算份额
    has_sell_operation = bool(np.any(integer_allocations < 0))

    if has_sell_operation:
        # 获取基金净值
        nav_dict = get_fund_latest_nav(fund_codes)
        fund_navs = np.array([nav_dict[code] for code in fund_codes], dtype=float)

        # 计算份额
        current_shares = current_values / fund_navs
        new_shares = new_values / fund_navs
        share_changes = new_shares - current_shares
    else:
        # 不需要卖出，份额相关字段设为 None
        fund_navs = None
        current_shares = None
        new_shares = None
        share_changes = None

    return AllocationResult(
        asset_names=asset_names,
        current_values=current_values,
        additional_allocations=integer_allocations,
        new_values=new_values,
        target_weights=target_weights,
        new_weights=new_weights,
        weight_diff=weight_diff,
        additional_capital=additional_capital,
        target_total_value=target_total_value,
        mode=mode,
        fund_codes=fund_codes,
        fund_navs=fund_navs,
        current_shares=current_shares,
        new_shares=new_shares,
        share_changes=share_changes,
    )


def print_allocation_result(result: AllocationResult) -> None:
    """使用Rich渲染分配结果"""
    console.print("[bold]===== 基础数据 =====[/bold]")
    mode_text_map = {"all": "允许买卖", "buy": "只买不卖", "sell": "只卖不买"}
    console.print(f"模式: [cyan]{mode_text_map.get(result.mode, result.mode)}[/cyan]")
    console.print(f"现有总市值: [green]{format_currency_2(result.current_total_value)}[/green] 元")
    console.print(
        f"追加资金（整数）: [green]{format_currency_int(result.additional_capital)}[/green] 元"
    )
    console.print(
        "变动后总市值（整数追加）: "
        f"[green]{format_currency_2(result.target_total_value)}[/green] 元"
    )

    # 检测是否有卖出操作（基于资金分配）
    has_sell = bool(np.any(result.additional_allocations < 0))

    table = Table(title="资金分配结果")
    table.add_column("资产", justify="left")
    table.add_column("当前市值", justify="right")
    table.add_column("追加金额", justify="right")
    table.add_column("变动后市值", justify="right")
    table.add_column("目标权重", justify="right")
    table.add_column("变动后权重", justify="right")
    table.add_column("权重偏差", justify="right")

    # 仅在需要卖出时添加份额列
    if has_sell:
        table.add_column("份额", justify="right")
        table.add_column("份额变动", justify="right")

    for name, cur, add, new, target_w, new_w, diff in zip(
        result.asset_names,
        result.current_values,
        result.additional_allocations,
        result.new_values,
        result.target_weights,
        result.new_weights,
        result.weight_diff,
        strict=True,
    ):
        row = [
            str(name),
            format_currency_2(float(cur)),
            format_currency_int(int(add)),
            format_currency_2(float(new)),
            format_percent(float(target_w)),
            format_percent(float(new_w)),
            format_percent_signed(float(diff)),
        ]

        # 仅在需要卖出且有份额数据时添加份额信息
        if has_sell and result.share_changes is not None and result.current_shares is not None:
            idx = result.asset_names.index(name)
            row.append(format_shares(float(result.current_shares[idx])))
            row.append(format_shares(float(result.share_changes[idx])))

        table.add_row(*row)

    console.print()
    console.print(table)

    console.print()
    console.print("[bold]===== 关键指标 =====[/bold]")
    sum_additional = int(result.additional_allocations.sum())
    console.print(f"追加金额总和: [yellow]{format_currency_int(sum_additional)}[/yellow] 元")
    console.print(f"欧几里得距离: [yellow]{result.euclidean_distance:.6f}[/yellow]")
    console.print(f"比例偏差平方和: [yellow]{float(result.weight_diff_sq.sum()):.6f}[/yellow]")


def optimize_integer_allocation(
    continuous_allocations: np.ndarray,
    total_int: int,
    current_values: np.ndarray,
    target_weights: np.ndarray,
    target_total_value: float,
    mode: str = "all",
) -> np.ndarray:
    """在连续最优解附近搜索良好的整数分配方案"""

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
                # 检查交易模式约束
                if mode == "buy" and candidate[i] < 0:
                    continue
                if mode == "sell" and candidate[j] > 0:
                    continue
                value = objective_int(candidate)
                if value < best_value - 1e-12:
                    allocations_int = candidate
                    best_value = value
                    improved = True
    return allocations_int


def get_fund_latest_nav(fund_codes: list[str]) -> dict[str, float]:
    """获取基金最新单位净值
    注意：此函数仅在检测到需要卖出时才会被调用
    """
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("需要 tushare 包来获取基金净值。请安装: pip install tushare") from None

    pro = ts.pro_api()
    nav_dict: dict[str, float] = {}

    for code in fund_codes:
        try:
            df = pro.fund_nav(ts_code=code)
            if df is None or df.empty:
                raise ValueError(f"未找到基金 {code} 的净值数据")
            # 获取最新一条记录（已按日期降序排列）
            latest_row = df.iloc[0]
            nav_value = float(latest_row["unit_nav"])
            nav_dict[code] = nav_value
        except Exception as e:
            console.print(f"[red]获取基金 {code} 净值失败: {e}[/red]")
            raise

    return nav_dict


def round_to_int_with_fixed_sum(values: np.ndarray, total_int: int) -> np.ndarray:
    """将浮点数向量四舍五入为整数，同时保持整数总和不变"""
    floor = np.floor(values).astype(int)
    remainder = int(total_int - floor.sum())
    if remainder <= 0:
        return floor

    frac = values - floor
    idx = np.argsort(-frac)
    floor[idx[:remainder]] += 1
    return floor


def format_currency_2(value: float) -> str:
    return f"{value:,.2f}"


def format_currency_int(value: float | int) -> str:
    return f"{int(value):,d}"


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_percent_signed(value: float) -> str:
    return f"{value * 100:+.2f}%"


def format_shares(value: float) -> str:
    """格式化份额显示（保留2位小数）"""
    return f"{value:.2f}"


if __name__ == "__main__":
    app()
