# /// script
# dependencies = [
#     "tushare>=1.4.0",
#     "pandas>=2.0",
#     "rich>=13.0",
#     "typer>=0.24.1",
# ]
# ///

"""
A股烟蒂股筛选器

基于"扣除净现金后的十倍估值法"筛选A股烟蒂股
"""

import os
import time
from datetime import datetime
from functools import wraps

import pandas as pd
import tushare as ts
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

CSV_COLUMNS = [
    "ts_code",
    "name",
    "level",
    "close",
    "total_mv",
    "pe_ttm",
    "pb",
    "dv_ttm",
    "net_cash",
    "net_cash_ratio",
    "adj_pe",
    "industry",
    "risk_warning",
]

EXCLUDED_INDUSTRIES = ["银行", "保险", "证券", "全国地产", "区域地产", "房产服务"]
TARGET_MARKETS = ["主板", "创业板", "科创板"]
MAX_RETRIES = 3
RETRY_DELAY = 2


def get_tushare_pro():
    return ts.pro_api()


def with_retry(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        console.print(
                            f"[yellow]请求失败，{delay}秒后重试 ({attempt + 1}/{max_retries}): {e}[/yellow]"
                        )
                        time.sleep(delay)
                    else:
                        console.print(f"[red]请求失败，已达到最大重试次数: {e}[/red]")
                        raise
            return None

        return wrapper

    return decorator


@with_retry()
def fetch_daily_basic(pro, ts_code: str, trade_date: str | None = None) -> pd.DataFrame:
    """获取股票的每日基本面指标（PE、PB、市值、股息率等）"""
    params = {"ts_code": ts_code}
    if trade_date:
        params["trade_date"] = trade_date
    return pro.daily_basic(**params)


@with_retry()
def fetch_balancesheet(pro, ts_code: str) -> pd.DataFrame:
    """获取股票的资产负债表数据"""
    return pro.balancesheet(ts_code=ts_code, limit=1)


def fetch_financial_data(stock_df: pd.DataFrame, pro, batch_size: int = 50) -> pd.DataFrame:
    """批量获取股票的财务数据

    Args:
        stock_df: 股票列表DataFrame
        pro: tushare pro api实例
        batch_size: 每批处理的股票数量

    Returns:
        包含财务数据的DataFrame
    """
    results = []
    total = len(stock_df)

    console.print(f"\n[bold]开始获取 {total} 只股票的财务数据...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("获取财务数据...", total=total)

        for _idx, row in stock_df.iterrows():
            ts_code = row["ts_code"]
            stock_name = row["name"]

            try:
                daily_data = fetch_daily_basic(pro, ts_code)
                balance_data = fetch_balancesheet(pro, ts_code)

                result = {
                    "ts_code": ts_code,
                    "name": stock_name,
                    "market": row.get("market", ""),
                    "industry": row.get("industry", ""),
                }

                if daily_data is not None and not daily_data.empty:
                    latest = daily_data.iloc[0]
                    result.update(
                        {
                            "pe": latest.get("pe"),
                            "pb": latest.get("pb"),
                            "total_mv": latest.get("total_mv"),
                            "circ_mv": latest.get("circ_mv"),
                            "dv_ratio": latest.get("dv_ratio"),
                            "dv_ttm": latest.get("dv_ttm"),
                            "turnover_rate": latest.get("turnover_rate"),
                            "volume_ratio": latest.get("volume_ratio"),
                            "trade_date": latest.get("trade_date"),
                        }
                    )

                if balance_data is not None and not balance_data.empty:
                    latest = balance_data.iloc[0]
                    result.update(
                        {
                            "money_cap": latest.get("money_cap"),
                            "total_liab": latest.get("total_liab"),
                            "total_assets": latest.get("total_assets"),
                            "total_cur_assets": latest.get("total_cur_assets"),
                            "total_cur_liab": latest.get("total_cur_liab"),
                            "total_ncl": latest.get("total_ncl"),
                            "end_date": latest.get("end_date"),
                        }
                    )

                results.append(result)

            except Exception as e:
                console.print(f"[yellow]获取 {ts_code} ({stock_name}) 数据失败: {e}[/yellow]")

            progress.update(task, advance=1)

            if (len(results) % batch_size) == 0:
                time.sleep(0.5)

    result_df = pd.DataFrame(results)
    console.print(f"[green]✓[/green] 成功获取 {len(result_df)} 只股票的财务数据")

    return result_df


def calculate_net_cash_t0(row: pd.Series) -> float:
    """计算T0级净现金：货币资金 - 总负债

    Args:
        row: 包含财务数据的Series

    Returns:
        T0级净现金
    """
    money_cap = row.get("money_cap", 0) or 0
    total_liab = row.get("total_liab", 0) or 0
    return money_cap - total_liab


def calculate_net_cash_t1(row: pd.Series) -> float:
    """计算T1级净现金：货币资金 + 交易性金融资产 - 有息负债

    有息负债 = 总负债 - 无息负债(应付账款等)
    简化计算：使用总负债作为有息负债的近似

    Args:
        row: 包含财务数据的Series

    Returns:
        T1级净现金
    """
    money_cap = row.get("money_cap", 0) or 0
    trad_asset = row.get("trad_asset", 0) or 0
    total_liab = row.get("total_liab", 0) or 0
    return money_cap + trad_asset - total_liab


def calculate_net_working_capital(row: pd.Series) -> float:
    """计算T2级净营运资本：流动资产 - 总负债

    Args:
        row: 包含财务数据的Series

    Returns:
        净营运资本
    """
    total_cur_assets = row.get("total_cur_assets", 0) or 0
    total_liab = row.get("total_liab", 0) or 0
    return total_cur_assets - total_liab


def calculate_pe_after_cash(row: pd.Series, net_cash: float) -> float | None:
    """计算扣除净现金后的PE

    Args:
        row: 包含财务数据的Series
        net_cash: 净现金金额

    Returns:
        扣除净现金后的PE，如果无法计算则返回None
    """
    pe = row.get("pe")
    total_mv = row.get("total_mv", 0) or 0

    if pd.isna(pe) or total_mv <= 0:
        return None

    # 扣除净现金后的市值 = 总市值 - 净现金
    # 注意：净现金可能为负（净负债）
    mv_after_cash = total_mv - net_cash

    if mv_after_cash <= 0:
        return None

    # 扣除净现金后的PE = 原PE * (扣除净现金后的市值 / 总市值)
    return pe * (mv_after_cash / total_mv)


def screen_t0_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """T0级筛选：净现金/市值 > 1

    净现金 = 货币资金 - 总负债

    Args:
        df: 包含财务数据的DataFrame

    Returns:
        通过T0级筛选的股票DataFrame
    """
    df = df.copy()

    # 计算T0级净现金
    df["net_cash_t0"] = df.apply(calculate_net_cash_t0, axis=1)

    # 计算净现金/市值比率
    df["net_cash_t0_ratio"] = df["net_cash_t0"] / df["total_mv"]

    # 筛选条件：净现金/市值 > 1
    mask = (df["net_cash_t0_ratio"] > 1) & (df["total_mv"] > 0)

    return df[mask].copy()


def screen_t1_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """T1级筛选：净现金/市值 > 0.5

    净现金 = 货币资金 + 交易性金融资产 - 有息负债

    Args:
        df: 包含财务数据的DataFrame

    Returns:
        通过T1级筛选的股票DataFrame
    """
    df = df.copy()

    # 计算T1级净现金
    df["net_cash_t1"] = df.apply(calculate_net_cash_t1, axis=1)

    # 计算净现金/市值比率
    df["net_cash_t1_ratio"] = df["net_cash_t1"] / df["total_mv"]

    # 筛选条件：净现金/市值 > 0.5
    mask = (df["net_cash_t1_ratio"] > 0.5) & (df["total_mv"] > 0)

    return df[mask].copy()


def screen_t2_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """T2级筛选：净营运资本/市值 > 0.5

    净营运资本 = 流动资产 - 总负债

    Args:
        df: 包含财务数据的DataFrame

    Returns:
        通过T2级筛选的股票DataFrame
    """
    df = df.copy()

    # 计算净营运资本
    df["net_working_capital"] = df.apply(calculate_net_working_capital, axis=1)

    # 计算净营运资本/市值比率
    df["net_working_capital_ratio"] = df["net_working_capital"] / df["total_mv"]

    # 筛选条件：净营运资本/市值 > 0.5
    mask = (df["net_working_capital_ratio"] > 0.5) & (df["total_mv"] > 0)

    return df[mask].copy()


def screen_by_dividend_and_pe(
    df: pd.DataFrame, min_dv: float = 3.0, max_pe: float = 15.0
) -> pd.DataFrame:
    """通用筛选：股息率 >= 3% 且 扣净现金后 PE < 15

    Args:
        df: 包含财务数据的DataFrame
        min_dv: 最低股息率要求(%)，默认3%
        max_pe: 最高PE要求，默认15

    Returns:
        通过通用筛选的股票DataFrame
    """
    df = df.copy()

    # 计算T0级净现金用于PE调整
    df["net_cash_t0"] = df.apply(calculate_net_cash_t0, axis=1)

    # 计算扣除净现金后的PE
    df["pe_after_cash"] = df.apply(
        lambda row: calculate_pe_after_cash(row, row["net_cash_t0"]), axis=1
    )

    # 筛选条件：股息率 >= min_dv% 且 扣净现金后 PE < max_pe
    mask = (df["dv_ttm"] >= min_dv) & (df["pe_after_cash"] < max_pe) & (df["pe_after_cash"].notna())

    return df[mask].copy()


def add_risk_warnings(df: pd.DataFrame) -> pd.DataFrame:
    """添加风险提示标注

    风险提示包括：
    - 高负债风险：总负债/总资产 > 0.7
    - 流动性风险：流动资产/流动负债 < 1.0
    - 低股息风险：股息率 < 1%
    - 高PE风险：PE > 50
    - 负现金流风险：货币资金 < 0

    Args:
        df: 包含财务数据的DataFrame

    Returns:
        添加了风险提示的DataFrame
    """
    df = df.copy()
    warnings = []

    for _, row in df.iterrows():
        stock_warnings = []

        # 高负债风险
        total_liab = row.get("total_liab", 0) or 0
        total_assets = row.get("total_assets", 0) or 0
        if total_assets > 0 and total_liab / total_assets > 0.7:
            stock_warnings.append("高负债")

        # 流动性风险
        total_cur_assets = row.get("total_cur_assets", 0) or 0
        total_cur_liab = row.get("total_cur_liab", 0) or 0
        if total_cur_liab > 0 and total_cur_assets / total_cur_liab < 1.0:
            stock_warnings.append("流动性不足")

        # 低股息风险
        dv_ttm = row.get("dv_ttm", 0) or 0
        if dv_ttm < 1.0:
            stock_warnings.append("低股息")

        # 高PE风险
        pe = row.get("pe")
        if pd.notna(pe) and pe > 50:
            stock_warnings.append("高估值")

        # 负现金流风险
        money_cap = row.get("money_cap", 0) or 0
        if money_cap < 0:
            stock_warnings.append("负现金")

        warnings.append(", ".join(stock_warnings) if stock_warnings else "")

    df["risk_warnings"] = warnings
    return df


def display_screening_results(df: pd.DataFrame, level: str):
    """显示筛选结果

    Args:
        df: 筛选后的DataFrame
        level: 筛选级别(T0/T1/T2/通用)
    """
    console.print(f"\n[bold green]{level}级筛选结果[/bold green]")
    console.print(f"通过筛选的股票数量: {len(df)}")

    if len(df) == 0:
        console.print("[yellow]没有股票通过该级别筛选[/yellow]")
        return

    table = Table(title=f"{level}级筛选股票列表")
    table.add_column("代码", style="cyan")
    table.add_column("名称", style="magenta")
    table.add_column("行业", style="green")
    table.add_column("市值(亿)", style="blue")
    table.add_column("股息率(%)", style="yellow")
    table.add_column("PE", style="cyan")

    if "risk_warnings" in df.columns:
        table.add_column("风险提示", style="red")

    for _, row in df.iterrows():
        rows = [
            row["ts_code"],
            row["name"],
            row.get("industry", "N/A"),
            f"{row.get('total_mv', 0):.2f}",
            f"{row.get('dv_ttm', 0):.2f}",
            f"{row.get('pe', 0):.2f}",
        ]

        if "risk_warnings" in df.columns:
            rows.append(row.get("risk_warnings", ""))

        table.add_row(*rows)

    console.print(table)


def export_to_csv(t0_df: pd.DataFrame, t1_df: pd.DataFrame, t2_df: pd.DataFrame):
    """导出筛选结果到CSV文件

    Args:
        t0_df: T0级筛选结果
        t1_df: T1级筛选结果
        t2_df: T2级筛选结果
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"cigar_butt_stocks_{date_str}.csv")

    all_results = []

    for level, df in [("T0", t0_df), ("T1", t1_df), ("T2", t2_df)]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            result = {
                "ts_code": row.get("ts_code", ""),
                "name": row.get("name", ""),
                "level": level,
                "close": row.get("close", ""),
                "total_mv": row.get("total_mv", 0),
                "pe_ttm": row.get("pe", ""),
                "pb": row.get("pb", ""),
                "dv_ttm": row.get("dv_ttm", 0),
                "net_cash": row.get("net_cash", 0),
                "net_cash_ratio": row.get("net_cash_ratio", 0),
                "adj_pe": row.get("adj_pe", ""),
                "industry": row.get("industry", ""),
                "risk_warning": row.get("risk_warnings", ""),
            }
            all_results.append(result)

    if not all_results:
        console.print("[yellow]没有筛选结果可导出[/yellow]")
        return

    result_df = pd.DataFrame(all_results)

    level_order = {"T0": 0, "T1": 1, "T2": 2}
    result_df["level_sort"] = result_df["level"].map(lambda x: level_order.get(x, 99))
    result_df = result_df.sort_values(by=["level_sort", "ts_code"]).drop(columns=["level_sort"])

    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    console.print(f"[green]✓[/green] CSV文件已保存: {output_file}")
    console.print(f"   共导出 {len(result_df)} 条记录")


def display_financial_summary(df: pd.DataFrame):
    """显示财务数据摘要"""
    console.print("\n[bold green]财务数据摘要[/bold green]")

    table = Table(title="估值指标统计")
    table.add_column("指标", style="cyan")
    table.add_column("平均值", style="magenta")
    table.add_column("中位数", style="green")
    table.add_column("最小值", style="blue")
    table.add_column("最大值", style="red")

    metrics = [
        ("PE", "pe"),
        ("PB", "pb"),
        ("总市值(亿)", "total_mv"),
        ("股息率(%)", "dv_ratio"),
    ]

    for name, col in metrics:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                table.add_row(
                    name,
                    f"{valid_data.mean():.2f}",
                    f"{valid_data.median():.2f}",
                    f"{valid_data.min():.2f}",
                    f"{valid_data.max():.2f}",
                )

    console.print(table)

    if "money_cap" in df.columns and "total_liab" in df.columns:
        console.print("\n[bold]资产负债表数据示例（前5只）:[/bold]")
        sample_cols = ["ts_code", "name", "money_cap", "total_liab", "total_assets"]
        available_cols = [c for c in sample_cols if c in df.columns]
        if available_cols:
            console.print(df[available_cols].head().to_string(index=False))


def fetch_stock_list(pro) -> pd.DataFrame:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("获取A股列表...", total=None)
        df = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        progress.update(task, completed=True)
    return df


def get_st_codes_from_names(df: pd.DataFrame) -> set:
    st_patterns = ["ST", r"\*ST", r"S\*ST", "SST", r"^S$"]
    mask = df["name"].str.contains("|".join(st_patterns), na=False, regex=True)
    return set(df.loc[mask, "ts_code"].tolist())


def filter_stocks(df: pd.DataFrame, st_codes: set) -> pd.DataFrame:
    console.print("\n[bold]开始过滤股票...[/bold]")
    initial_count = len(df)
    console.print(f"初始股票数量: {initial_count}")

    df = df[df["market"].isin(TARGET_MARKETS)]
    after_market = len(df)
    console.print(f"市场过滤后: {after_market} (排除 {initial_count - after_market} 只)")

    df = df[~df["ts_code"].isin(st_codes)]
    after_st = len(df)
    console.print(f"ST过滤后: {after_st} (排除 {after_market - after_st} 只)")

    console.print(f"排除行业: {', '.join(sorted(EXCLUDED_INDUSTRIES))}")
    df = df[~df["industry"].isin(EXCLUDED_INDUSTRIES)]
    after_industry = len(df)
    console.print(f"行业过滤后: {after_industry} (排除 {after_st - after_industry} 只)")

    return df


def display_results(df: pd.DataFrame):
    console.print("\n[bold green]过滤完成！[/bold green]")
    console.print(f"最终股票数量: {len(df)}\n")

    table = Table(title="按市场分布")
    table.add_column("市场", style="cyan")
    table.add_column("数量", style="magenta")
    table.add_column("占比", style="green")

    market_counts = df["market"].value_counts()
    total = len(df)
    for market, count in market_counts.items():
        pct = count / total * 100
        table.add_row(str(market), str(count), f"{pct:.1f}%")
    table.add_row("总计", str(total), "100.0%", style="bold")
    console.print(table)

    console.print("\n[bold]前10只股票示例：[/bold]")
    sample = df.head(10)[["ts_code", "name", "market", "industry"]]
    console.print(sample.to_string(index=False))


def main(limit: int = typer.Option(None, "--limit", "-l", help="限制处理的股票数量（用于测试）")):
    console.print("[bold blue]A股烟蒂股筛选器[/bold blue]")
    console.print("基于'扣除净现金后的十倍估值法'筛选A股烟蒂股\n")

    pro = get_tushare_pro()

    console.print("[bold]步骤1: 获取股票列表[/bold]")
    stock_df = fetch_stock_list(pro)

    st_codes = get_st_codes_from_names(stock_df)

    filtered_df = filter_stocks(stock_df, st_codes)

    if limit:
        filtered_df = filtered_df.head(limit)
        console.print(f"\n[yellow]测试模式：仅处理前 {limit} 只股票[/yellow]")

    display_results(filtered_df)

    console.print("\n[green]✓[/green] 股票列表获取与过滤完成")

    console.print("\n[bold]步骤2: 获取财务数据[/bold]")
    financial_df = fetch_financial_data(filtered_df, pro)

    display_financial_summary(financial_df)

    console.print("\n[green]✓[/green] 财务数据获取完成")

    console.print("\n[bold]步骤3: 执行烟蒂股筛选[/bold]")
    financial_df = add_risk_warnings(financial_df)

    t0_df = screen_t0_stocks(financial_df)
    display_screening_results(t0_df, "T0")

    t1_df = screen_t1_stocks(financial_df)
    display_screening_results(t1_df, "T1")

    t2_df = screen_t2_stocks(financial_df)
    display_screening_results(t2_df, "T2")

    console.print("\n[green]✓[/green] 筛选完成")

    console.print("\n[bold]步骤4: 导出CSV文件[/bold]")
    export_to_csv(t0_df, t1_df, t2_df)
    console.print("[green]✓[/green] CSV导出完成")


if __name__ == "__main__":
    typer.run(main)
