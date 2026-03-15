#!/usr/bin/env python3
"""黄金投资组合分析工具 - 一次性脚本

分析黄金 ETF、黄金股票与指数之间的相关性及收益对比。
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============ 配置区 ============
GOLD_ETF_CODE = "518880.SH"  # 华安黄金 ETF
STOCK_CODES = ["601899.SH", "600489.SH", "600547.SH"]  # 紫金矿业，中金黄金，山东黄金
INDEX_CODES = ["931238.CSI", "000985.CSI"]  # 中证黄金股指数，中证全指

ASSET_NAMES = {
    "518880.SH": "黄金 ETF",
    "601899.SH": "紫金矿业",
    "600489.SH": "中金黄金",
    "600547.SH": "山东黄金",
    "931238.CSI": "黄金股指数",
    "000985.CSI": "中证全指",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configure Chinese font support
plt.rcParams["font.sans-serif"] = ["MiSans", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9


# ============ 数据获取 ============
class DataFetcherError(Exception):
    """数据获取异常"""


def get_tushare_pro():
    """获取 Tushare Pro API 实例"""
    try:
        import tushare as ts
    except ImportError as e:
        raise DataFetcherError("tushare 未安装，运行：pip install tushare") from e

    return ts.pro_api()


def fetch_stock_prices(pro, ts_codes, start_date, end_date, adj="hfq"):
    """获取股票价格数据（后复权）"""
    results = {}
    for code in ts_codes:
        logger.info(f"Fetching stock: {code}")
        df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date, adj=adj)
        if df is None or df.empty:
            raise DataFetcherError(f"无数据：{code}")
        df = df.sort_values("trade_date").reset_index(drop=True)
        results[code] = df
        logger.info(f"  Got {len(df)} records")
    return results


def fetch_index_data(pro, index_codes, start_date, end_date):
    """获取指数数据"""
    results = {}
    for code in index_codes:
        logger.info(f"Fetching index: {code}")
        df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            raise DataFetcherError(f"无数据：{code}")
        df = df.sort_values("trade_date").reset_index(drop=True)
        results[code] = df
        logger.info(f"  Got {len(df)} records")
    return results


def align_trading_days(dataframes, date_col="trade_date"):
    """对齐交易日，返回收盘价 DataFrame"""
    price_series = {}
    for code, df in dataframes.items():
        if df.empty:
            continue
        df_normalized = df.copy()
        df_normalized[date_col] = pd.to_datetime(df_normalized[date_col]).dt.date
        df_dedup = df_normalized.drop_duplicates(subset=[date_col], keep="last")
        series = df_dedup.set_index(date_col)["close"]
        price_series[code] = series

    if not price_series:
        raise DataFetcherError("无有效数据")

    price_df = pd.DataFrame(price_series).dropna()
    if price_df.empty:
        raise DataFetcherError("无共同交易日")

    return price_df.sort_index()


# ============ 计算函数 ============
def calc_daily_returns(prices):
    """计算日收益率"""
    return prices.pct_change()


def calc_cumulative_returns(prices, base=100.0):
    """计算累计收益率"""
    daily_returns = calc_daily_returns(prices).fillna(0)
    return (1 + daily_returns).cumprod() * base


def calc_correlation_matrix(daily_returns):
    """计算相关性矩阵"""
    if daily_returns.empty:
        raise ValueError("收益率数据为空")
    if daily_returns.shape[1] < 2:
        raise ValueError("至少需要 2 个资产计算相关性")
    return daily_returns.corr(method="pearson")


# ============ 图表绘制 ============
def plot_returns_comparison(cumulative_returns, output_path, dpi=300):
    """绘制累计收益对比图"""
    figsize = (1920 / dpi, 1080 / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.set_facecolor("white")

    styles = [
        {"color": "#C41E3A", "label": "AU9999", "ls": "-"},
        {"color": "#0066CC", "label": "紫金矿业", "ls": "-"},
        {"color": "#228B22", "label": "中金黄金", "ls": "--"},
        {"color": "#FF8C00", "label": "山东黄金", "ls": "-"},
        {"color": "#8B008B", "label": "黄金股指数", "ls": "-"},
        {"color": "#4682B4", "label": "中证全指", "ls": "-."},
    ]

    for column, style in zip(cumulative_returns.columns, styles, strict=True):
        ax.plot(
            cumulative_returns.index,
            cumulative_returns[column],
            label=column,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.5,
            alpha=0.85,
        )

    ax.set_title("黄金投资组合累计收益率对比", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("日期", fontsize=10)
    ax.set_ylabel("累计收益率 (基准=100)", fontsize=10)

    y_min = cumulative_returns.min().min()
    y_max = cumulative_returns.max().max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, color="gray")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95, edgecolor="gray", fancybox=False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(correlation_matrix, output_path, dpi=300):
    """绘制相关性矩阵热力图"""
    figsize = (1000 / dpi, 800 / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.set_facecolor("white")

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "相关系数", "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8, "weight": "normal"},
    )

    ax.set_title("资产相关性矩阵", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


# ============ 主函数 ============
def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="黄金投资组合分析工具")
    parser.add_argument(
        "--start-date",
        type=str,
        default="20240101",
        help="开始日期（YYYYMMDD，默认：20240101）",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help="结束日期（YYYYMMDD，默认：今天）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录（默认：output）",
    )
    args = parser.parse_args()

    if not args.end_date:
        args.end_date = date.today().strftime("%Y%m%d")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录：{output_dir}")

    # 检查 Tushare Token
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        token_file = Path.home() / ".tushare_token"
        if token_file.exists():
            logger.info("Tushare token 文件找到")
        else:
            logger.warning("Tushare token 未设置，请设置 TUSHARE_TOKEN 环境变量")

    try:
        # 获取数据
        logger.info("初始化 Tushare API...")
        pro = get_tushare_pro()

        all_data = {}
        logger.info(f"获取黄金 ETF: {GOLD_ETF_CODE}")
        all_data[GOLD_ETF_CODE] = fetch_stock_prices(
            pro, [GOLD_ETF_CODE], args.start_date, args.end_date
        )[GOLD_ETF_CODE]

        logger.info("获取黄金股票...")
        all_data.update(fetch_stock_prices(pro, STOCK_CODES, args.start_date, args.end_date))

        logger.info("获取指数数据...")
        all_data.update(fetch_index_data(pro, INDEX_CODES, args.start_date, args.end_date))

        # 对齐交易日并计算
        logger.info("对齐交易日...")
        aligned_prices = align_trading_days(all_data)
        aligned_prices.columns = [ASSET_NAMES.get(col, col) for col in aligned_prices.columns]
        logger.info(f"对齐后数据形状：{aligned_prices.shape}")

        logger.info("计算累计收益率...")
        cumulative_returns = pd.DataFrame(
            {
                col: calc_cumulative_returns(aligned_prices[col], base=100.0)
                for col in aligned_prices.columns
            }
        )

        daily_returns = pd.DataFrame(
            {col: calc_daily_returns(aligned_prices[col]) for col in aligned_prices.columns}
        )

        logger.info("计算相关性矩阵...")
        corr_matrix = calc_correlation_matrix(daily_returns)

        # 生成图表
        returns_path = output_dir / "returns_comparison.png"
        logger.info(f"生成收益对比图：{returns_path}")
        plot_returns_comparison(cumulative_returns, returns_path)
        logger.info(f"  已保存至 {returns_path}")

        corr_path = output_dir / "correlation_matrix.png"
        logger.info(f"生成相关性矩阵图：{corr_path}")
        plot_correlation_matrix(corr_matrix, corr_path)
        logger.info(f"  已保存至 {corr_path}")

        # 输出结果
        logger.info("=" * 50)
        logger.info("分析完成！")
        logger.info(f"图表已保存至：{output_dir}")
        logger.info("=" * 50)

        print("\n最终累计收益率（基准=100）:")
        print("-" * 40)
        final_returns = cumulative_returns.iloc[-1]
        for name, value in final_returns.items():
            print(f"  {name}: {value:.2f}")

        print("\n相关性矩阵:")
        print("-" * 40)
        print(corr_matrix.round(3).to_string())

        return 0

    except DataFetcherError as e:
        logger.error(f"数据获取错误：{e}")
        print(f"\n错误：{e}", file=sys.stderr)
        print("\n请检查:", file=sys.stderr)
        print("  1. TUSHARE_TOKEN 环境变量已正确设置", file=sys.stderr)
        print("  2. 网络连接正常", file=sys.stderr)
        print("  3. 日期范围有效", file=sys.stderr)
        return 1

    except Exception as e:
        logger.exception(f"意外错误：{e}")
        print(f"\n意外错误：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
