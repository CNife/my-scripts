"""中证A50指数成分股PE/PB分位点及股息率计算。

以中证A50 (930050.CSI) 成分股为股票池，使用周频 daily_basic 数据计算 PE/PB 分位点，
使用 income 和 dividend 接口计算股息率（3年平均股利支付率 / PE_TTM），
按 PE 分位点从低到高排列，输出为 Markdown 文件。

数据获取策略：按周频 trade_date 拉全市场 daily_basic，本地过滤成分股。
每次 API 调用拉一个交易日的全市场数据（约5000+行），10年周频约520次调用。
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts
from tqdm import tqdm

INDEX_CODE = "930050.CSI"
LOOKBACK_YEARS = 10
PAYOUT_LOOKBACK_YEARS = 3
OUTPUT_DIR = Path(__file__).parent


def get_pro():
    return ts.pro_api()


def get_latest_trade_date(pro) -> str:
    """获取最近的交易日。"""
    today = datetime.now().strftime("%Y%m%d")
    df = pro.trade_cal(exchange="SSE", is_open=1, start_date="20260101", end_date=today)
    return df["cal_date"].max()


def get_index_constituents(pro, trade_date: str) -> list[str]:
    """获取中证A50最新成分股代码列表。"""
    df = pro.index_weight(index_code=INDEX_CODE, start_date=trade_date, end_date=trade_date)
    if df.empty:
        df = pro.index_weight(index_code=INDEX_CODE, end_date=trade_date)
        latest_date = df["trade_date"].max()
        df = df[df["trade_date"] == latest_date]
    codes = df["con_code"].tolist()
    print(f"成分股数量: {len(codes)}")
    return codes


def get_stock_names(pro, ts_codes: list[str]) -> dict[str, str]:
    """获取股票名称映射。"""
    df = pro.stock_basic(fields="ts_code,name")
    return dict(zip(df["ts_code"], df["name"], strict=True))


def get_weekly_trade_dates(pro, end_date: str, lookback_years: int) -> list[str]:
    """获取回溯期内的周频交易日（每周最后一个交易日）。"""
    start_date = (
        datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365 * lookback_years)
    ).strftime("%Y%m%d")
    df = pro.trade_cal(exchange="SSE", is_open=1, start_date=start_date, end_date=end_date)
    all_dates = pd.to_datetime(df["cal_date"], format="%Y%m%d")

    weekly = all_dates.to_frame(name="date")
    weekly["year_week"] = (
        weekly["date"].dt.isocalendar().year.astype(str)
        + "_"
        + weekly["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    last_per_week = weekly.groupby("year_week")["date"].max()
    result = last_per_week.sort_values().dt.strftime("%Y%m%d").tolist()
    print(f"周频交易日数量: {len(result)}（{result[0]} ~ {result[-1]}）")
    return result


def fetch_weekly_daily_basic(pro, ts_codes: list[str], weekly_dates: list[str]) -> pd.DataFrame:
    """按周频交易日拉全市场 daily_basic，过滤成分股。

    每次调用拉一个 trade_date 的全市场数据，在本地过滤目标股票。
    """
    ts_code_set = set(ts_codes)
    all_records = []

    for trade_date in tqdm(weekly_dates, desc="拉取周频数据"):
        for attempt in range(3):
            try:
                df = pro.daily_basic(trade_date=trade_date, fields="ts_code,trade_date,pe_ttm,pb")
                if df is not None and not df.empty:
                    filtered = df[df["ts_code"].isin(ts_code_set)]
                    if not filtered.empty:
                        all_records.append(filtered)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1)
                continue

    if not all_records:
        return pd.DataFrame(columns=["ts_code", "trade_date", "pe_ttm", "pb"])

    result = pd.concat(all_records, ignore_index=True)
    result = result.drop_duplicates(subset=["ts_code", "trade_date"]).sort_values(
        ["ts_code", "trade_date"]
    )
    return result


def fetch_dividend_data(pro, ts_code: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = pro.dividend(ts_code=ts_code, fields="ts_code,end_date,cash_div")
            if df is not None and not df.empty:
                annual = df[df["end_date"].str.endswith("1231")].copy()
                annual = annual.sort_values("end_date", ascending=False)
                return annual
            return pd.DataFrame(columns=["ts_code", "end_date", "cash_div"])
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
            continue
    return pd.DataFrame(columns=["ts_code", "end_date", "cash_div"])


def fetch_eps_data(pro, ts_code: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = pro.income(
                ts_code=ts_code, fields="ts_code,ann_date,end_date,basic_eps", report_type="1"
            )
            if df is not None and not df.empty:
                annual = df[df["end_date"].str.endswith("1231")].copy()
                annual = annual.drop_duplicates(subset=["end_date"], keep="first")
                annual = annual.sort_values("end_date", ascending=False)
                return annual[["ts_code", "end_date", "basic_eps"]]
            return pd.DataFrame(columns=["ts_code", "end_date", "basic_eps"])
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
            continue
    return pd.DataFrame(columns=["ts_code", "end_date", "basic_eps"])


def calc_dividend_yield(pro, ts_code: str, pe_ttm: float) -> float:
    """计算股息率 = 3年平均股利支付率 / PE_TTM。

    股利支付率 = 每股分红 / 每股收益，取最近3个完整财年的算术平均。
    EPS为负的年份跳过。
    """
    if pd.isna(pe_ttm) or pe_ttm <= 0:
        return float("nan")

    dividends = fetch_dividend_data(pro, ts_code)
    eps_data = fetch_eps_data(pro, ts_code)

    if dividends.empty or eps_data.empty:
        return float("nan")

    div_annual = dividends.groupby("end_date", as_index=False)["cash_div"].sum()
    div_annual = div_annual[div_annual["cash_div"] > 0]

    merged = pd.merge(div_annual, eps_data, on="end_date", how="inner")
    merged = merged.sort_values("end_date", ascending=False).head(PAYOUT_LOOKBACK_YEARS)

    if merged.empty:
        return float("nan")

    payout_ratios = []
    for _, row in merged.iterrows():
        dps = row["cash_div"]
        eps = row["basic_eps"]
        if pd.isna(dps) or pd.isna(eps) or eps <= 0:
            continue
        payout_ratios.append(dps / eps)

    if not payout_ratios:
        return float("nan")

    avg_payout_ratio = sum(payout_ratios) / len(payout_ratios)
    return avg_payout_ratio / pe_ttm * 100  # 转为百分比


def calc_percentile(series: pd.Series, current_value: float) -> float:
    """计算分位点：历史数据中 <= 当前值的比例。"""
    if pd.isna(current_value) or len(series) == 0:
        return float("nan")
    historical = series.dropna()
    if len(historical) == 0:
        return float("nan")
    return (historical <= current_value).sum() / len(historical) * 100


def build_markdown_table(result_df: pd.DataFrame) -> str:
    """手动构建 Markdown 表格，避免依赖 tabulate。"""
    lines = []
    cols = ["ts_code", "name", "pe_ttm", "pe_percentile", "pb", "pb_percentile", "dividend_yield"]
    headers = ["排名", "代码", "名称", "PE(TTM)", "PE分位点(%)", "PB", "PB分位点(%)", "股息率(%)"]
    aligns = ["---:"] + [":---"] * 2 + ["---:"] * 5

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")

    for rank, (_, row) in enumerate(result_df.iterrows(), 1):
        vals = [str(rank)]
        for col in cols:
            v = row[col]
            if pd.isna(v):
                vals.append("-")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}" if col in ("pe_ttm", "pb", "dividend_yield") else f"{v:.1f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main():
    pro = get_pro()

    # 1. 获取最新交易日
    latest_date = get_latest_trade_date(pro)
    print(f"最新交易日: {latest_date}")

    # 2. 获取成分股
    ts_codes = get_index_constituents(pro, latest_date)

    # 3. 获取股票名称
    names = get_stock_names(pro, ts_codes)

    # 4. 获取周频交易日列表
    weekly_dates = get_weekly_trade_dates(pro, latest_date, LOOKBACK_YEARS)

    # 5. 批量拉取周频 daily_basic 数据
    print("\n开始拉取数据...")
    all_data = fetch_weekly_daily_basic(pro, ts_codes, weekly_dates)
    print(f"总数据行数: {len(all_data)}")

    if all_data.empty:
        print("无数据，退出")
        return

    # 6. 逐股计算分位点和股息率
    print("\n开始计算分位点和股息率...")
    results = []
    for ts_code in tqdm(ts_codes, desc="逐股计算"):
        stock_data = all_data[all_data["ts_code"] == ts_code].sort_values("trade_date")

        if stock_data.empty:
            print(f"  {ts_code} ({names.get(ts_code, '')}) 无数据")
            continue

        current_pe = stock_data.iloc[-1]["pe_ttm"]
        current_pb = stock_data.iloc[-1]["pb"]

        pe_pct = calc_percentile(stock_data["pe_ttm"], current_pe)
        pb_pct = calc_percentile(stock_data["pb"], current_pb)
        div_yield = calc_dividend_yield(pro, ts_code, current_pe)

        results.append(
            {
                "ts_code": ts_code,
                "name": names.get(ts_code, ""),
                "pe_ttm": round(current_pe, 2) if pd.notna(current_pe) else float("nan"),
                "pb": round(current_pb, 2) if pd.notna(current_pb) else float("nan"),
                "pe_percentile": round(pe_pct, 1) if pd.notna(pe_pct) else float("nan"),
                "pb_percentile": round(pb_pct, 1) if pd.notna(pb_pct) else float("nan"),
                "dividend_yield": round(div_yield, 2) if pd.notna(div_yield) else float("nan"),
                "data_points": len(stock_data),
            }
        )

    # 7. 按 PE 分位点排序
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("pe_percentile", na_position="last").reset_index(drop=True)

    # 8. 输出 Markdown
    output_path = OUTPUT_DIR / "a50_valuation.md"
    start_date = weekly_dates[0]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# 中证A50指数成分股 PE/PB 分位点及股息率\n\n")
        f.write(f"- **指数代码**: {INDEX_CODE}\n")
        f.write(f"- **成分股数量**: {len(ts_codes)}\n")
        f.write("- **数据频率**: 周频（每周最后一个交易日）\n")
        f.write(f"- **回溯区间**: {start_date} ~ {latest_date} ({LOOKBACK_YEARS}年)\n")
        f.write("- **分位点定义**: 历史周频数据中 ≤ 当前值的比例\n")
        f.write(
            f"- **股息率**: {PAYOUT_LOOKBACK_YEARS}年平均股利支付率 / PE_TTM "
            f"（股利支付率=每股分红/每股收益）\n"
        )
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(build_markdown_table(result_df))
        f.write("\n")

    print(f"\n结果已输出到: {output_path}")
    print(f"共 {len(result_df)} 只股票")
    print("\nPE分位点最低的5只:")
    for _, row in result_df.head(5).iterrows():
        print(
            f"  {row['name']}({row['ts_code']}): "
            f"PE={row['pe_ttm']}, 分位点={row['pe_percentile']}%, "
            f"股息率={row['dividend_yield']}%"
        )


if __name__ == "__main__":
    main()
