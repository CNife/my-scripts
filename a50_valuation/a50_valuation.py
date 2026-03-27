"""中证A50指数成分股 EP/BP 分位点及股息率计算。

以中证A50 (930050.CSI) 成分股为股票池，使用周频数据计算 EP/BP 分位点：
- EP = EPS / 收盘价（每股收益/股价）
- BP = BVPS / 收盘价（每股净资产/股价）
可正确处理盈利为负的企业。

数据获取策略：
- 价格数据：按周频 trade_date 拉全市场 daily 收盘价，本地过滤成分股
- 财务数据：从 fina_indicator 获取 EPS 和 BVPS，按 ann_date 前向填充到周频
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
    today = datetime.now().strftime("%Y%m%d")
    df = pro.trade_cal(exchange="SSE", is_open=1, start_date="20260101", end_date=today)
    return df["cal_date"].max()


def get_index_constituents(pro, trade_date: str) -> list[str]:
    df = pro.index_weight(index_code=INDEX_CODE, start_date=trade_date, end_date=trade_date)
    if df.empty:
        df = pro.index_weight(index_code=INDEX_CODE, end_date=trade_date)
        latest_date = df["trade_date"].max()
        df = df[df["trade_date"] == latest_date]
    codes = df["con_code"].tolist()
    print(f"成分股数量: {len(codes)}")
    return codes


def get_stock_names(pro, ts_codes: list[str]) -> dict[str, str]:
    df = pro.stock_basic(fields="ts_code,name")
    return dict(zip(df["ts_code"], df["name"], strict=True))


def get_weekly_trade_dates(pro, end_date: str, lookback_years: int) -> list[str]:
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


def fetch_weekly_prices(pro, ts_codes: list[str], weekly_dates: list[str]) -> pd.DataFrame:
    ts_code_set = set(ts_codes)
    all_records = []

    for trade_date in tqdm(weekly_dates, desc="拉取周频价格"):
        for attempt in range(3):
            try:
                df = pro.daily(trade_date=trade_date, fields="ts_code,trade_date,close")
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
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])

    result = pd.concat(all_records, ignore_index=True)
    result = result.drop_duplicates(subset=["ts_code", "trade_date"]).sort_values(
        ["ts_code", "trade_date"]
    )
    return result


def fetch_financial_data(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,ann_date,end_date,eps,bps",
            )
            if df is not None and not df.empty:
                df = df.sort_values("ann_date")
                return df
            return pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "eps", "bps"])
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
            continue
    return pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "eps", "bps"])


def align_financial_to_weekly(financial_df: pd.DataFrame, weekly_dates: list[str]) -> pd.DataFrame:
    """将季度财务数据前向填充到周频交易日。

    按 ann_date（公告日）生效：在公告日当天及之后使用新数据，
    在公告日之前使用上一期数据。
    """
    if financial_df.empty:
        return pd.DataFrame(
            {
                "trade_date": weekly_dates,
                "eps": [float("nan")] * len(weekly_dates),
                "bps": [float("nan")] * len(weekly_dates),
            }
        )

    weekly_dates_dt = pd.to_datetime(weekly_dates, format="%Y%m%d")
    financial_df["ann_date_dt"] = pd.to_datetime(financial_df["ann_date"], format="%Y%m%d")

    result = []
    for trade_date in weekly_dates_dt:
        available = financial_df[financial_df["ann_date_dt"] <= trade_date]
        if available.empty:
            result.append(
                {
                    "trade_date": trade_date.strftime("%Y%m%d"),
                    "eps": float("nan"),
                    "bps": float("nan"),
                }
            )
        else:
            latest = available.iloc[-1]
            result.append(
                {
                    "trade_date": trade_date.strftime("%Y%m%d"),
                    "eps": latest["eps"],
                    "bps": latest["bps"],
                }
            )

    return pd.DataFrame(result)


def calc_ep_bp(prices_df: pd.DataFrame, financial_aligned_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(prices_df, financial_aligned_df, on="trade_date", how="inner")
    merged["ep"] = merged["eps"] / merged["close"]
    merged["bp"] = merged["bps"] / merged["close"]
    return merged[["ts_code", "trade_date", "ep", "bp"]]


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


def calc_dividend_yield(pro, ts_code: str, ep: float) -> float:
    """计算股息率 = 3年平均股利支付率 × EP × 100。

    无分红或盈利为负的年份，股利支付率设为 0。
    """
    if pd.isna(ep) or ep <= 0:
        return float("nan")

    dividends = fetch_dividend_data(pro, ts_code)
    eps_data = fetch_eps_data(pro, ts_code)

    if eps_data.empty:
        return float("nan")

    div_annual = dividends.groupby("end_date", as_index=False)["cash_div"].sum()

    merged = pd.merge(div_annual, eps_data, on="end_date", how="outer")
    merged = merged.sort_values("end_date", ascending=False).head(PAYOUT_LOOKBACK_YEARS)

    if merged.empty:
        return float("nan")

    payout_ratios = []
    for _, row in merged.iterrows():
        dps = row.get("cash_div", 0) or 0
        eps = row.get("basic_eps", None)
        if pd.isna(eps) or eps is None or eps <= 0:
            payout_ratios.append(0.0)
        else:
            payout_ratios.append(dps / eps)

    avg_payout_ratio = sum(payout_ratios) / len(payout_ratios)
    return avg_payout_ratio * ep * 100


def calc_percentile(series: pd.Series, current_value: float) -> float:
    """计算分位点：P(历史 >= 当前值)。

    用于 EP/BP，值越高越便宜，低分位点 = 被低估。
    """
    if pd.isna(current_value) or len(series) == 0:
        return float("nan")
    historical = series.dropna()
    if len(historical) == 0:
        return float("nan")
    return (historical >= current_value).sum() / len(historical) * 100


def build_markdown_table(result_df: pd.DataFrame) -> str:
    lines = []
    cols = ["ts_code", "name", "ep", "ep_percentile", "bp", "bp_percentile", "dividend_yield"]
    headers = ["排名", "代码", "名称", "EP", "EP分位点(%)", "BP", "BP分位点(%)", "股息率(%)"]
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
                vals.append(f"{v:.4f}" if col in ("ep", "bp") else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main():
    pro = get_pro()

    latest_date = get_latest_trade_date(pro)
    print(f"最新交易日: {latest_date}")

    ts_codes = get_index_constituents(pro, latest_date)

    names = get_stock_names(pro, ts_codes)

    weekly_dates = get_weekly_trade_dates(pro, latest_date, LOOKBACK_YEARS)
    start_date = weekly_dates[0]

    print("\n开始拉取价格数据...")
    prices_data = fetch_weekly_prices(pro, ts_codes, weekly_dates)
    print(f"价格数据行数: {len(prices_data)}")

    if prices_data.empty:
        print("无价格数据，退出")
        return

    print("\n开始拉取财务数据并计算 EP/BP...")
    results = []
    for ts_code in tqdm(ts_codes, desc="逐股计算"):
        stock_prices = prices_data[prices_data["ts_code"] == ts_code].sort_values("trade_date")

        if stock_prices.empty:
            print(f"  {ts_code} ({names.get(ts_code, '')}) 无价格数据")
            continue

        financial_df = fetch_financial_data(pro, ts_code, start_date, latest_date)
        financial_aligned = align_financial_to_weekly(financial_df, weekly_dates)

        ep_bp_df = calc_ep_bp(stock_prices, financial_aligned)

        if ep_bp_df.empty:
            print(f"  {ts_code} ({names.get(ts_code, '')}) 无 EP/BP 数据")
            continue

        current_ep = ep_bp_df.iloc[-1]["ep"]
        current_bp = ep_bp_df.iloc[-1]["bp"]

        ep_pct = calc_percentile(ep_bp_df["ep"], current_ep)
        bp_pct = calc_percentile(ep_bp_df["bp"], current_bp)
        div_yield = calc_dividend_yield(pro, ts_code, current_ep)

        results.append(
            {
                "ts_code": ts_code,
                "name": names.get(ts_code, ""),
                "ep": round(current_ep, 4) if pd.notna(current_ep) else float("nan"),
                "bp": round(current_bp, 4) if pd.notna(current_bp) else float("nan"),
                "ep_percentile": round(ep_pct, 1) if pd.notna(ep_pct) else float("nan"),
                "bp_percentile": round(bp_pct, 1) if pd.notna(bp_pct) else float("nan"),
                "dividend_yield": round(div_yield, 2) if pd.notna(div_yield) else float("nan"),
                "data_points": len(ep_bp_df),
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("ep_percentile", na_position="last").reset_index(drop=True)

    output_path = OUTPUT_DIR / "a50_valuation.md"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# 中证A50指数成分股 EP/BP 分位点及股息率\n\n")
        f.write(f"- **指数代码**: {INDEX_CODE}\n")
        f.write(f"- **成分股数量**: {len(ts_codes)}\n")
        f.write("- **数据频率**: 周频（每周最后一个交易日）\n")
        f.write(f"- **回溯区间**: {start_date} ~ {latest_date} ({LOOKBACK_YEARS}年)\n")
        f.write("- **计算方式**: EP = EPS/收盘价, BP = BVPS/收盘价\n")
        f.write("- **分位点定义**: P(历史 >= 当前值)，低分位点 = 被低估\n")
        f.write(
            f"- **股息率**: {PAYOUT_LOOKBACK_YEARS}年平均股利支付率 × EP × 100 "
            f"（无分红/亏损年份支付率=0）\n"
        )
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(build_markdown_table(result_df))
        f.write("\n")

    print(f"\n结果已输出到: {output_path}")
    print(f"共 {len(result_df)} 只股票")
    print("\nEP分位点最低的5只:")
    for _, row in result_df.head(5).iterrows():
        print(
            f"  {row['name']}({row['ts_code']}): "
            f"EP={row['ep']}, EP分位点={row['ep_percentile']}%, "
            f"股息率={row['dividend_yield']}%"
        )


if __name__ == "__main__":
    main()
