import os

import tushare as ts

# 股票代码映射
STOCK_CODES = {
    "中金公司": "601995.SH",
    "东兴证券": "601198.SH",
    "信达证券": "601059.SH",
}

# 换股吸收合并参数（2026-05-18 公告报告书草案）
MERGER_PARAMS = {
    "中金公司": {"换股价格": 36.68},
    "东兴证券": {
        "换股比例": 0.4376,
        "换股价格": 16.05,
        "现金选择权价格": 13.13,
    },
    "信达证券": {
        "换股比例": 0.5210,
        "换股价格": 19.11,
        "现金选择权价格": 17.79,
    },
}


def get_latest_prices():
    """通过 tushare 获取实时股价"""
    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        print("⚠️ 未找到 TUSHARE_TOKEN 环境变量，将使用手动输入模式")
        return None

    try:
        pro = ts.pro_api(token)
        codes = list(STOCK_CODES.values())
        df = pro.rt_k(ts_code=",".join(codes))

        if df.empty:
            print("⚠️ 未获取到股价数据，将使用手动输入模式")
            return None

        latest_prices = {}
        for name, code in STOCK_CODES.items():
            stock_df = df[df["ts_code"] == code]
            if not stock_df.empty:
                latest_prices[name] = stock_df.iloc[0]["close"]
            else:
                print(f"⚠️ 未找到 {name}({code}) 的股价数据")

        return latest_prices if len(latest_prices) == len(STOCK_CODES) else None

    except Exception as e:
        print(f"⚠️ 获取股价失败：{e}，将使用手动输入模式")
        return None


def input_prices_manually():
    """手动输入股价"""
    try:
        prices = {
            "中金公司": float(input("请输入中金公司当前股价（元）：")),
            "东兴证券": float(input("请输入东兴证券当前股价（元）：")),
            "信达证券": float(input("请输入信达证券当前股价（元）：")),
        }
        return prices
    except ValueError:
        print("❌ 输入错误！请输入数字格式的股价（例如：34.20）")
        return None


def input_position():
    """可选：输入持仓信息，用于个性化分析"""
    try:
        resp = input("\n是否输入持仓做个性化分析？(y/n，默认 n): ").strip().lower()
        if resp != "y":
            return None
        name = input("证券名称（东兴证券/信达证券，默认东兴证券）: ").strip() or "东兴证券"
        shares = int(input("持仓股数: ").strip())
        cost = float(input("平均成本（元/股）: ").strip())
        return {"name": name, "shares": shares, "cost": cost}
    except (ValueError, EOFError):
        print("⚠️ 输入无效，跳过持仓分析")
        return None


def print_section(title: str):
    """打印分区标题"""
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def calculate_arbitrage(prices: dict):
    """计算中金换股吸收合并东兴/信达的套利空间"""
    cns_price = prices["中金公司"]

    print_section("换股方案摘要（2026-05-18 公告）")
    print(f"中金公司换股价格: {MERGER_PARAMS['中金公司']['换股价格']} 元/股")
    for name in ["东兴证券", "信达证券"]:
        p = MERGER_PARAMS[name]
        print(f"  {name}: 换股价格 {p['换股价格']} 元/股")
        print(f"         换股比例 1:{p['换股比例']}（1股{name[:2]} = {p['换股比例']}股中金）")
        print(f"         现金选择权 {p['现金选择权价格']} 元/股")

    print_section("套利空间计算")
    for name in ["东兴证券", "信达证券"]:
        p = MERGER_PARAMS[name]
        sp = prices[name]

        # 正向换股套利：买入被合并方 → 换股 → 卖出中金
        eq_cns = sp / p["换股比例"]  # 被合并方当前价折算成等效中金成本
        swap_arb = (cns_price - eq_cns) / eq_cns * 100

        # 现金选择权套利
        cash_arb = (p["现金选择权价格"] - sp) / sp * 100

        # 换股平价对比：东兴公允价值 = 中金价 × 换股比例
        parity = cns_price * p["换股比例"]

        print(f"\n【{name}】")
        print(f"  当前股价: {sp:.2f} 元")
        print(f"  换股平价（公允价值）: {parity:.2f} 元（中金{cns_price:.2f} × {p['换股比例']}）")
        print(f"  折价/溢价: {(sp - parity) / parity * 100:+.2f}%（负=折价，正=溢价）")
        print(
            f"  正向换股套利收益: {swap_arb:.2f}%（等效中金成本 {eq_cns:.2f} 元 → 中金现价 {cns_price:.2f} 元）"
        )
        print(f"  现金选择权套利收益: {cash_arb:.2f}%")

    print_section("结果解读")
    print("1. 正向换股套利：收益>3% 具备实操价值；收益≤0 无套利价值")
    print("2. 现金选择权套利：收益>0 有保底收益（需投反对票+持续持股）；收益≤0 无价值")
    print("3. 换股平价对比：东兴/信达低于平价 = 折价买入换股权，有利加仓")
    print("4. 优先正向换股套利，仅当正向套利为负时考虑现金选择权")


def analyze_position(prices: dict, pos: dict):
    """分析个人持仓的三种方案"""
    p = MERGER_PARAMS[pos["name"]]
    sp = prices[pos["name"]]
    cns_price = prices["中金公司"]

    shares = pos["shares"]
    cost = pos["cost"]
    total_cost = shares * cost

    print_section(f"持仓分析 — {pos['name']} {shares}股 @ {cost}元")

    # 方案一：持有至换股
    cns_received = shares * p["换股比例"]
    eq_cns = cost / p["换股比例"]
    swap_value = cns_received * cns_price
    swap_profit = swap_value - total_cost

    # 方案二：现金选择权
    cash_profit = (p["现金选择权价格"] - cost) * shares

    # 方案三：现在卖出
    sell_profit = (sp - cost) * shares

    print(f"  总成本: {total_cost:.2f} 元\n")

    print("  【方案一】持有至换股")
    print(f"    获得中金: {cns_received:.2f} 股（不足1股发现金）")
    print(f"    等效中金成本: {eq_cns:.2f} 元/股")
    print(f"    换股价值: {swap_value:.2f} 元（中金{cns_price:.2f}计价）")
    print(f"    预期盈亏: {swap_profit:+.2f} 元（{swap_profit / total_cost * 100:+.2f}%）")
    # 中金保本价
    cns_breakeven = cost / p["换股比例"]
    print(
        f"    中金保本价: {cns_breakeven:.2f} 元（中金现价{cns_price:.2f}，安全空间 {abs(cns_price - cns_breakeven) / cns_price * 100:.1f}%）"
    )

    print("\n  【方案二】现金选择权（需投反对票）")
    print(
        f"    行权价 {p['现金选择权价格']} 元/股 → 盈亏: {cash_profit:+.2f} 元（{cash_profit / total_cost * 100:+.2f}%）"
    )

    print("\n  【方案三】当前卖出")
    print(
        f"    现价 {sp:.2f} 元/股 → 盈亏: {sell_profit:+.2f} 元（{sell_profit / total_cost * 100:+.2f}%）"
    )

    # 加仓/减仓参考
    print("\n  【交易参考】")
    parity = cns_price * p["换股比例"]
    discount_pct = (sp - parity) / parity * 100
    print(f"    换股平价: {parity:.2f} 元（中金{cns_price:.2f} × {p['换股比例']}）")
    print(f"    当前折价: {discount_pct:.2f}%（负=市价低于平价）")
    if discount_pct < 0:
        print("    → 市价低于换股平价，加仓可降低等效中金成本")
    else:
        print("    → 市价高于换股平价，可考虑提前卖出锁定利润")
    print(f"    加仓盈亏平衡: 买入价 ≤ {parity:.2f} 元时，换股后等效中金成本 ≤ 中金现价")


def main():
    """主函数"""
    print("===== 中金换股套利计算器 v2（2026-05-18 方案）=====")

    # 获取股价
    print("\n正在获取最新股价...")
    prices = get_latest_prices()

    if prices:
        print("✅ 股价获取成功：")
        for name, price in prices.items():
            print(f"  {name}: {price:.2f} 元")
    else:
        print("\n请手动输入股价：")
        prices = input_prices_manually()
        if not prices:
            return

    # 套利空间计算
    calculate_arbitrage(prices)

    # 持仓分析（可选）
    pos = input_position()
    if pos:
        analyze_position(prices, pos)


if __name__ == "__main__":
    main()
