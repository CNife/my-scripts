import os

import tushare as ts

# 股票代码映射
STOCK_CODES = {
    "中金公司": "601995.SH",
    "东兴证券": "601198.SH",
    "信达证券": "601059.SH",
}


def get_latest_prices():
    """
    通过 tushare 获取实时股价
    返回: dict {股票名称: 最新价} 或 None（获取失败时）
    """
    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        print("⚠️ 未找到 TUSHARE_TOKEN 环境变量，将使用手动输入模式")
        return None

    try:
        pro = ts.pro_api(token)
        # 使用 rt_k 接口获取实时行情
        codes = list(STOCK_CODES.values())
        df = pro.rt_k(ts_code=",".join(codes))

        if df.empty:
            print("⚠️ 未获取到股价数据，将使用手动输入模式")
            return None

        # rt_k 返回的 close 字段是最新价
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
    """
    手动输入股价
    返回: dict {股票名称: 股价} 或 None（输入错误时）
    """
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


def calculate_arbitrage(prices: dict):
    """
    计算中金换股吸收合并东兴/信达的套利空间
    输入: dict {股票名称: 股价}
    输出: 打印套利计算结果
    """
    # 固定参数（换股比例、现金选择权价格，无需修改）
    params = {
        "东兴证券": {"换股比例": 0.4373, "现金选择权价格": 13.13},
        "信达证券": {"换股比例": 0.5188, "现金选择权价格": 17.79},
    }

    cns_price = prices["中金公司"]
    dx_price = prices["东兴证券"]
    xd_price = prices["信达证券"]

    # 计算套利空间
    print("\n===== 套利空间计算结果 =====")

    # 1. 东兴证券套利计算
    # 正向换股套利
    dx_equivalent_cns = dx_price / params["东兴证券"]["换股比例"]  # 东兴换股后等效中金成本
    dx_swap_arbitrage = (cns_price - dx_equivalent_cns) / dx_equivalent_cns * 100
    # 现金选择权套利
    dx_cash_arbitrage = (params["东兴证券"]["现金选择权价格"] - dx_price) / dx_price * 100

    # 2. 信达证券套利计算
    # 正向换股套利
    xd_equivalent_cns = xd_price / params["信达证券"]["换股比例"]  # 信达换股后等效中金成本
    xd_swap_arbitrage = (cns_price - xd_equivalent_cns) / xd_equivalent_cns * 100
    # 现金选择权套利
    xd_cash_arbitrage = (params["信达证券"]["现金选择权价格"] - xd_price) / xd_price * 100

    # 输出结果（保留2位小数，更直观）
    print("【东兴证券】")
    print(f"正向换股套利收益：{dx_swap_arbitrage:.2f}%")
    print(f"现金选择权套利收益：{dx_cash_arbitrage:.2f}%")

    print("\n【信达证券】")
    print(f"正向换股套利收益：{xd_swap_arbitrage:.2f}%")
    print(f"现金选择权套利收益：{xd_cash_arbitrage:.2f}%")

    # 结果解读提示
    print("\n===== 结果解读 =====")
    print("1. 正向换股套利：收益＞3% 具备实操价值（覆盖税费+资金成本），收益≤0 无套利价值；")
    print("2. 现金选择权套利：收益＞0 有保底收益（需满足投反对票+持续持股条件），收益≤0 无价值；")
    print("3. 优先选择正向换股套利（收益更高），仅当正向套利为负时考虑现金选择权。")


def main():
    """
    主函数：自动获取股价并计算套利空间
    """
    print("===== 中金换股套利计算器 =====")

    # 尝试自动获取股价
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

    # 计算套利空间
    calculate_arbitrage(prices)


if __name__ == "__main__":
    main()
