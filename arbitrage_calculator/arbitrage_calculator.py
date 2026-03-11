def calculate_arbitrage():
    """
    计算中金换股吸收合并东兴/信达的套利空间
    输入：中金股价、东兴股价、信达股价
    输出：正向换股套利收益、现金选择权套利收益
    """
    # 固定参数（换股比例、现金选择权价格，无需修改）
    params = {
        "东兴证券": {"换股比例": 0.4373, "现金选择权价格": 13.13},
        "信达证券": {"换股比例": 0.5188, "现金选择权价格": 17.79},
    }

    # 输入股价（做异常处理，防止输入非数字）
    try:
        cns_price = float(input("请输入中金公司当前股价（元）："))
        dx_price = float(input("请输入东兴证券当前股价（元）："))
        xd_price = float(input("请输入信达证券当前股价（元）："))
    except ValueError:
        print("❌ 输入错误！请输入数字格式的股价（例如：34.20）")
        return

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


# 执行函数
if __name__ == "__main__":
    calculate_arbitrage()
