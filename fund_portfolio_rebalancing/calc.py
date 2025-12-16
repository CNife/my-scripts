import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ===================== 1. 基础数据定义 =====================
# 标的名称、现有市值（元）、目标比例（小数）
data = {
    "标的": [
        "020602 红利低波",
        "023917 自由现金流",
        "021457 恒生红利低波",
        "006961 长期国债",
        "008701 黄金",
        "纳斯达克100",
        "标普500",
    ],
    "现有市值": [385.27, 1919.70, 395.81, 399.91, 3213.58, 2928.13, 2808.87],
    "目标比例": [0.15, 0.14, 0.14, 0.14, 0.15, 0.14, 0.14],
}
目标金额 = 13000
追加资金总额 = 目标金额 - sum(data["现有市值"])
# data = {"标的": ["红利", "大盘", "成长"], "现有市值": [3416.54, 3507.79, 0.00], "目标比例": [0.5, 0.3, 0.2]}
# 追加资金总额 = 1400
追加资金总额整数 = int(np.rint(追加资金总额))

df = pd.DataFrame(data)
现有总市值 = df["现有市值"].sum()
变动后总市值 = 现有总市值 + 追加资金总额整数


# ===================== 2. 定义优化目标函数 =====================
# 目标：最小化欧几里得距离（等价于最小化比例偏差的平方和）
def objective(x):
    # x: 各标的追加金额数组
    变动后市值 = df["现有市值"].values + x
    变动后比例 = 变动后市值 / 变动后总市值
    比例偏差 = 变动后比例 - df["目标比例"].values
    return np.sum(比例偏差**2)  # 平方和（平方根不影响最小化结果）


# ===================== 3. 定义约束条件 =====================
# 约束1：追加金额总和 = 1000（等式约束）
def constraint_sum(x):
    return np.sum(x) - 追加资金总额整数


# 约束2：追加金额 ≥ 0（不等式约束，只买不卖）
constraints = [
    {"type": "eq", "fun": constraint_sum},  # 等式约束
    {"type": "ineq", "fun": lambda x: x},  # 不等式约束（x ≥ 0）
]

# ===================== 4. 求解优化问题 =====================
# 初始猜测：平均分配（仅作为优化起点）
x0 = np.ones(len(df)) * (追加资金总额整数 / len(df))
# 优化求解（SLSQP算法适合带约束的非线性优化）
result = minimize(
    fun=objective,
    x0=x0,
    method="SLSQP",
    constraints=constraints,
    tol=1e-9,  # 精度
    options={"maxiter": 1000},
)


# ===================== 5. 结果整理与验证 =====================
def round_to_int_with_sum(x, total_int):
    floor = np.floor(x).astype(int)
    remainder = int(total_int - floor.sum())
    if remainder > 0:
        frac = x - floor
        idx = np.argsort(-frac)
        floor[idx[:remainder]] += 1
    return floor


def optimize_integer_allocation(x_cont, total_int):
    x_int = round_to_int_with_sum(x_cont, total_int)

    def obj_int(v):
        变动后市值 = df["现有市值"].values + v
        变动后比例 = 变动后市值 / 变动后总市值
        比例偏差 = 变动后比例 - df["目标比例"].values
        return np.sum(比例偏差**2)

    best_val = obj_int(x_int)
    n = len(x_int)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if x_int[i] == 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                y = x_int.copy()
                y[i] -= 1
                y[j] += 1
                val = obj_int(y)
                if val < best_val - 1e-12:
                    x_int = y
                    best_val = val
                    improved = True
    return x_int


df["追加金额"] = optimize_integer_allocation(result.x, 追加资金总额整数)
# 计算变动后数据
df["变动后市值"] = df["现有市值"] + df["追加金额"]
df["变动后比例"] = df["变动后市值"] / 变动后总市值
df["比例偏差"] = df["变动后比例"] - df["目标比例"]
df["比例偏差平方"] = df["比例偏差"] ** 2

# 计算欧几里得距离
欧几里得距离 = np.sqrt(df["比例偏差平方"].sum())

# ===================== 6. 输出结果 =====================
print("===== 基础数据 =====")
print(f"现有总市值：{现有总市值:.2f} 元")
print(f"追加资金（整数）：{追加资金总额整数} 元")
print(f"变动后总市值（整数追加）：{变动后总市值:.2f} 元")

print("\n===== 资金分配结果 =====")


def format_currency_2(x):
    return f"{x:,.2f}"


def format_currency_int(x):
    return f"{int(x):,d}"


def format_percent(x):
    return f"{x * 100:.2f}%"


def format_percent_signed(x):
    return f"{x * 100:+.2f}%"


cols = ["标的", "现有市值", "追加金额", "变动后市值", "目标比例", "变动后比例", "比例偏差"]
formatters = {
    "现有市值": format_currency_2,
    "追加金额": format_currency_int,
    "变动后市值": format_currency_2,
    "目标比例": format_percent,
    "变动后比例": format_percent,
    "比例偏差": format_percent_signed,
}
print(df[cols].to_string(index=False, formatters=formatters))

print("\n===== 关键指标 =====")
print(f"追加金额总和：{df['追加金额'].sum():d} 元（验证：应等于{追加资金总额整数}）")
print(f"欧几里得距离：{欧几里得距离:.6f}")
print(f"比例偏差平方和：{df['比例偏差平方'].sum():.6f}")
