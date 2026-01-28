# 基金投资组合再平衡工具

**目录**: fund_portfolio_rebalancing/
**功能**: 使用非线性优化算法实现基金投资组合的再平衡
**技术栈**: Python + Typer + SciPy + Rich + Tushare

## 概述

该工具使用 Typer 构建命令行界面，通过 SciPy 的优化算法（SLSQP）计算最佳资金分配方案，实现投资组合的再平衡。支持三种交易模式：允许买卖、只买不卖、只卖不买。

## 文件结构
```
fund_portfolio_rebalancing/
├── rebalance_portfolio.py       # 主脚本和核心逻辑
├── jiufeite.json                # 基金配置：酒菲特组合
├── personal_pension.json        # 基金配置：个人养老金
└── wide.json                    # 基金配置：宽基组合
```

## 核心功能

### 主要命令
```bash
uv run python rebalance_portfolio.py --help
uv run python rebalance_portfolio.py --config jiufeite.json --additional-cash 10000 --mode all
```

### 关键参数
| 参数 | 说明 |
|------|------|
| --config, -c | 投资组合配置 JSON 文件路径 |
| --target-total, -t | 目标投资组合总市值 |
| --additional-cash, -a | 追加投资的总金额（可负数） |
| --current-values, -v | 各资产当前市值 |
| --mode, -m | 交易模式：all/buy/sell |

## 代码地图

### 数据类
| 类名 | 位置 | 作用 |
|------|------|------|
| AllocationParams | rebalance_portfolio.py:17-26 | 投资组合分配参数 |
| AllocationResult | rebalance_portfolio.py:29-48 | 投资组合分配结果 |

### 核心函数
| 函数 | 位置 | 作用 |
|------|------|------|
| main | rebalance_portfolio.py:62-156 | 主入口函数，处理命令行参数 |
| load_portfolio_config | rebalance_portfolio.py:159-183 | 从 JSON 文件加载配置 |
| prompt_current_values | rebalance_portfolio.py:186-202 | 交互式询问当前市值 |
| compute_optimal_allocation | rebalance_portfolio.py:205-306 | 计算最优资金分配 |
| optimize_integer_allocation | rebalance_portfolio.py:379-421 | 整数分配优化 |
| get_fund_latest_nav | rebalance_portfolio.py:424-449 | 获取基金最新净值 |
| print_allocation_result | rebalance_portfolio.py:309-376 | 显示分配结果 |

### 配置文件格式
```json
{
  "assets": [
    {
      "name": "资产名称",
      "code": "基金代码",
      "target_weight": 0.1
    }
  ]
}
```

## 算法原理

### 优化目标
- **目标函数**: 最小化权重偏差平方和
- **约束条件**: 资金总和约束、交易模式约束
- **方法**: 序列二次规划 (SLSQP)

### 整数调整
- 连续解四舍五入
- 邻域搜索优化整数解
- 保持总金额不变

## 使用场景

### 1. 追加投资
```bash
uv run python rebalance_portfolio.py --config jiufeite.json --additional-cash 10000
```

### 2. 目标市值
```bash
uv run python rebalance_portfolio.py --config jiufeite.json --target-total 100000
```

### 3. 只买模式
```bash
uv run python rebalance_portfolio.py --config jiufeite.json --additional-cash 5000 --mode buy
```

## 注意事项

1. **Tushare API**: 获取基金净值需要 tushare 库和 API 密钥
2. **网络依赖**: 基金净值查询需要网络连接
3. **精度**: 使用整数分配以确保实际操作的可行性
