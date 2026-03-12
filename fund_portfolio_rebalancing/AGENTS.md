# 基金投资组合再平衡工具

**目录**: fund_portfolio_rebalancing/
**功能**: 使用非线性优化算法实现基金投资组合的再平衡和风险平价计算
**技术栈**: Python + Typer + SciPy + Rich + Tushare + pytest

## 概述

该工具包含两个主要功能模块：
1. **投资组合再平衡**: 使用 Typer 构建命令行界面，通过 SciPy 的优化算法（SLSQP）计算最佳资金分配方案，实现投资组合的再平衡。支持三种交易模式：允许买卖、只买不卖、只卖不买。
2. **风险平价计算器**: 计算 ETF 组合的风险平价权重，使每个资产对组合总风险的贡献相等，实现真正的风险分散化配置。

## 文件结构
```
fund_portfolio_rebalancing/
├── __init__.py                  # 包初始化文件
├── rebalance_portfolio.py       # 投资组合再平衡主脚本
├── risk_parity_calculator.py    # 风险平价计算器
├── data_fetcher.py              # Tushare 数据获取模块
├── test_risk_parity.py          # 风险平价测试套件
├── jiufeite.json                # 基金配置：酒菲特组合
├── personal_pension.json        # 基金配置：个人养老金
├── wide.json                    # 基金配置：宽基组合
└── risk_parity_config.json      # 风险平价默认 ETF 配置
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
| RiskParityResult | risk_parity_calculator.py:41-52 | 风险平价计算结果 |
| DataQualityReport | data_fetcher.py:35-44 | 数据质量检查报告 |

### 核心函数 - 投资组合再平衡
| 函数 | 位置 | 作用 |
|------|------|------|
| main | rebalance_portfolio.py:62-156 | 主入口函数，处理命令行参数 |
| load_portfolio_config | rebalance_portfolio.py:159-183 | 从 JSON 文件加载配置 |
| prompt_current_values | rebalance_portfolio.py:186-202 | 交互式询问当前市值 |
| compute_optimal_allocation | rebalance_portfolio.py:205-306 | 计算最优资金分配 |
| optimize_integer_allocation | rebalance_portfolio.py:379-421 | 整数分配优化 |
| get_fund_latest_nav | rebalance_portfolio.py:424-449 | 获取基金最新净值 |
| print_allocation_result | rebalance_portfolio.py:309-376 | 显示分配结果 |

### 核心函数 - 风险平价
| 函数 | 位置 | 作用 |
|------|------|------|
| calculate_risk_parity_weights | risk_parity_calculator.py:168-249 | 计算风险平价权重 |
| calculate_risk_contribution | risk_parity_calculator.py:252-282 | 计算各资产风险贡献 |
| calculate_balance_score | risk_parity_calculator.py:347-361 | 计算风险贡献均衡度 |
| load_etf_config | risk_parity_calculator.py:139-165 | 加载 ETF 配置 |
| print_results | risk_parity_calculator.py:285-344 | 显示风险平价结果 |

### 核心类 - 数据获取
| 类名 | 位置 | 作用 |
|------|------|------|
| RiskParityData | data_fetcher.py:47-478 | ETF 数据获取与缓存管理 |
| DataFetcherError | data_fetcher.py:23-25 | 数据获取异常基类 |
| TushareAPIError | data_fetcher.py:27-28 | Tushare API 异常 |
| DataQualityError | data_fetcher.py:31-32 | 数据质量异常 |

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

### 投资组合再平衡

#### 1. 追加投资
```bash
uv run python rebalance_portfolio.py --config jiufeite.json --additional-cash 10000
```

#### 2. 目标市值
```bash
uv run python rebalance_portfolio.py --config jiufeite.json --target-total 100000
```

#### 3. 只买模式
```bash
uv run python rebalance_portfolio.py --config jiufeite.json --additional-cash 5000 --mode buy
```

### 风险平价计算

#### 1. 使用默认配置
```bash
uv run python risk_parity_calculator.py
```

#### 2. 指定日期范围
```bash
uv run python risk_parity_calculator.py --start-date 20220101 --end-date 20231231
```

#### 3. 离线模式（仅使用缓存）
```bash
uv run python risk_parity_calculator.py --offline
```

#### 4. 自定义 ETF 配置
```bash
uv run python risk_parity_calculator.py --config my_etfs.json
```

## 测试

### 运行测试
```bash
# 运行所有风险平价测试
uv run pytest fund_portfolio_rebalancing/test_risk_parity.py -v

# 运行特定测试类
uv run pytest fund_portfolio_rebalancing/test_risk_parity.py::TestCalculateRiskParityWeights -v

# 运行特定测试方法
uv run pytest fund_portfolio_rebalancing/test_risk_parity.py::TestRiskContributionEquality::test_risk_contributions_approximately_equal -v
```

### 测试覆盖
- **权重计算测试**: 验证权重和为 1，高波动资产权重更低
- **风险贡献测试**: 验证风险贡献相等（1/n），风险贡献和为 1
- **数据获取测试**: 验证模块导入，离线模式初始化
- **合成数据测试**: 使用合成协方差矩阵测试 2/5/10 资产组合

## 注意事项

1. **Tushare API**: 获取基金净值需要 tushare 库和 API 密钥
2. **网络依赖**: 基金净值查询需要网络连接
3. **精度**: 使用整数分配以确保实际操作的可行性
4. **测试**: 测试使用合成协方差矩阵，不依赖外部 API
