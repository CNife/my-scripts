# 基金投资组合再平衡工具

**技术栈**: Python + Typer + SciPy + Rich + Tushare

## 文件结构
```
fund_portfolio_rebalancing/
├── rebalance_portfolio.py       # 投资组合再平衡
├── risk_parity_calculator.py    # 风险平价计算
├── data_fetcher.py              # Tushare 数据获取
├── test_risk_parity.py          # 测试套件
├── jiufeite.json                # 酒菲特组合配置
├── personal_pension.json        # 个人养老金配置
└── wide.json                    # 宽基组合配置
```

## CLI 命令
```bash
# 投资组合再平衡
uv run python rebalance_portfolio.py --config jiufeite.json --additional-cash 10000
uv run python rebalance_portfolio.py --config jiufeite.json --target-total 100000
uv run python rebalance_portfolio.py --config jiufeite.json --additional-cash 5000 --mode buy

# 风险平价
uv run python risk_parity_calculator.py
uv run python risk_parity_calculator.py --offline
```

## 核心符号
| 符号 | 位置 | 作用 |
|------|------|------|
| AllocationParams | rebalance_portfolio.py | 分配参数数据类 |
| AllocationResult | rebalance_portfolio.py | 分配结果数据类 |
| RiskParityData | data_fetcher.py | 数据获取与缓存 |
| compute_optimal_allocation | rebalance_portfolio.py | 最优资金分配 |

## 配置格式
```json
{
  "assets": [
    {"name": "资产名称", "code": "基金代码", "target_weight": 0.1}
  ]
}
```

## 测试
```bash
uv run pytest fund_portfolio_rebalancing/test_risk_parity.py -v
```

## 注意事项
- 需要 Tushare API 密钥
- 使用 SLSQP 优化 + 整数调整
- 测试使用合成数据，不依赖外部 API
