# A股ETF风险平价计算器

**技术栈**: Python + Typer + SciPy + Tushare + pytest

## 文件结构
```
risk_parity_a_share/
├── risk_parity_calculator.py    # 主入口
├── data_fetcher.py              # 数据获取与缓存
├── test_risk_parity.py          # 测试套件
└── risk_parity_config.json      # 默认ETF配置
```

## CLI 命令
```bash
uv run python risk_parity_a_share/risk_parity_calculator.py
uv run python risk_parity_a_share/risk_parity_calculator.py --start-date 20220101 --end-date 20231231
uv run python risk_parity_a_share/risk_parity_calculator.py --offline  # 仅使用缓存
```

## 核心符号
| 符号 | 位置 | 作用 |
|------|------|------|
| RiskParityResult | risk_parity_calculator.py | 计算结果数据类 |
| RiskParityData | data_fetcher.py | ETF 数据获取与缓存 |
| calculate_risk_parity_weights | risk_parity_calculator.py | 计算风险平价权重 |
| calculate_risk_contribution | risk_parity_calculator.py | 计算风险贡献 |

## 测试
```bash
uv run pytest risk_parity_a_share/test_risk_parity.py -v
uv run pytest risk_parity_a_share/test_risk_parity.py -m "not network"  # 跳过网络测试
```

## 注意事项
- 需要 Tushare API 密钥（积分≥300 获取指数数据）
- 数据自动缓存，支持离线运行
- 多 ETF 数据自动对齐到共同交易日
