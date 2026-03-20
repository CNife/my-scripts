# personal-scripts 项目知识库

**生成时间**: 2026-03-20 | **Commit**: 29e9831 | **Branch**: main

## 项目结构
```
./
├── fund_portfolio_rebalancing/    # 基金再平衡（Typer CLI）
├── risk_parity_a_share/           # 风险平价（Typer + 测试）
├── organize_onedrive_album/       # 图像整理（数字前缀脚本）
├── arbitrage_calculator/          # 套利计算（交互式）
├── cigar_butt_screener/           # 烟蒂股筛选（PEP 723）
├── gold_analysis/                 # 黄金分析（argparse）
├── skills/                        # Agent 技能模块
│   ├── crawl-xueqiu-my-timeline/  # 雪球首页时间线
│   ├── crawl-xueqiu-user-timeline/# 用户时间线
│   └── check-opencode-updates/    # OpenCode 更新检查
├── pyproject.toml
└── .pre-commit-config.yaml
```

## 目录定位
| 任务 | 位置 |
|------|------|
| 基金再平衡 | fund_portfolio_rebalancing/ |
| 风险平价 | risk_parity_a_share/ |
| 图像整理 | organize_onedrive_album/ |
| 套利计算 | arbitrage_calculator/ |
| 烟蒂股筛选 | cigar_butt_screener/ |
| 黄金分析 | gold_analysis/ |
| 雪球爬取 | skills/crawl-xueqiu-*/ |

## 独特约定
### CLI 框架
| 类型 | 框架 | 示例 |
|------|------|------|
| 复杂 CLI | Typer (typer.Typer) | rebalance_portfolio.py |
| 简单 CLI | Typer (typer.run) | organize_onedrive_album/*.py |
| 参数化 | argparse | gold_analysis.py |
| 交互式 | 无框架 | arbitrage_calculator.py |

### 文件命名
- `organize_onedrive_album/` 数字前缀表示处理顺序（1_, 2_1_, 3_, 4_）
- `skills/` 包含 SKILL.md + scripts/ 子目录
- `cigar_butt_screener/screener.py` 使用 PEP 723 单脚本依赖声明

## 代码质量
- **Ruff**: 行长度 100，目标 py314，启用 B/C4/I/PIE/RUF/S/SIM/UP/W
- **Pre-commit**: ruff-format → ruff-check --fix → uv-lock

## 命令参考
```bash
# 运行
uv run python fund_portfolio_rebalancing/rebalance_portfolio.py --help
uv run python risk_parity_a_share/risk_parity_calculator.py --help
uv run python arbitrage_calculator/arbitrage_calculator.py

# 测试
uv run pytest risk_parity_a_share/test_risk_parity.py -v

# 代码质量
ruff format --line-length 100 <file>
ruff check --fix <file>
```

## 待改进
1. organize_onedrive_album 重构为单一 Typer 应用
2. 添加完整测试套件
3. 优化基金净值获取错误处理
