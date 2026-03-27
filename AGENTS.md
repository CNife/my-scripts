# personal-scripts 项目

**类型**: Python 脚本项目，使用 uv 管理依赖
**Python 版本**: 3.14
**用途**: 个人实用脚本集合（图像处理、投资组合再平衡等）

## 项目特定规则

### Ruff 配置
- **目标版本**: py314
- **行长度**: 100 字符
- **启用规则**: B, C4, I, PIE, RUF, S, SIM, UP, W
- **忽略规则**: RUF001/002/003（Unicode 字符）, S607（shell 启动）, S101（断言）

### 预提交钩子
1. 基础检查：trailing-whitespace, end-of-file-fixer, mixed-line-ending
2. 文件验证：check-json, check-toml
3. 依赖管理：uv-lock
4. 代码格式化：ruff-format
5. 代码检查：ruff-check --fix

## 模块规则
- 基金投资组合再平衡 → 参考 `fund_portfolio_rebalancing/AGENTS.md`
| 风险平价 → 参考 `risk_parity_a_share/AGENTS.md`
| 图像整理 → 参考 `organize_onedrive_album/AGENTS.md`
| 套利计算 → 参考 `arbitrage_calculator/AGENTS.md`

## 文档
- 项目结构、代码地图、项目特点 → 参考 `docs/AGENTS.md`