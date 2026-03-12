# personal-scripts 项目知识库

**生成时间**: 2026-03-12
**项目类型**: Python 脚本项目，使用 uv 管理依赖
**Python 版本**: 3.12+
**主要用途**: 个人实用脚本集合（图像处理、投资组合再平衡、套利计算、雪球数据爬取等）

## 项目结构
```
./
├── fund_portfolio_rebalancing/    # 基金投资组合再平衡工具
├── organize_onedrive_album/       # OneDrive 相册整理工具
├── arbitrage_calculator/          # 中金换股套利计算器
├── skills/                        # Agent 技能模块
│   ├── crawl-xueqiu-my-timeline/  # 爬取雪球首页时间线
│   └── crawl-xueqiu-user-timeline/# 爬取指定用户时间线
├── pyproject.toml                # 项目配置和依赖管理
├── .pre-commit-config.yaml        # 代码质量检查配置
└── AGENTS.md                      # 项目知识和开发指导
```

## 目录定位
| 任务类型 | 位置 | 说明 |
|---------|------|------|
| 基金投资组合再平衡 | fund_portfolio_rebalancing/ | 使用 Typer CLI 的优化算法实现 |
| 图像整理和重命名 | organize_onedrive_album/ | 多脚本 pipeline 处理图像 EXIF 和重命名 |
| 套利计算器 | arbitrage_calculator/ | 中金换股吸收合并套利计算 |
| 雪球时间线爬取 | skills/crawl-xueqiu-my-timeline/ | Agent 技能：爬取关注列表时间线 |
| 雪球用户爬取 | skills/crawl-xueqiu-user-timeline/ | Agent 技能：爬取指定用户发言 |
| 项目配置 | pyproject.toml | 依赖管理和 lint 配置 |

## 代码地图
| 符号 | 类型 | 位置 | 作用 |
|------|------|------|------|
| AllocationParams | 数据类 | rebalance_portfolio.py | 投资组合分配参数 |
| AllocationResult | 数据类 | rebalance_portfolio.py | 投资组合分配结果 |
| calculate_arbitrage | 函数 | arbitrage_calculator.py | 套利空间计算主函数 |
| list_images | 函数 | 2_1_mark_time_by_exif.py | 递归列出图像文件 |
| process_image | 函数 | 2_1_mark_time_by_exif.py | 处理单个图像的 EXIF 数据 |
| move_file_safely | 函数 | 4_rename_images.py | 安全移动文件 |

## 独特约定
### 文件命名
- `organize_onedrive_album/` 使用数字前缀表示处理顺序（1_create_database.py、2_1_mark_time_by_exif.py 等）
- 主脚本通常命名为 `[功能].py`（rebalance_portfolio.py）
- `skills/` 目录下的模块包含 SKILL.md 和 scripts/ 子目录

### 依赖管理
使用 uv 作为包管理器，配置见 pyproject.toml：
- **主要依赖**: exifread, typer, rich, scipy, pandas, peewee, openpyxl, tqdm, pillow, pillow-heif, tushare
- **开发依赖**: ipython, pre-commit

## 代码质量规则
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

## 命令参考
```bash
# 依赖管理
uv add <package>          # 添加依赖
uv remove <package>       # 移除依赖
uv sync                   # 同步依赖

# 代码质量
ruff format --line-length 100 <file/dir>   # 格式化代码
ruff check --fix <file/dir>                # 检查并修复代码
pre-commit run --all-files                 # 运行所有预提交检查

# 运行脚本
uv run python fund_portfolio_rebalancing/rebalance_portfolio.py --help
uv run python organize_onedrive_album/1_create_database.py --help
uv run python arbitrage_calculator/arbitrage_calculator.py

# 测试（暂无测试文件）
uv run pytest <test_file>  # 运行单个测试文件
uv run pytest -k <test_name>  # 运行特定测试
```

## 项目特点
1. **基金再平衡工具**: 使用非线性优化（scipy.optimize）实现投资组合优化
2. **图像整理工具**: 多步骤 pipeline 处理 EXIF 数据、重复检测和文件重命名
3. **套利计算器**: 针对中金公司吸收合并东兴/信达证券的换股套利计算
4. **雪球数据爬取**: Agent 技能模块，支持首页关注时间线和指定用户发言爬取
5. **CLI 设计**: 主要脚本使用 Typer 构建命令行界面
6. **数据库**: organize_onedrive_album 使用 SQLite 存储图像处理状态
7. **进度显示**: 使用 tqdm 显示处理进度
8. **Rich 输出**: 使用 Rich 库美化终端输出（表格、颜色等）

## 待改进建议
1. 将 organize_onedrive_album 的多个脚本重构为单一入口的 Typer 应用
2. 为工具添加完整的测试套件
3. 为 organize_onedrive_album 添加配置文件支持
4. 优化基金净值获取的错误处理
5. 添加更详细的文档字符串
