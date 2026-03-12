# 雪球首页时间线爬取技能

**目录**: skills/crawl-xueqiu-my-timeline/
**功能**: 爬取雪球首页关注的时间线，生成投资分析报告和 PDF
**技术栈**: Python + Playwright

## 概述

Agent 技能模块，用于爬取雪球首页关注列表的时间线数据，按发言人分组输出，支持 AI 分析总结并生成 PDF 投资报告。

## 文件结构
```
skills/crawl-xueqiu-my-timeline/
├── SKILL.md                      # 技能定义文档（主文档）
├── scripts/
│   ├── crawl_xueqiu_home_timeline_api.py  # 爬取脚本（主入口）
│   ├── check-cdp.sh              # 检查 Chrome Debug 模式
│   └── check-agent-browser.sh    # 检查 agent-browser 安装
├── assets/
│   └── github-markdown.css       # PDF 样式文件
└── evals/
    └── evals.json                # 测试用例集
```

## 核心功能

### 爬取脚本参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hours` | 爬取最近 N 小时 | 24 |
| `--days` | 爬取最近 N 天 | - |
| `--start-date` | 开始日期 (YYYY-MM-DD) | - |
| `--end-date` | 结束日期 (YYYY-MM-DD) | 今天 |
| `-o, --output` | 输出文件名 | 自动生成 |

### 特点
- 自动过滤官方账号（上市公司、指数、ETF 等行情播报）
- 按发言人分组输出，发言数量降序排列
- 组内按时间倒序，越新的内容越靠前

## 使用方式

### 前置检查
```bash
sh skills/crawl-xueqiu-my-timeline/scripts/check-cdp.sh
sh skills/crawl-xueqiu-my-timeline/scripts/check-agent-browser.sh
```

### 爬取时间线
```bash
# 爬取最近 24 小时（默认）
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py

# 爬取最近 2 小时
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py --hours 2

# 指定日期范围
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py --start-date 2026-03-01 --end-date 2026-03-06
```

### AI 分析流程
1. 读取生成的 `home_timeline_YYYYMMDD_YYYYMMDD.md`
2. 按发言人数量创建分析 TODO 列表
3. 启动 subagent 分析用户发言（并发 ≤3）
4. 整合分析结果，生成投资分析报告
5. 转换为 PDF（使用 mdpdf）

## 输出格式

### Markdown 文件结构
- 文件头：时间范围、生成时间、发言统计表格
- 按发言人分组：发言数量降序排列
- 每组内时间倒序：越新的发言越靠前
- 每条发言：发布时间、内容、引用内容、原文链接

### 观点归属判断
| 标记 | 含义 | 是否总结 |
|------|------|----------|
| 无标记行 | 发言人本人的发言 | ✅ 需总结 |
| `//` 开头 | 被评论的其他用户发言 | ❌ 非本人观点 |
| `>` 开头 | 被引用的转发内容 | ❌ 非本人观点 |

## 注意事项

1. 需要先登录雪球账号
2. 如遇 Verification 验证页面，需手动完成验证后重新运行
3. 爬取过程自动处理分页和 md5__1038 令牌
4. 输出文件保存在当前工作目录
5. Subagent 并发数 ≤3，失败自动重试 2 次

## 触发条件

✅ **应该触发**:
- "帮我看看雪球上今天有什么投资要闻"
- "生成一份市场热点分析报告"
- "把关注的雪球大 V 观点整理成 PDF"
- "爬取雪球时间线"

❌ **不应触发**:
- "打开雪球网站"（只需浏览器操作）
- "查看某只股票行情"（查询单一数据）
