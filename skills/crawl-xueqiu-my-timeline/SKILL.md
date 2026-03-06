---
name: crawl-xueqiu-my-timeline
description: 爬取雪球首页关注的时间线，生成投资分析报告和 PDF。当用户提及雪球时间线、投资要闻、市场热点、大 V 观点、投资分析、生成 PDF 报告时立即使用此技能
---

# crawl-xueqiu-my-timeline

爬取雪球首页关注的时间线，保存为按发言人分组的 Markdown 文件，供 AI 分析总结并生成 PDF 投资报告。

**特点**：
- 自动过滤官方账号（上市公司、指数、ETF 等行情播报）
- 按发言人分组输出，发言数量降序排列
- 组内按时间倒序，越新的内容越靠前

## 快速开始

```bash
# 1. 检查环境
sh skills/crawl-xueqiu-my-timeline/scripts/check-cdp.sh
sh skills/crawl-xueqiu-my-timeline/scripts/check-agent-browser.sh

# 2. 爬取最近 24 小时时间线
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py

# 3. AI 自动读取、分析、生成 PDF 报告
```

## 何时使用

✅ **应该触发**：
- "帮我看看雪球上今天有什么投资要闻"
- "生成一份市场热点分析报告"
- "把关注的雪球大 V 观点整理成 PDF"
- "最近有什么投资热点"
- "爬取雪球时间线"
- "生成投资分析报告"

❌ **不应触发**：
- "打开雪球网站"（只需浏览器操作）
- "查看某只股票行情"（查询单一数据）
- "帮我登录雪球"（简单网页操作）

## 前置准备

确保 Chrome 处于 Debug 模式并安装 agent-browser：

```bash
sh skills/crawl-xueqiu-my-timeline/scripts/check-cdp.sh
sh skills/crawl-xueqiu-my-timeline/scripts/check-agent-browser.sh
```

## AI 任务清单

1. ✅ **检查环境**：Chrome Debug 模式 + agent-browser
2. ✅ **执行爬取**：运行 `crawl_xueqiu_home_timeline_api.py`
3. ✅ **读取文件**：读取生成的 `home_timeline_YYYYMMDD_YYYYMMDD.md`
4. ✅ **分析总结**：逐一总结所有发言人观点（不得遗漏）
5. ✅ **生成报告**：写入 Markdown 格式投资分析报告
6. ✅ **转换 PDF**：使用 mdpdf 转换为 PDF
7. ✅ **清理文件**：删除临时 Markdown 文件

---

## AI 分析指引

### ⚠️ 重要：观点归属判断

**只有不以 `//` 或 `>` 开头的内容才是发言人自己的观点**：

| 标记 | 含义 | 是否总结 |
|------|------|---------|
| 无标记行 | 发言人本人的发言 | ✅ 需总结 |
| `//` 开头 | 被评论的其他用户发言 | ❌ 非本人观点 |
| `>` 开头 | 被引用的转发内容 | ❌ 非本人观点 |

**注释**：`回复@XXX：` 前缀是雪球自动生成的，整行都是发言人自己的观点，应总结。

**示例**：
```
回复@阿企笔记：要知道很多农村的人需要方言语音设备
//回复@青年完：支付宝本来就有音响
> @阿企笔记：中概股失去增长动力
```

**正确总结**：发言人认为农村用户需要方言语音设备

**错误总结**：发言人讨论了支付宝音响、中概股财报

### 读取输出文件

爬取脚本会在当前目录生成 Markdown 文件：`home_timeline_YYYYMMDD_YYYYMMDD.md`

```bash
# 读取最新生成的时间线文件
cat home_timeline_*.md
```

### 分析报告格式

AI 应按以下结构生成投资分析报告：

```markdown
# 雪球关注时间线分析报告

## 时间范围
2026-03-05 至 2026-03-06

## 发言人统计
- @产业驱动博弈：19 条
- @管我财：6 条
...（逐一列出所有发言人）

## 发言观点总结

### @发言人 A（发言最多的用户）
- **发言数量**: X 条
- **主要观点**:
  - 观点 1 摘要
  - 观点 2 摘要
- **涉及标的**: $股票 1$, $股票 2$
- **情绪倾向**: 乐观/中性/悲观

### @发言人 B
- **发言数量**: X 条
- **主要观点**:
  - 观点摘要
- **涉及标的**: $股票 3$
- **情绪倾向**: 乐观/中性/悲观

...（继续列出所有发言人，直到最后）...

## 市场热点 TOP3
1. 热点主题 1 - 讨论人数 X
2. 热点主题 2 - 讨论人数 X
3. 热点主题 3 - 讨论人数 X

## 投资建议/风险提示
（基于发言内容的综合分析）
```

**核心原则**：
- ✅ **必须包含所有发言人**，即使只有 1 条发言
- ✅ 按发言数量降序排列
- ✅ 只总结本人观点（非 `//` 或 `>` 开头的内容）

### 分析维度

1. **按发言人分组**：汇总同一用户的所有发言
2. **观点提取**：识别核心观点，去重合并相似内容
3. **标的识别**：提取 `$股票名$` 格式的代码
4. **情绪分析**：根据用词判断市场情绪
5. **热点聚合**：统计跨用户讨论的话题

---

## 爬取脚本参数

```bash
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hours` | 爬取最近 N 小时 | 24 |
| `--days` | 爬取最近 N 天 | - |
| `--start-date` | 开始日期 (YYYY-MM-DD) | - |
| `--end-date` | 结束日期 (YYYY-MM-DD) | 今天 |
| `-o, --output` | 输出文件名 | 自动生成 |

**注意**：`--hours`、`--days`、`--start-date` 三个参数互斥。

### 示例

```bash
# 爬取最近 24 小时（默认）
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py

# 爬取最近 2 小时
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py --hours 2

# 爬取最近 7 天
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py --days 7

# 指定日期范围
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py --start-date 2026-03-01 --end-date 2026-03-06
```

---

## PDF 输出流程

### 1. 保存分析报告

将分析报告保存为 `雪球时间线_YYYYMMDD_YYYYMMDD.md`，日期为时间范围的起始和结束日期。

示例：`雪球时间线_20260305_20260306.md`

### 2. 转换为 PDF

```bash
bunx mdpdf 雪球时间线_YYYYMMDD_YYYYMMDD.md \
  --style=/home/cnife/code/my-scripts/skills/crawl-xueqiu-my-timeline/assets/github-markdown.css
```

例如：
```bash
bunx mdpdf 雪球时间线_20260305_20260306.md \
  --style=/home/cnife/code/my-scripts/skills/crawl-xueqiu-my-timeline/assets/github-markdown.css
```

**可选参数**：
- `--border=<size>`：页边距（默认 20mm），如 `--border=15mm`
- `--format=<format>`：纸张大小（默认 A4），如 `--format=A4`
- `--orientation=<orientation>`：方向（默认 portrait）
- `--title=<title>`：PDF 标题

### 3. 清理临时文件

```bash
rm 雪球时间线_YYYYMMDD_YYYYMMDD.md
```

### 完整示例

```bash
# AI 写入总结文件 → 雪球时间线_20260305_20260306.md

# 转换为 PDF（当前目录生成 PDF）
bunx mdpdf 雪球时间线_20260305_20260306.md \
  --style=/home/cnife/code/my-scripts/skills/crawl-xueqiu-my-timeline/assets/github-markdown.css

# 删除临时 Markdown 文件
rm 雪球时间线_20260305_20260306.md
```

**结果**：当前工作目录生成 `雪球时间线_20260305_20260306.pdf`

---

## 输出格式

爬取脚本生成的 `home_timeline_*.md` 包含：
- 文件头：时间范围、生成时间、发言统计表格
- 按发言人分组：发言数量降序排列
- 每组内时间倒序：越新的发言越靠前
- 每条发言：发布时间、内容、引用内容、原文链接
- 自动过滤：官方账号行情播报

## 注意事项

1. 需要先登录雪球账号
2. 如遇 Verification 验证页面，需手动完成验证后重新运行
3. 爬取过程自动处理分页和 md5__1038 令牌
4. 输出文件保存在当前工作目录

## 典型使用场景

1. **每日投资要闻回顾**：爬取最近 24 小时，AI 总结市场热点并生成 PDF 报告
2. **关注大 V 动态**：爬取最近 N 天，AI 分析投资观点变化并生成 PDF 报告
3. **特定事件追踪**：指定日期范围，AI 整理事件期间的讨论并生成 PDF 报告

## 资源文件

- `assets/github-markdown.css`: PDF 样式文件（mdpdf 使用）
- `scripts/check-cdp.sh`: 检查 Chrome Debug 模式
- `scripts/check-agent-browser.sh`: 检查 agent-browser 安装
- `scripts/crawl_xueqiu_home_timeline_api.py`: 爬取脚本（主入口）
- `evals/evals.json`: 测试用例集（用于验证技能功能）
