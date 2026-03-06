---
name: crawl-xueqiu-my-timeline
description: 爬取雪球首页关注的时间线，供 AI 分析总结
---

# crawl-xueqiu-my-timeline

爬取雪球首页关注的时间线，保存为按发言人分组的 Markdown 文件，供 AI 分析总结。

**特点**：
- 自动过滤官方账号（上市公司、指数、ETF 等行情播报）
- 按发言人分组输出，发言数量降序排列
- 组内按时间倒序，越新的内容越靠前

## 前置准备

Base directory for this skill: `{base_dir}`

**调用时请将 `{base_dir}` 替换为实际路径**，例如：`/home/cnife/code/my-scripts/skills/crawl-xueqiu-my-timeline`

确保 Chrome 处于 Debug 模式并安装 agent-browser：

```bash
sh {base_dir}/scripts/check-cdp.sh
sh {base_dir}/scripts/check-agent-browser.sh
```

## 使用方法

### 完整流程

```bash
# 1. 检查环境
sh {base_dir}/scripts/check-cdp.sh
sh {base_dir}/scripts/check-agent-browser.sh

# 2. 爬取首页时间线（默认最近 24 小时）
{base_dir}/scripts/crawl_xueqiu_home_timeline_api.py

# 3. AI 读取生成的 .md 文件并总结
```

### 爬取脚本参数

直接运行爬取脚本：

```bash
{base_dir}/scripts/crawl_xueqiu_home_timeline_api.py [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hours` | 爬取最近 N 小时 | 24 |
| `--days` | 爬取最近 N 天 | - |
| `--start-date` | 开始日期 (YYYY-MM-DD) | - |
| `--end-date` | 结束日期 (YYYY-MM-DD) | 今天 |
| `-o, --output` | 输出文件名 | 自动生成 |

**注意**：
- 不传参数默认爬取最近 24 小时
- `--hours`、`--days`、`--start-date` 三个参数互斥，不能同时使用

### 示例

```bash
# 爬取最近 24 小时（默认）
{base_dir}/scripts/crawl_xueqiu_home_timeline_api.py

# 爬取最近 2 小时
{base_dir}/scripts/crawl_xueqiu_home_timeline_api.py --hours 2

# 爬取最近 7 天
{base_dir}/scripts/crawl_xueqiu_home_timeline_api.py --days 7

# 指定日期范围
{base_dir}/scripts/crawl_xueqiu_home_timeline_api.py --start-date 2026-03-01 --end-date 2026-03-06

# 指定输出文件名
{base_dir}/scripts/crawl_xueqiu_home_timeline_api.py -o my_timeline.md
```

## 输出格式

生成 Markdown 文件，包含：
- 文件头：时间范围、生成时间、发言统计表格
- **按发言人分组**：同一发言人的所有发言归为一组，按发言数量降序排列
- 每组内按时间倒序：越新的发言越靠前
- 每条发言包含：发布时间、内容、引用内容（如有）、原文链接
- **自动过滤**：上市公司、指数、ETF 等官方账号的行情播报内容

## 注意事项

1. 需要先登录雪球账号
2. 如遇 Verification 验证页面，需手动完成验证后重新运行
3. 爬取过程中会自动处理分页和 md5__1038 令牌
4. 输出文件保存在当前工作目录
5. 爬取完成后，由调用此 skill 的 AI 读取 .md 文件并进行分析总结

## 典型使用场景

1. **每日投资要闻回顾**: 爬取最近 24 小时，AI 总结市场热点
2. **关注大动态**: 爬取最近 N 天，AI 分析关注列表的投资观点变化
3. **特定事件追踪**: 指定日期范围，AI 整理某事件期间的讨论

---

**技能流程**：
1. 检查 Chrome Debug 模式和 agent-browser 环境
2. 爬取雪球首页关注时间线（API 方式，自动处理反爬）
3. 过滤官方账号（上市公司、指数、ETF 等）
4. 保存为按发言人分组的 Markdown 文件
5. AI 读取文件内容，**必须包含所有发言人**，逐一总结观点并生成投资分析报告

## AI 分析指引

### 读取输出文件

爬取脚本会在当前目录生成 Markdown 文件，命名格式：`home_timeline_YYYYMMDD_YYYYMMDD.md`

```bash
# 读取最新生成的时间线文件
cat home_timeline_*.md
```

### 按发言人汇总的格式建议

AI 应该按以下结构分析和总结：

```markdown
# 雪球关注时间线分析报告

## 时间范围
2026-03-05 至 2026-03-06

## 发言人统计
（根据 Markdown 文件头的统计表格，逐一列出所有发言人）

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
  - 观点 1 摘要
  - 观点 2 摘要
- **涉及标的**: $股票 3$
- **情绪倾向**: 乐观/中性/悲观

### @发言人 C（发言较少的用户也要包括）
- **发言数量**: X 条
- **主要观点**:
  - 观点摘要
- **涉及标的**: $股票 4$
- **情绪倾向**: 乐观/中性/悲观

...（继续列出所有发言人，直到最后）...

## 市场热点 TOP3
1. 热点主题 1 - 讨论人数 X
2. 热点主题 2 - 讨论人数 X
3. 热点主题 3 - 讨论人数 X

## 投资建议/风险提示
（基于发言内容的综合分析）
```

**重要提醒**:
- ✅ **必须包含所有发言人**，无论发言数量多少（即使只有 1 条发言也要总结）
- ✅ 按发言数量降序排列，发言越多的用户越靠前

### 分析维度

1. **按发言人分组**: 提取 Markdown 中的 `@用户名`，汇总同一用户的所有发言
2. **必须包含所有发言人**: 无论发言数量多少，每位发言人的观点都必须被总结，不得遗漏
3. **观点提取**: 识别每条发言的核心观点，去重合并相似内容
4. **标的识别**: 提取 `$股票名$` 格式的股票/基金代码
5. **情绪分析**: 根据用词判断发言人的市场情绪
6. **热点聚合**: 统计跨用户讨论的话题，识别市场关注焦点
