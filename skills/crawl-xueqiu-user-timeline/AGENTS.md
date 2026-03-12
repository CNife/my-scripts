# 雪球用户时间线爬取技能

**目录**: skills/crawl-xueqiu-user-timeline/
**功能**: 抓取雪球用户的发言时间线，保存为 Markdown 文件
**技术栈**: Python + Playwright

## 概述

Agent 技能模块，用于抓取指定雪球用户的发言时间线，保存为 Markdown 文件供后续分析。

## 文件结构
```
skills/crawl-xueqiu-user-timeline/
├── SKILL.md                      # 技能定义文档（主文档）
└── scripts/
    └── crawl_xueqiu_user_timeline_api.py  # 爬取脚本
```

## 核心功能

### 爬取脚本参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `url` | 雪球用户主页链接 | 必填 |
| `--days` | 爬取最近 N 天 | 3 |
| `--start-date` | 开始日期 (YYYY-MM-DD) | 3 天前 |
| `--end-date` | 结束日期 (YYYY-MM-DD) | 今天 |
| `-o, --output` | 输出文件名 | 自动生成 |

**注意**: `--days` 和 `--start-date` 参数互斥，不能同时使用。

## 使用方式

### 前置检查
```bash
sh skills/crawl-xueqiu-user-timeline/scripts/check-cdp.sh
sh skills/crawl-xueqiu-user-timeline/scripts/check-agent-browser.sh
```

### 爬取用户时间线
```bash
# 爬取最近 3 天（默认）
skills/crawl-xueqiu-user-timeline/scripts/crawl_xueqiu_user_timeline_api.py https://xueqiu.com/u/9493911686

# 爬取最近 7 天
skills/crawl-xueqiu-user-timeline/scripts/crawl_xueqiu_user_timeline_api.py https://xueqiu.com/u/9493911686 --days 7

# 指定日期范围
skills/crawl-xueqiu-user-timeline/scripts/crawl_xueqiu_user_timeline_api.py https://xueqiu.com/u/9493911686 --start-date 2026-01-01 --end-date 2026-03-05

# 指定输出文件名
skills/crawl-xueqiu-user-timeline/scripts/crawl_xueqiu_user_timeline_api.py https://xueqiu.com/u/9493911686 -o my_timeline.md
```

## 输出格式

生成 Markdown 文件，包含：
- 用户基本信息（UID、粉丝、关注、帖子数）
- 按时间排序的发言记录
- 每条发言包含：发布时间、内容、引用内容（如有）、互动数据、原文链接

## 注意事项

1. 需要先登录雪球账号
2. 如遇 Verification 验证页面，需手动完成验证后重新运行
3. 爬取过程中会自动处理分页和 md5__1038 令牌
4. 输出文件保存在当前工作目录

## 后续分析

爬取完成后，询问用户是否需要把雪球用户的发言总结分析一下。如果需要，可以：
1. 读取生成的 Markdown 文件
2. AI 分析发言内容、投资观点和情绪倾向
3. 生成总结报告
