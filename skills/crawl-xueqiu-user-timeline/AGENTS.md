# 雪球用户时间线爬取技能

**技术栈**: Python + Playwright

## 文件结构
```
skills/crawl-xueqiu-user-timeline/
├── SKILL.md
└── scripts/crawl_xueqiu_user_timeline_api.py
```

## CLI 参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| url | 用户主页链接 | 必填 |
| --days | 最近 N 天 | 3 |
| --start-date | 开始日期 | - |
| -o | 输出文件名 | 自动生成 |

## 使用
```bash
# 前置检查
sh skills/crawl-xueqiu-user-timeline/scripts/check-cdp.sh

# 爬取
skills/crawl-xueqiu-user-timeline/scripts/crawl_xueqiu_user_timeline_api.py https://xueqiu.com/u/9493911686
skills/crawl-xueqiu-user-timeline/scripts/crawl_xueqiu_user_timeline_api.py https://xueqiu.com/u/9493911686 --days 7
```

## 输出
- 用户基本信息
- 按时间排序的发言记录
- Markdown 格式

## 注意事项
- 需先登录雪球账号
- `--days` 和 `--start-date` 互斥
