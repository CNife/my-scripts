# 雪球首页时间线爬取技能

**技术栈**: Python + Playwright

## 文件结构
```
skills/crawl-xueqiu-my-timeline/
├── SKILL.md          # 技能定义
├── scripts/
│   └── crawl_xueqiu_home_timeline_api.py
└── evals/evals.json
```

## CLI 参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --hours | 最近 N 小时 | 24 |
| --days | 最近 N 天 | - |
| --start-date | 开始日期 | - |
| --end-date | 结束日期 | 今天 |

## 使用
```bash
# 前置检查
sh skills/crawl-xueqiu-my-timeline/scripts/check-cdp.sh

# 爬取
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py
skills/crawl-xueqiu-my-timeline/scripts/crawl_xueqiu_home_timeline_api.py --hours 2
```

## 输出
- 按发言人分组，发言数量降序
- 自动过滤官方账号
- Markdown 格式

## 注意事项
- 需先登录雪球账号
- 如遇验证页面需手动完成
- 输出保存在当前工作目录
