# 中证A50成分股 EP/BP 分位点及股息率

**技术栈**: Python + tushare + pandas

## 文件结构
```
a50_valuation/
├── a50_valuation.py    # 主脚本
└── a50_valuation.md    # 输出结果（运行后生成）
```

## 使用
```bash
uv run python a50_valuation/a50_valuation.py
# 运行约 3 分钟（周频价格数据 + 50 只股票财务数据 + 股息率计算）
```

## 计算逻辑

| 指标 | 公式 | 数据源 |
|------|------|--------|
| EP | EPS / 收盘价（每股收益/股价） | `daily`（close）+ `fina_indicator`（eps） |
| BP | BVPS / 收盘价（每股净资产/股价） | `daily`（close）+ `fina_indicator`（bps） |
| EP/BP 分位点 | P(历史 >= 当前值)，低分位点 = 被低估 | 周频 EP/BP 序列 |
| 股息率(%) | 3年平均股利支付率 × EP × 100 | `dividend`（cash_div）+ `income`（basic_eps） |
| 股利支付率 | 每股现金分红 / 每股收益 | — |

## 关键常量

| 常量 | 值 | 说明 |
|------|----|------|
| INDEX_CODE | 930050.CSI | 中证A50指数 |
| LOOKBACK_YEARS | 10 | EP/BP 回溯区间 |
| PAYOUT_LOOKBACK_YEARS | 3 | 股利支付率回溯年数 |

## 数据获取策略

### 价格数据
- 按周频 `trade_date` 拉全市场 `daily` 收盘价，本地过滤成分股
- 每次调用约 5000+ 行，10 年周频约 520 次 API 调用

### 财务数据
- 从 `fina_indicator` 获取 EPS（每股收益）和 BPS（每股净资产）
- 财务数据为季度发布，按 `ann_date`（公告日）前向填充到周频交易日
- 公告日当天及之后使用新数据，公告日之前使用上一期数据

## 股息率计算细节

无分红或盈利为负的年份，股利支付率设为 0（而非跳过）：
- 无分红记录 → 支付率 = 0
- EPS ≤ 0 → 支付率 = 0
- EPS > 0 且有分红 → 支付率 = cash_div / basic_eps

## 已知坑

- `daily` **不支持** `ts_code + start_date/end_date` 范围查询，只支持 `trade_date`（全市场）
- `fina_indicator` 财务数据为季度发布，需要前向填充到周频
- `dividend` 同一财年可能返回多条记录（含送股、转增等 `cash_div=0` 的事件），按 `end_date` 聚合求和
- Tushare 频率限制约 200 次/分钟，脚本内置重试（3 次）+ sleep

## 为什么用 EP/BP 而非 PE/PB？

PE/PB 对亏损企业返回负值或 NaN，无法正确计算分位点排序。用倒数（EP/BP）：
- 盈利为正 → EP 为正，值越大越便宜
- 盈利为负 → EP 为负，自然落入"昂贵"区间
- 盈利为零 → EP = 0
