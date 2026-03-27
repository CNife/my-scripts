# 中证A50成分股 PE/PB 分位点及股息率

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
# 运行约 2.5 分钟（511 次周频 API 调用 + 50 只股票股息率计算）
```

## 计算逻辑

| 指标 | 公式 | 数据源 |
|------|------|--------|
| PE/PB 分位点 | 历史周频数据中 ≤ 当前值的比例 | `daily_basic`（按 trade_date 拉全市场，本地过滤） |
| 股息率(%) | 3年平均股利支付率 / PE_TTM × 100 | `dividend`（cash_div）+ `income`（basic_eps） |
| 股利支付率 | 每股现金分红 / 每股收益（EPS≤0 的年份跳过） | — |

## 关键常量

| 常量 | 值 | 说明 |
|------|----|------|
| INDEX_CODE | 930050.CSI | 中证A50指数 |
| LOOKBACK_YEARS | 10 | PE/PB 回溯区间 |
| PAYOUT_LOOKBACK_YEARS | 3 | 股利支付率回溯年数 |

## 已知坑

- `daily_basic` **不支持** `ts_code + start_date/end_date` 范围查询，只支持 `trade_date`（全市场）或 `ts_code + trade_date`（单股单日）
- `dividend` 同一财年可能返回多条记录（含送股、转增等 `cash_div=0` 的事件），**必须按 end_date 聚合求和后过滤 > 0** 再与 EPS 合并
- Tushare 频率限制约 200 次/分钟，脚本内置重试（3 次）+ sleep
