# A股ETF风险平价计算器使用指南

## 概述

风险平价（Risk Parity）是一种资产配置方法，其核心思想是让投资组合中每个资产对总风险的贡献相等。不同于传统的等权重配置，风险平价根据资产的波动率和相关性来分配权重，实现更优的风险分散化效果。

## 安装配置

### 环境要求

- Python 3.12+
- uv 包管理器

### 依赖安装

项目依赖已配置在 `pyproject.toml` 中，使用 uv 同步环境：

```bash
uv sync
```

所需主要依赖：
- numpy: 数值计算
- scipy: 优化算法
- pandas: 数据处理
- tushare: A股数据获取
- rich: 终端美化输出
- typer: CLI框架

### Tushare Token 配置

风险平价计算器需要从 Tushare 获取 ETF 历史价格数据。需要配置 Tushare API Token：

1. 访问 [Tushare官网](https://tushare.pro/) 注册账号
2. 在个人主页获取 Token
3. 设置环境变量（推荐）：

```bash
export TUSHARE_TOKEN="your_token_here"
```

或者在代码中直接设置（不推荐用于生产环境）：

```python
import tushare as ts
ts.set_token("your_token_here")
```

## 配置文件格式

### 配置文件结构

创建 JSON 格式的配置文件，例如 `risk_parity_config.json`：

```json
{
    "etfs": [
        {
            "name": "沪深300ETF",
            "code": "510300",
            "category": "stock",
            "description": "大盘蓝筹指数"
        }
    ],
    "settings": {
        "default_date_range": "1y",
        "start_date": "20250313",
        "end_date": "20260313",
        "risk_target": null,
        "offline_mode": false
    }
}
```

### 配置字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `etfs` | 数组 | 是 | ETF 列表 |
| `etfs[].name` | 字符串 | 是 | ETF 名称 |
| `etfs[].code` | 字符串 | 是 | ETF 代码（交易所代码，如 510300） |
| `etfs[].category` | 字符串 | 是 | 资产类别：`stock`（股票）、`bond`（债券）、`commodity`（商品）、`cash`（现金） |
| `etfs[].description` | 字符串 | 否 | ETF 描述 |
| `settings` | 对象 | 否 | 计算设置 |
| `settings.default_date_range` | 字符串 | 否 | 默认日期范围，如 `1y`（1年）、`6m`（6个月） |
| `settings.start_date` | 字符串 | 否 | 数据起始日期，格式：YYYYMMDD |
| `settings.end_date` | 字符串 | 否 | 数据结束日期，格式：YYYYMMDD |
| `settings.risk_target` | 数字 | 否 | 目标波动率（可选） |
| `settings.offline_mode` | 布尔 | 否 | 是否使用离线缓存数据 |

### ETF 类别建议

建议在配置中覆盖不同资产类别以实现更好的风险分散：

- **股票ETF**：沪深300、中证500、创业板、红利等
- **债券ETF**：国债ETF、企业债ETF
- **商品ETF**：黄金ETF、白银ETF
- **现金管理**：货币ETF

## 使用方法

### 基本用法

使用默认配置（内置6只ETF）运行：

```bash
uv run python fund_portfolio_rebalancing/risk_parity_calculator.py
```

### 指定配置文件

```bash
uv run python fund_portfolio_rebalancing/risk_parity_calculator.py \
    --config fund_portfolio_rebalancing/risk_parity_config.json
```

### 自定义日期范围

```bash
uv run python fund_portfolio_rebalancing/risk_parity_calculator.py \
    --config fund_portfolio_rebalancing/risk_parity_config.json \
    --start-date 20240101 \
    --end-date 20241231
```

### 离线模式

仅使用本地缓存数据，不请求网络：

```bash
uv run python fund_portfolio_rebalancing/risk_parity_calculator.py \
    --config fund_portfolio_rebalancing/risk_parity_config.json \
    --offline
```

### 参数速查

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--config` | `-c` | ETF配置文件路径 | 内置默认配置 |
| `--start-date` | `-s` | 数据起始日期(YYYYMMDD) | 20230101 |
| `--end-date` | `-e` | 数据结束日期(YYYYMMDD) | 20241231 |
| `--offline` | `-o` | 离线模式，仅用缓存 | false |

## 输出解读

### 输出示例

```
===== 风险平价计算 =====
数据区间: 20240101 至 20241231
离线模式: 否
ETF数量: 7

===== 风险平价权重分配 =====
ETF名称     ETF代码    权重      风险贡献   目标风险贡献  偏差
沪深300ETF  510300    15.23%    14.28%     14.28%       -0.00%
中证500ETF  510500    18.45%    14.29%     14.28%       +0.01%
创业板ETF   159915    12.67%    14.28%     14.28%       -0.00%
红利ETF     510880    20.12%    14.27%     14.28%       -0.01%
黄金ETF     518880    11.23%    14.29%     14.28%       +0.01%
国债ETF     511010    18.56%    14.28%     14.28%       -0.00%
货币ETF     511880    3.74%     14.31%     14.28%       +0.03%
合计        -          100.00%   100.00%    100.00%      -

===== 组合风险指标 =====
组合年化波动率: 12.45%
风险贡献均衡度: 0.000023

===== 风险贡献可视化 =====
沪深300ETF |████████████████████████| 14.28%
中证500ETF |████████████████████████| 14.29%
创业板ETF  |████████████████████████| 14.28%
红利ETF    |████████████████████████| 14.27%
黄金ETF    |████████████████████████| 14.29%
国债ETF    |████████████████████████| 14.28%
货币ETF    |████████████████████████| 14.31%
```

### 关键指标说明

| 指标 | 说明 | 解读 |
|------|------|------|
| **权重** | 各ETF在组合中的资金占比 | 风险平价会自动给低波动资产分配更高权重 |
| **风险贡献** | 各ETF对组合总风险的贡献比例 | 理想情况下每个资产应为 1/N |
| **偏差** | 实际风险贡献与目标的差异 | 越接近0越好，说明风险分布越均衡 |
| **组合年化波动率** | 整个组合的年化标准差 | 反映组合的整体风险水平 |
| **风险贡献均衡度** | 风险贡献偏离程度的度量 | 数值越接近0表示风险分布越均衡 |

### 风险平价原理

风险平价的核心公式：

```
RC_i = (w_i * (Σw)_i) / (w^T Σ w)
```

其中：
- RC_i：资产 i 的风险贡献
- w_i：资产 i 的权重
- Σ：协方差矩阵
- w^T Σ w：组合方差

风险平价的目标是让每个资产的 RC_i 相等，即每个资产对组合风险的贡献相同。

## 注意事项

1. **数据质量**：风险平价计算结果依赖于历史数据的准确性和代表性，过往表现不代表未来收益。

2. **Tushare限制**：免费版Tushare有API调用频率限制，如遇限速可使用离线模式。

3. **流动性风险**：部分ETF可能流动性较差，实际交易时滑点较大。

4. **费用未计入**：计算结果未考虑交易费用、管理费等实际成本。

5. **再平衡频率**：建议定期（如季度或半年）重新计算并调整仓位。

## 故障排除

### 无法获取数据

- 检查 TUSHARE_TOKEN 环境变量是否正确设置
- 确认网络连接正常
- 尝试使用 `--offline` 模式

### 优化失败

- 减少ETF数量（建议4-10只）
- 检查是否有ETF代码错误
- 尝试延长数据时间范围

### 波动率为0

- 检查数据是否获取成功
- 确认ETF代码正确（应为交易所代码如510300）
