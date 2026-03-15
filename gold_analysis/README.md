# 黄金投资组合分析工具

一次性脚本，分析黄金 ETF、黄金股票与指数之间的相关性及收益对比。

## 快速开始

```bash
# 设置 Tushare Token
export TUSHARE_TOKEN="your_token_here"

# 运行（默认 2024 年 1 月 1 日至今）
uv run python gold_analysis/gold_analysis.py

# 指定日期范围
uv run python gold_analysis/gold_analysis.py --start-date 20240101 --end-date 20250601
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--start-date` | 开始日期（YYYYMMDD） | 20240101 |
| `--end-date` | 结束日期（YYYYMMDD） | 今日 |
| `--output-dir` | 图表输出目录 | output |

## 输出

### 图表文件
- `returns_comparison.png` - 累计收益对比图
- `correlation_matrix.png` - 相关性矩阵热力图

### 终端输出
- 最终累计收益率
- 相关性矩阵数值

## 依赖

```bash
uv add pandas tushare matplotlib seaborn
```

## Tushare Token

访问 [Tushare Pro](https://tushare.pro/) 注册获取 Token，然后：

```bash
export TUSHARE_TOKEN="your_token_here"
# 或
echo "your_token_here" > ~/.tushare_token
```

## 开发注意事项

### 数据获取限制

1. **Tushare API 权限**
   - 基础积分用户只能获取部分数据
   - 日线行情需要至少 120 积分
   - 指数数据需要至少 300 积分
   - 检查积分：登录 Tushare 官网查看

2. **交易日对齐**
   - 不同市场（股票、指数）交易日可能不同
   - 脚本会自动对齐到共同交易日（内连接）
   - 如果数据量差异大，检查是否有停牌/节假日

3. **复权处理**
   - 默认使用**后复权**（`hfq`），包含分红再投资
   - 如需不复权数据，修改代码中 `adj=""`

### 图表定制

修改 `gold_analysis.py` 中的配色和样式：

```python
# 第 135-145 行：颜色配置
styles = [
    {"color": "#C41E3A", "label": "AU9999", "ls": "-"},   # 深红
    {"color": "#0066CC", "label": "紫金矿业", "ls": "-"}, # 深蓝
    # ... 自定义颜色
]

# 第 158 行：字体大小
plt.rcParams["font.size"] = 9  # 调整全局字体

# 第 182 行：热力图颜色
cmap="RdBu_r"  # 红蓝配色，可改为 "coolwarm", "viridis" 等
```

### 常见问题

**Q: 报错 "无数据返回"**
- 检查日期范围是否在交易日内
- 确认 Tushare Token 积分足够
- 检查股票代码是否正确（需要 `.SH` 或 `.SZ` 后缀）

**Q: 图表中文乱码**
- 脚本默认使用 `MiSans` 字体
- Linux 用户安装：`sudo apt install fonts-noto-cjk`
- 或修改第 42 行字体配置为系统已有字体

**Q: 相关性矩阵全是 1**
- 检查输入数据是否只有 1 个资产
- 确认日期范围有足够交易日（至少 2 天）

### 性能优化

- **数据量过大**：缩小日期范围，或分批处理
- **内存占用**：脚本会一次性加载所有数据到内存
- **重复运行**：手动添加缓存（当前版本无缓存，每次重新获取）

### 扩展资产

添加新资产到分析：

```python
# 修改第 20-22 行配置
STOCK_CODES = ["601899.SH", "600489.SH", "600547.SH", "新代码"]
INDEX_CODES = ["931238.CSI", "000985.CSI", "新指数"]

# 修改第 24-29 行名称映射
ASSET_NAMES = {
    # ... 添加新资产名称
}
```

### 数据导出

如需导出原始数据到 CSV，在 `main()` 函数末尾添加：

```python
# 保存对齐后的价格数据
aligned_prices.to_csv(output_dir / "prices.csv")
# 保存累计收益率
cumulative_returns.to_csv(output_dir / "cumulative_returns.csv")
# 保存相关性矩阵
corr_matrix.to_csv(output_dir / "correlation.csv")
```