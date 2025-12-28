"""从 Excel 文件导入投资组合数据到数据库"""

import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import typer
from portfolio_database import (
    ClosedPosition,
    Position,
    Transaction,
    TransactionType,
    create_tables,
    database,
)
from rich.console import Console

console = Console()

# 交易类别映射
TRANSACTION_TYPE_MAP = {
    "收入": TransactionType.DEPOSIT.value,
    "入账": TransactionType.DEPOSIT.value,
    "银证转入": TransactionType.DEPOSIT.value,
    "银证转出": TransactionType.WITHDRAW.value,
    "出账": TransactionType.WITHDRAW.value,
    "买入": TransactionType.BUY.value,
    "卖出": TransactionType.SELL.value,
    "申购": TransactionType.SUBSCRIBE.value,
    "赎回": TransactionType.REDEEM.value,
    "融券": TransactionType.BUY.value,  # 融券视为买入
    "融券购回": TransactionType.SELL.value,  # 融券购回视为卖出
}


def parse_datetime(date_str: str | None, time_str: str | None = None) -> datetime | None:
    """解析日期时间字符串"""
    if date_str is None or (isinstance(date_str, (str, float)) and pd.isna(date_str)):
        return None

    if isinstance(date_str, datetime):
        date = date_str
    elif isinstance(date_str, str):
        try:
            date = pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            return None
    else:
        try:
            date = pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            return None

    if time_str is not None and not (isinstance(time_str, (str, float)) and pd.isna(time_str)):
        if isinstance(time_str, str):
            # 解析时间字符串，格式可能是 "HH:MM:SS" 或 "HH:MM"
            time_match = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(time_str))
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                second = int(time_match.group(3)) if time_match.group(3) else 0
                date = datetime(
                    year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=hour,
                    minute=minute,
                    second=second,
                )

    return date


def safe_decimal(value, default=Decimal("0")):
    """安全转换为 Decimal"""
    if pd.isna(value) or value is None:
        return default
    try:
        return Decimal(str(value))
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """安全转换为整数"""
    if pd.isna(value) or value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_str(value, default=""):
    """安全转换为字符串"""
    if pd.isna(value) or value is None:
        return default
    return str(value).strip()


def _safe_decimal_or_none(value):
    """安全转换为 Decimal 或 None"""
    if value is None:
        return None
    if isinstance(value, (str, float)) and pd.isna(value):
        return None
    try:
        return safe_decimal(value)
    except Exception:
        return None


def import_positions(excel_path: Path, account_name: str) -> int:
    """导入持仓数据"""
    xl = pd.ExcelFile(excel_path)
    df = pd.read_excel(xl, sheet_name="持仓数据")

    # 过滤掉汇总行（代码为"汇总"或名称为空）
    df = df[df["代码"].notna() & (df["代码"] != "汇总") & (df["名称"].notna())]

    count = 0
    for _, row in df.iterrows():
        code = safe_str(row["代码"])
        if not code:
            continue

        position, created = Position.get_or_create(
            account=account_name,
            code=code,
            defaults={
                "name": safe_str(row["名称"]),
                "mv": safe_decimal(row.get("持有金额", 0)),
                "qty": safe_decimal(row.get("持有数量", 0)),
                "pnl": safe_decimal(row.get("持有盈亏", 0)),
                "days": safe_int(row.get("持仓天数", 0)),
                "total_pnl": safe_decimal(row.get("累计盈亏", 0)),
                "updated_at": datetime.now(),
            },
        )
        if not created:
            # 更新现有记录
            position.name = safe_str(row["名称"])
            position.mv = safe_decimal(row.get("持有金额", 0))
            position.qty = safe_decimal(row.get("持有数量", 0))
            position.pnl = safe_decimal(row.get("持有盈亏", 0))
            position.days = safe_int(row.get("持仓天数", 0))
            position.total_pnl = safe_decimal(row.get("累计盈亏", 0))
            position.updated_at = datetime.now()
            position.save()
        count += 1

    return count


def import_transactions(excel_path: Path, account_name: str) -> int:
    """导入交易记录"""
    xl = pd.ExcelFile(excel_path)
    df = pd.read_excel(xl, sheet_name="交易记录")

    count = 0
    for _, row in df.iterrows():
        # 解析日期和时间
        trade_date = parse_datetime(row.get("成交日期"))
        if trade_date is None:
            continue

        trade_time = parse_datetime(row.get("成交日期"), row.get("成交时间"))

        # 获取交易类别
        transaction_type_str = safe_str(row.get("交易类别", ""))
        transaction_type = TRANSACTION_TYPE_MAP.get(transaction_type_str, transaction_type_str)

        # 对于出入账，可能没有代码
        code = safe_str(row.get("代码", ""))

        # 确定金额字段（可能是"发生金额"或"成交金额"）
        amount = safe_decimal(row.get("发生金额") or row.get("成交金额", 0))

        Transaction.create(
            date=trade_date.date(),
            time=trade_time,
            account=account_name,
            code=code,
            type=transaction_type,
            qty=safe_decimal(row.get("成交数量", 0)),
            price=safe_decimal(row.get("成交价格")),
            amount=amount,
            fee=safe_decimal(row.get("费用", 0)),
            notes=safe_str(row.get("备注")),
            created_at=datetime.now(),
        )
        count += 1

    return count


def import_closed_positions(excel_path: Path, account_name: str) -> int:
    """导入清仓记录"""
    xl = pd.ExcelFile(excel_path)
    df = pd.read_excel(xl, sheet_name="已清仓")

    count = 0
    for _, row in df.iterrows():
        code = safe_str(row.get("代码"))
        if not code:
            continue

        close_date = parse_datetime(row.get("清仓日期"))
        open_date = parse_datetime(row.get("建仓日期"))

        if close_date is None:
            continue

        ClosedPosition.create(
            account=account_name,
            code=code,
            name=safe_str(row.get("名称")),
            total_pnl=safe_decimal(row.get("总盈亏", 0)),
            pnl_ratio=_safe_decimal_or_none(row.get("盈亏比")),
            buy_price=_safe_decimal_or_none(row.get("买入均价")),
            sell_price=_safe_decimal_or_none(row.get("卖出均价")),
            fee=safe_decimal(row.get("交易费用", 0)),
            open_date=open_date.date() if open_date else None,
            close_date=close_date.date(),
            created_at=datetime.now(),
        )
        count += 1

    return count


def main(
    excel_path: Path = typer.Argument(..., help="Excel 文件路径"),
    account: str | None = typer.Argument(None, help="账户名称（默认从文件名提取）"),
    db_path: Path = typer.Option(Path(__file__).parent / "portfolio.db", help="数据库文件路径"),
) -> None:
    """从 Excel 文件导入投资组合数据"""
    # 检查文件是否存在
    if not excel_path.exists():
        console.print(f"[red]错误: 文件不存在 {excel_path}[/red]")
        return

    # 如果没有提供账户名，从文件名提取（去掉扩展名）
    if account is None:
        account = excel_path.stem

    # 更新数据库路径
    database.init(str(db_path))
    database.connect()

    try:
        # 创建表
        create_tables()

        # 导入账户数据
        console.print(f"[cyan]导入账户数据: {excel_path} (账户: {account})[/cyan]")

        pos_count = import_positions(excel_path, account)
        console.print(f"  ✓ 导入持仓记录: {pos_count} 条")

        trans_count = import_transactions(excel_path, account)
        console.print(f"  ✓ 导入交易记录: {trans_count} 条")

        closed_count = import_closed_positions(excel_path, account)
        console.print(f"  ✓ 导入清仓记录: {closed_count} 条")

        console.print("\n[green]✓ 数据导入完成！[/green]")

    finally:
        database.close()


if __name__ == "__main__":
    typer.run(main)
