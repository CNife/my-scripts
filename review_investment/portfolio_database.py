"""投资组合数据库模型定义"""

# pyright: reportIncompatibleVariableOverride=false
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pathlib import Path

import typer
from peewee import (
    DateField,
    DateTimeField,
    DecimalField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

# 数据库连接（可以根据需要修改为其他数据库）
database = SqliteDatabase("portfolio.db")


class TransactionType(str, Enum):
    """交易类型枚举"""

    DEPOSIT = "入账"  # 资金入账
    WITHDRAW = "出账"  # 资金出账
    BUY = "买入"  # 买入
    SELL = "卖出"  # 卖出
    SUBSCRIBE = "申购"  # 申购（基金）
    REDEEM = "赎回"  # 赎回（基金）


class BaseModel(Model):
    """基础模型类"""

    class Meta:
        database = database


class Position(BaseModel):
    """仓位表（持仓表）"""

    account = TextField(verbose_name="账户", index=True)
    code = TextField(verbose_name="代码", index=True)  # 股票代码、基金代码、币种代码等
    name = TextField(verbose_name="名称")  # 股票名称、基金名称、币种名称等
    mv = DecimalField(
        max_digits=20, decimal_places=4, default=Decimal("0"), verbose_name="市值"
    )  # 当前市值
    qty = DecimalField(
        max_digits=20, decimal_places=8, default=Decimal("0"), verbose_name="数量"
    )  # 持有数量（支持小数，如基金份额、虚拟货币）
    pnl = DecimalField(
        max_digits=20, decimal_places=4, default=Decimal("0"), verbose_name="盈亏"
    )  # 当前持仓盈亏
    days = IntegerField(default=0, verbose_name="持有天数")  # 持有天数
    total_pnl = DecimalField(
        max_digits=20, decimal_places=4, default=Decimal("0"), verbose_name="累计盈亏"
    )  # 累计盈亏（包括已清仓的盈亏）
    updated_at = DateTimeField(default=datetime.now, verbose_name="更新时间")

    class Meta:
        table_name = "position"
        indexes = (
            (("account", "code"), True),  # 账户和代码唯一索引
        )


class Transaction(BaseModel):
    """交易记录表"""

    date = DateField(verbose_name="成交日期", index=True)
    time = DateTimeField(null=True, verbose_name="成交时间")  # 精确到时分秒
    account = TextField(verbose_name="账户", index=True)
    code = TextField(verbose_name="代码", index=True)  # 股票代码、基金代码、币种代码等
    type = TextField(verbose_name="交易类别")  # 交易类型（使用 TransactionType 枚举值）
    qty = DecimalField(
        max_digits=20, decimal_places=8, default=Decimal("0"), verbose_name="数量"
    )  # 成交数量（支持小数）
    price = DecimalField(
        max_digits=20, decimal_places=8, null=True, verbose_name="价格"
    )  # 成交价格（出入账可能没有价格）
    amount = DecimalField(
        max_digits=20, decimal_places=4, default=Decimal("0"), verbose_name="金额"
    )  # 成交金额
    fee = DecimalField(
        max_digits=20, decimal_places=4, default=Decimal("0"), verbose_name="手续费"
    )  # 手续费
    notes = TextField(null=True, verbose_name="备注")  # 备注信息
    created_at = DateTimeField(default=datetime.now, verbose_name="创建时间")

    class Meta:
        table_name = "transaction"
        indexes = (
            (("account", "code", "date"), False),  # 复合索引，便于查询
        )


class ClosedPosition(BaseModel):
    """清仓记录表"""

    account = TextField(verbose_name="账户", index=True)
    code = TextField(verbose_name="代码", index=True)  # 股票代码、基金代码、币种代码等
    name = TextField(verbose_name="名称")  # 股票名称、基金名称、币种名称等
    total_pnl = DecimalField(
        max_digits=20, decimal_places=4, default=Decimal("0"), verbose_name="总盈亏"
    )  # 总盈亏
    pnl_ratio = DecimalField(
        max_digits=10, decimal_places=4, null=True, verbose_name="盈亏比"
    )  # 盈亏比（百分比，如 15.5 表示 15.5%）
    buy_price = DecimalField(
        max_digits=20, decimal_places=8, null=True, verbose_name="买入均价"
    )  # 买入均价
    sell_price = DecimalField(
        max_digits=20, decimal_places=8, null=True, verbose_name="卖出均价"
    )  # 卖出均价
    fee = DecimalField(
        max_digits=20, decimal_places=4, default=Decimal("0"), verbose_name="交易费用"
    )  # 总交易费用
    open_date = DateField(verbose_name="建仓日期", index=True)  # 建仓日期
    close_date = DateField(verbose_name="清仓日期", index=True)  # 清仓日期
    created_at = DateTimeField(default=datetime.now, verbose_name="创建时间")

    class Meta:
        table_name = "closed_position"
        indexes = (
            (("account", "code", "close_date"), False),  # 复合索引
        )


def create_tables():
    """创建所有表"""
    if database.is_closed():
        database.connect()
    database.create_tables([Position, Transaction, ClosedPosition], safe=True)


def drop_tables():
    """删除所有表"""
    if database.is_closed():
        database.connect()
    database.drop_tables([Position, Transaction, ClosedPosition], safe=True)


def clear_tables():
    """清空所有表的数据（保留表结构）"""
    if database.is_closed():
        database.connect()
    Position.delete().execute()
    Transaction.delete().execute()
    ClosedPosition.delete().execute()


app = typer.Typer()


@app.command()
def create(
    db_path: Path = typer.Option(Path(__file__).parent / "portfolio.db", help="数据库文件路径"),
) -> None:
    """创建所有数据库表"""
    database.init(str(db_path))
    create_tables()
    database.close()
    typer.echo("数据库表创建完成！")


@app.command()
def drop(
    db_path: Path = typer.Option(Path(__file__).parent / "portfolio.db", help="数据库文件路径"),
) -> None:
    """删除所有数据库表"""
    database.init(str(db_path))
    drop_tables()
    database.close()
    typer.echo("数据库表已删除！")


@app.command()
def clear(
    db_path: Path = typer.Option(Path(__file__).parent / "portfolio.db", help="数据库文件路径"),
) -> None:
    """清空所有表的数据（保留表结构）"""
    database.init(str(db_path))
    clear_tables()
    database.close()
    typer.echo("数据库数据已清空！")


if __name__ == "__main__":
    app()
