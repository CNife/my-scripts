import sqlite3
from pathlib import Path
from typing import Annotated

import typer


def main(db_path: Annotated[Path, typer.Option(help="数据库路径")] = Path("images.db")) -> None:
    """创建数据库并初始化表结构"""
    db_path_obj = Path(db_path)
    with sqlite3.connect(db_path_obj) as connection:
        create_table(connection)
    print(f"数据库初始化完成: {db_path_obj}")


def create_table(connection: sqlite3.Connection) -> None:
    """创建 image_datetimes 表"""
    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_datetimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                time TEXT,
                delete_mark INTEGER DEFAULT 0
            )
            """
        )
        connection.commit()


if __name__ == "__main__":
    typer.run(main)
