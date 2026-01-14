import sqlite3
from pathlib import Path
from typing import Annotated, Literal

import typer
from tqdm import tqdm


def get_marked_records(connection: sqlite3.Connection) -> list[tuple[int, str]]:
    """查询所有 delete_mark = 1 的记录，返回 [(id, path), ...]"""
    cursor = connection.execute(
        "SELECT id, path FROM image_datetimes WHERE delete_mark = 1 ORDER BY id"
    )
    return cursor.fetchall()


def delete_file_safely(file_path: Path) -> Literal["success", "not_exists", "failed"]:
    """安全删除文件，返回删除结果"""
    if not file_path.exists():
        return "not_exists"

    try:
        file_path.unlink()
        # 确认文件已删除
        if file_path.exists():
            return "failed"
        return "success"
    except Exception:
        return "failed"


def process_record(
    connection: sqlite3.Connection, record_id: int, file_path: Path
) -> Literal["success", "not_exists", "failed"]:
    """处理单条记录：删除文件，成功后删除数据库记录"""
    result = delete_file_safely(file_path)

    # 只有文件删除成功或文件不存在时，才删除数据库记录
    if result in ("success", "not_exists"):
        connection.execute("DELETE FROM image_datetimes WHERE id = ?", (record_id,))
        connection.commit()

    return result


def main(db_path: Annotated[Path, typer.Option(help="数据库路径")] = Path("images.db")) -> None:
    """删除数据库中 delete_mark = 1 的记录及其对应的文件"""
    db_path_obj = Path(db_path).resolve()
    if not db_path_obj.exists():
        print(f"错误: 数据库文件不存在: {db_path_obj}")
        raise typer.Exit(1)

    with sqlite3.connect(db_path_obj) as connection:
        records = get_marked_records(connection)

    if not records:
        print("没有找到标记为删除的记录")
        return

    print(f"找到 {len(records)} 条标记为删除的记录")

    # 统计信息
    success_count = 0
    not_exists_count = 0
    failed_count = 0
    db_deleted_count = 0

    # 逐个处理记录
    with sqlite3.connect(db_path_obj) as connection:
        for record_id, file_path_str in tqdm(records, desc="删除文件", unit="个"):
            file_path = Path(file_path_str)
            result = process_record(connection, record_id, file_path)

            if result == "success":
                success_count += 1
                db_deleted_count += 1
            elif result == "not_exists":
                not_exists_count += 1
                db_deleted_count += 1
            else:  # failed
                failed_count += 1
                tqdm.write(f"删除失败: {file_path_str}")

    # 显示统计结果
    print("\n删除完成！统计结果：")
    print(f"  成功删除的文件数: {success_count}")
    print(f"  文件不存在的数量: {not_exists_count}")
    print(f"  删除失败的文件数: {failed_count}")
    print(f"  删除的数据库记录数: {db_deleted_count}")


if __name__ == "__main__":
    typer.run(main)
