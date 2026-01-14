import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import typer
from tqdm import tqdm


def get_records_with_time(connection: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """查询所有 time IS NOT NULL 的记录，返回 [(id, path, time), ...]"""
    cursor = connection.execute(
        "SELECT id, path, time FROM image_datetimes WHERE time IS NOT NULL ORDER BY id"
    )
    return cursor.fetchall()


def parse_time_string(time_str: str) -> datetime:
    """解析时间字符串为 datetime 对象，格式: YYYY-MM-DD HH:MM:SS"""
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


def generate_new_filename(datetime_obj: datetime, original_path: Path) -> str:
    """生成新文件名 YYYY-MM-DD_HHmmSS.<suffix>"""
    suffix = original_path.suffix.lower()
    filename = datetime_obj.strftime("%Y-%m-%d_%H%M%S")
    return f"{filename}{suffix}"


def get_target_path(base_dir: Path, datetime_obj: datetime, original_path: Path) -> Path:
    """生成目标路径 base_dir/YYYY/MM/YYYY-MM-DD_HHmmSS.<suffix>"""
    year = datetime_obj.strftime("%Y")
    month = datetime_obj.strftime("%m")
    filename = generate_new_filename(datetime_obj, original_path)
    return base_dir / year / month / filename


def move_file_safely(
    source_path: Path, target_path: Path
) -> Literal["success", "not_exists", "exists", "failed"]:
    """安全移动文件，返回移动结果"""
    if not source_path.exists():
        return "not_exists"

    if target_path.exists():
        return "exists"

    try:
        # 创建目标目录（如果不存在）
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # 移动文件
        source_path.rename(target_path)
        # 确认文件已移动
        if not target_path.exists() or source_path.exists():
            return "failed"
        return "success"
    except Exception:
        return "failed"


def process_record(
    connection: sqlite3.Connection, record_id: int, source_path: Path, target_path: Path
) -> Literal["success", "not_exists", "exists", "failed"]:
    """处理单条记录：移动文件，成功后更新数据库"""
    result = move_file_safely(source_path, target_path)

    # 只有文件移动成功时才更新数据库记录
    if result == "success":
        target_path_str = str(target_path.absolute())
        connection.execute(
            "UPDATE image_datetimes SET path = ? WHERE id = ?", (target_path_str, record_id)
        )
        connection.commit()

    return result


def main(
    base_dir: Annotated[Path, typer.Argument(help="目标基础目录")],
    db_path: Annotated[Path, typer.Option(help="数据库路径")] = Path("images.db"),
) -> None:
    """重命名和移动有时间的图片文件到按年月组织的文件夹"""
    base_dir_obj = Path(base_dir).resolve()
    db_path_obj = Path(db_path).resolve()

    if not db_path_obj.exists():
        print(f"错误: 数据库文件不存在: {db_path_obj}")
        raise typer.Exit(1)

    with sqlite3.connect(db_path_obj) as connection:
        records = get_records_with_time(connection)

    if not records:
        print("没有找到有时间的记录")
        return

    print(f"找到 {len(records)} 条有时间的记录")
    print(f"目标基础目录: {base_dir_obj}")

    # 统计信息
    success_count = 0
    not_exists_count = 0
    exists_count = 0
    failed_count = 0
    db_updated_count = 0

    # 逐个处理记录
    with sqlite3.connect(db_path_obj) as connection:
        for record_id, source_path_str, time_str in tqdm(records, desc="处理文件", unit="个"):
            try:
                datetime_obj = parse_time_string(time_str)
                source_path = Path(source_path_str)
                target_path = get_target_path(base_dir_obj, datetime_obj, source_path)

                result = process_record(connection, record_id, source_path, target_path)

                if result == "success":
                    success_count += 1
                    db_updated_count += 1
                elif result == "not_exists":
                    not_exists_count += 1
                elif result == "exists":
                    exists_count += 1
                    tqdm.write(f"目标文件已存在，跳过: {target_path}")
                else:  # failed
                    failed_count += 1
                    tqdm.write(f"移动失败: {source_path_str} -> {target_path}")
            except Exception as e:
                failed_count += 1
                tqdm.write(f"处理失败: {source_path_str}, 错误: {e}")

    # 显示统计结果
    print("\n处理完成！统计结果：")
    print(f"  成功移动的文件数: {success_count}")
    print(f"  文件不存在的数量: {not_exists_count}")
    print(f"  目标文件已存在的数量: {exists_count}")
    print(f"  移动失败的文件数: {failed_count}")
    print(f"  更新的数据库记录数: {db_updated_count}")


if __name__ == "__main__":
    typer.run(main)
