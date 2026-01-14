import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm


def get_records_without_time(connection: sqlite3.Connection) -> list[tuple[int, str]]:
    """查询所有 time IS NULL 的记录，返回 [(id, path), ...]"""
    cursor = connection.execute(
        "SELECT id, path FROM image_datetimes WHERE time IS NULL ORDER BY id"
    )
    return cursor.fetchall()


def is_valid_year(datetime_obj: datetime) -> bool:
    """验证时间是否在2000-2099年范围内"""
    return 2000 <= datetime_obj.year <= 2099


def parse_unix_timestamp_from_filename(filename: str) -> datetime | None:
    """从文件名中解析Unix时间戳（支持毫秒和秒）"""
    # 提取文件名中的所有数字序列
    numbers = re.findall(r"\d+", filename)

    for num_str in numbers:
        num_len = len(num_str)
        # 13位为毫秒时间戳，10位为秒时间戳
        if num_len == 13:
            try:
                timestamp_ms = int(num_str)
                timestamp_s = timestamp_ms / 1000
                dt = datetime.fromtimestamp(timestamp_s)
                if is_valid_year(dt):
                    return dt
            except (ValueError, OSError):
                continue
        elif num_len == 10:
            try:
                timestamp_s = int(num_str)
                dt = datetime.fromtimestamp(timestamp_s)
                if is_valid_year(dt):
                    return dt
            except (ValueError, OSError):
                continue

    return None


def parse_datetime_from_filename(filename: str) -> datetime | None:
    """从文件名中解析 YYYYMMDDHHmmSS 格式的时间"""
    # 提取文件名中的所有数字序列
    numbers = re.findall(r"\d+", filename)

    for num_str in numbers:
        # 查找14位连续数字
        if len(num_str) == 14:
            try:
                # 解析为 YYYYMMDDHHmmSS 格式
                year = int(num_str[0:4])
                month = int(num_str[4:6])
                day = int(num_str[6:8])
                hour = int(num_str[8:10])
                minute = int(num_str[10:12])
                second = int(num_str[12:14])
                dt = datetime(year, month, day, hour, minute, second)
                if is_valid_year(dt):
                    return dt
            except (ValueError, TypeError):
                continue

    return None


def parse_ios_datetime_from_filename(filename: str) -> datetime | None:
    """从文件名中解析 iOS 格式的时间：YYYYMMDD_HHmmSSmmm_后缀.扩展名"""
    # 匹配格式：8位日期_9位时间（HHmmSSmmm）_后缀
    pattern = r"(\d{8})_(\d{9})_"
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHmmSSmmm
        try:
            # 解析日期部分
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            # 解析时间部分（前6位是时分秒，后3位是毫秒，datetime不支持毫秒所以忽略）
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            dt = datetime(year, month, day, hour, minute, second)
            if is_valid_year(dt):
                return dt
        except (ValueError, TypeError):
            return None
    return None


def parse_time_from_filename(filename: str) -> datetime | None:
    """统一的时间解析入口，按顺序尝试三种解析方式"""
    # 先尝试Unix时间戳解析
    dt = parse_unix_timestamp_from_filename(filename)
    if dt is not None:
        return dt

    # 再尝试 iOS 格式解析：YYYYMMDD_HHmmSSmmm_后缀
    dt = parse_ios_datetime_from_filename(filename)
    if dt is not None:
        return dt

    # 最后尝试 YYYYMMDDHHmmSS 格式解析
    dt = parse_datetime_from_filename(filename)
    if dt is not None:
        return dt

    return None


def update_record_time(
    connection: sqlite3.Connection, record_id: int, datetime_obj: datetime
) -> None:
    """更新数据库记录的 time 字段"""
    time_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
    connection.execute("UPDATE image_datetimes SET time = ? WHERE id = ?", (time_str, record_id))
    connection.commit()


def main(db_path: Annotated[Path, typer.Option(help="数据库路径")] = Path("images.db")) -> None:
    """根据文件名解析时间并更新数据库中 time IS NULL 的记录"""
    db_path_obj = Path(db_path).resolve()

    if not db_path_obj.exists():
        print(f"错误: 数据库文件不存在: {db_path_obj}")
        raise typer.Exit(1)

    with sqlite3.connect(db_path_obj) as connection:
        records = get_records_without_time(connection)

    if not records:
        print("没有找到 time IS NULL 的记录")
        return

    print(f"找到 {len(records)} 条 time IS NULL 的记录")

    # 统计信息
    parse_failed_count = 0
    db_updated_count = 0

    # 逐个处理记录
    with sqlite3.connect(db_path_obj) as connection:
        for record_id, source_path_str in tqdm(records, desc="处理文件", unit="个"):
            try:
                source_path = Path(source_path_str)
                filename = source_path.name

                # 从文件名解析时间
                datetime_obj = parse_time_from_filename(filename)
                if datetime_obj is None:
                    parse_failed_count += 1
                    continue

                # 更新数据库的 time 字段
                update_record_time(connection, record_id, datetime_obj)
                db_updated_count += 1
            except Exception as e:
                tqdm.write(f"处理失败: {source_path_str}, 错误: {e}")

    # 显示统计结果
    print("\n处理完成！统计结果：")
    print(f"  成功更新数据库的记录数: {db_updated_count}")
    print(f"  无法解析文件名的数量: {parse_failed_count}")


if __name__ == "__main__":
    typer.run(main)
