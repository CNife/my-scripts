import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm


def main(
    db_path: Annotated[Path, typer.Option(help="数据库路径")] = Path("images.db"),
    skip_marked: Annotated[bool, typer.Option(help="跳过已标记的组")] = False,
) -> None:
    db_path_obj = Path(db_path).resolve()
    if not db_path_obj.exists():
        tqdm.write(f"错误: 数据库文件不存在: {db_path_obj}")
        raise typer.Exit(1)

    # 查询重复时间的图片
    duplicate_groups = get_duplicate_time_images(db_path_obj, skip_marked)
    if not duplicate_groups:
        tqdm.write("没有找到重复时间的图片")
        return

    tqdm.write(f"找到 {len(duplicate_groups)} 组重复时间的图片")

    # 处理每组重复图片
    for idx, (time_str, images) in enumerate(
        tqdm(duplicate_groups, desc="处理重复图片组", unit="组"), 1
    ):
        tqdm.write(f"\n处理第 {idx}/{len(duplicate_groups)} 组")
        tqdm.write(f"时间: {time_str}")
        tqdm.write(f"图片数量: {len(images)}")

        # 自动选择尺寸最大的图片
        selected_index = find_largest_image(images)
        if selected_index is None:
            tqdm.write("跳过此组（无法确定文件大小）")
            continue

        # 标记需要删除的图片
        mark_images_for_deletion(db_path_obj, images, selected_index)
        tqdm.write(f"已保留尺寸最大的图片（索引 {selected_index + 1}），其他已标记为删除")

    tqdm.write("\n处理完成！")


def get_duplicate_time_images(
    db_path: Path, skip_marked: bool = False
) -> list[tuple[str, list[tuple[int, str]]]]:
    """查询重复时间的图片，返回 [(time, [(id, path), ...]), ...]"""
    with sqlite3.connect(db_path) as connection:
        # 先获取所有重复的时间
        cursor = connection.execute(
            """
            SELECT time
            FROM image_datetimes
            WHERE time IS NOT NULL
            GROUP BY time
            HAVING COUNT(*) > 1
            ORDER BY time
            """
        )
        duplicate_times = [row[0] for row in cursor.fetchall()]

        # 对每个时间，获取所有对应的图片
        result = []
        for time_str in tqdm(duplicate_times, desc="查询图片信息", unit="时间", leave=False):
            if skip_marked:
                cursor = connection.execute(
                    """
                    SELECT id, path
                    FROM image_datetimes
                    WHERE time = ? AND (delete_mark IS NULL OR delete_mark = 0)
                    ORDER BY id
                    """,
                    (time_str,),
                )
            else:
                cursor = connection.execute(
                    """
                    SELECT id, path
                    FROM image_datetimes
                    WHERE time = ?
                    ORDER BY id
                    """,
                    (time_str,),
                )
            images = cursor.fetchall()
            if len(images) > 1:  # 确保仍然有重复
                result.append((time_str, images))

        return result


def find_largest_image(images: list[tuple[int, str]]) -> int | None:
    """找到尺寸最大的图片，返回其索引"""
    sizes = []
    image_info = []

    # 显示图片信息表格
    tqdm.write("\n重复图片列表")
    tqdm.write("=" * 100)

    # 表头
    header = f"{'序号':<6} {'ID':<8} {'文件路径':<50} {'文件大小':<12} {'修改时间':<20}"
    tqdm.write(header)
    tqdm.write("-" * 100)

    for idx, (image_id, image_path) in enumerate(images):
        path_obj = Path(image_path)
        size = 0
        size_str = "文件不存在"
        mtime_str = "未知"

        try:
            if path_obj.exists():
                stat = path_obj.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime)
                size_str = (
                    f"{size / 1024:.1f} KB"
                    if size < 1024 * 1024
                    else f"{size / (1024 * 1024):.1f} MB"
                )
                mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
            else:
                size = -1  # 文件不存在，标记为 -1
        except Exception as e:
            size = -1
            size_str = f"错误: {e}"

        sizes.append(size)
        image_info.append((image_id, image_path, size_str, mtime_str))

        # 截断过长的路径
        path_display = str(image_path)
        if len(path_display) > 50:
            path_display = path_display[:47] + "..."

        row = f"{idx + 1:<6} {image_id:<8} {path_display:<50} {size_str:<12} {mtime_str:<20}"
        tqdm.write(row)

    tqdm.write("=" * 100)

    # 找到尺寸最大的图片索引
    # 如果所有文件都不存在或出错，返回 None
    valid_sizes = [(i, s) for i, s in enumerate(sizes) if s >= 0]
    if not valid_sizes:
        return None

    # 找到最大尺寸的索引
    largest_idx = max(valid_sizes, key=lambda x: x[1])[0]

    # 显示选择结果
    largest_id, largest_path, largest_size_str, _ = image_info[largest_idx]
    tqdm.write("\n自动选择尺寸最大的图片:")
    tqdm.write(f"  序号: {largest_idx + 1}")
    tqdm.write(f"  ID: {largest_id}")
    tqdm.write(f"  路径: {largest_path}")
    tqdm.write(f"  大小: {largest_size_str}")

    return largest_idx


def mark_images_for_deletion(db_path: Path, images: list[tuple[int, str]], keep_index: int) -> None:
    """标记需要删除的图片（除了保留的那张）"""
    with sqlite3.connect(db_path) as connection:
        for idx, (image_id, _image_path) in enumerate(images):
            if idx != keep_index:
                connection.execute(
                    "UPDATE image_datetimes SET delete_mark = 1 WHERE id = ?", (image_id,)
                )
        connection.commit()


if __name__ == "__main__":
    typer.run(main)
