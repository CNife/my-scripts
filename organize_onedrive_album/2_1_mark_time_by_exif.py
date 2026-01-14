import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from PIL.ExifTags import TAGS
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def main(
    image_dir: Annotated[Path, typer.Argument(help="图像文件夹")],
    db_path: Annotated[Path, typer.Option(help="数据库路径")] = Path("images.db"),
) -> None:
    db_path_obj = Path(db_path)
    images = list_images(image_dir)
    stats = {"total": len(images), "skipped": 0, "success": 0, "no_time": 0, "error": 0}
    with sqlite3.connect(db_path_obj) as connection:
        for image in tqdm(images, desc="处理图像"):
            result = process_image(connection, image)
            if result == "skipped":
                stats["skipped"] += 1
            elif result == "success":
                stats["success"] += 1
            elif result == "no_time":
                stats["no_time"] += 1
            elif result == "error":
                stats["error"] += 1

    print_summary(stats)


def list_images(image_dir: Path) -> list[Path]:
    result_images = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            result_images.append(p)
        elif p.is_dir():
            result_images.extend(list_images(p))
    return result_images


def should_skip(connection: sqlite3.Connection, image_path: str) -> bool:
    cursor = connection.execute(
        "SELECT COUNT(*) FROM image_datetimes WHERE path = ? AND time IS NOT NULL", (image_path,)
    )
    count = cursor.fetchone()[0]
    return count > 0


def process_image(connection: sqlite3.Connection, image: Path) -> str:
    """处理单个图像，返回处理结果：'skipped', 'success', 'no_time', 'error'"""
    try:
        image_path = str(image.absolute())
        if should_skip(connection, image_path):
            return "skipped"

        exif = read_exif(image)
        image_datetime = get_exif_datetime(exif)
        save_data(connection, image_path, image_datetime)
        return "success" if image_datetime else "no_time"
    except Exception as e:
        print(f"处理图像 {image} 时出错: {e}")
        return "error"


def read_exif(image: Path) -> dict:
    """使用 Pillow 读取 EXIF 数据，支持 HEIF/HEIC 和 PNG
    同时读取主 IFD 和 Exif IFD 中的数据
    """
    try:
        with Image.open(image) as img:
            exif_data = img.getexif()
            if not exif_data:
                return {}
            # 将 EXIF tag 编号转换为可读的标签名
            exif_dict = {}
            # 读取主 IFD 的数据
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = value
            # 读取 Exif IFD 的数据（包含 DateTimeOriginal 等）
            try:
                if hasattr(exif_data, "get_ifd"):
                    exif_ifd = exif_data.get_ifd(0x8769)  # Exif IFD
                    for tag_id, value in exif_ifd.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        # Exif IFD 中的标签优先（如 DateTimeOriginal）
                        exif_dict[tag_name] = value
            except Exception:
                ...  # 如果没有 Exif IFD，继续使用主 IFD 的数据
            return exif_dict
    except Exception:
        return {}


def get_exif_datetime(exif: dict) -> datetime | None:
    """从 Pillow EXIF 字典中提取时间"""
    # EXIF 时间标签：DateTimeOriginal (36867), DateTimeDigitized (36868), DateTime (306)
    datetime_str = (
        exif.get("DateTimeOriginal") or exif.get("DateTimeDigitized") or exif.get("DateTime")
    )
    if not datetime_str:
        return None

    try:
        return parse_exif_datetime(datetime_str)
    except (ValueError, TypeError):
        return None


def parse_exif_datetime(raw: str) -> datetime:
    """解析 EXIF 时间字符串，支持多种格式：
    - 标准格式：YYYY:MM:DD HH:MM:SS
    - 带空格的格式：YYYY: M:DD  H:MM:SS 或 YYYY: M: D  H:MM:SS
    """
    s = raw.strip()
    # 规范化字符串：将多个连续空格替换为单个空格，统一处理各种空格变体
    normalized = re.sub(r"\s+", " ", s)
    # 匹配格式：YYYY:MM:DD HH:MM:SS 或 YYYY: M:DD H:MM:SS 等变体
    # \s* 允许冒号前后有空格，\s+ 要求日期和时间之间至少有一个空格
    m = re.match(
        r"(\d{4})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})\s+(\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})",
        normalized,
    )
    if not m:
        raise ValueError(f"Invalid EXIF datetime format: {raw!r}")
    y, mo, d, h, mi, se = map(int, m.groups())
    return datetime(y, mo, d, h, mi, se)


def save_data(
    connection: sqlite3.Connection, image_path: str, image_datetime: datetime | None
) -> None:
    time_str = image_datetime.strftime("%Y-%m-%d %H:%M:%S") if image_datetime else None
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO image_datetimes (path, time) VALUES (?, ?)",
            (image_path, time_str),
        )
        connection.commit()


def print_summary(stats: dict[str, int]) -> None:
    """打印处理总结"""
    print("\n" + "=" * 50)
    print("处理总结")
    print("=" * 50)
    print(f"总图像数量: {stats['total']}")
    print(f"  跳过（已有时间信息）: {stats['skipped']}")
    print(f"  成功提取时间: {stats['success']}")
    print(f"  无时间信息: {stats['no_time']}")
    print(f"  处理出错: {stats['error']}")
    print("=" * 50)


if __name__ == "__main__":
    typer.run(main)
