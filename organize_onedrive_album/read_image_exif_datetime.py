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
    with sqlite3.connect(db_path_obj) as connection:
        create_table(connection)
        for image in tqdm(images, desc="处理图像"):
            process_image(connection, image)


def list_images(image_dir: Path) -> list[Path]:
    result_images = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            result_images.append(p)
        elif p.is_dir():
            result_images.extend(list_images(p))
    return result_images


def create_table(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_datetimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                time TEXT
            )
            """
        )
        connection.commit()


def should_skip(connection: sqlite3.Connection, image_path: str) -> bool:
    cursor = connection.execute(
        "SELECT COUNT(*) FROM image_datetimes WHERE path = ? AND time IS NOT NULL", (image_path,)
    )
    count = cursor.fetchone()[0]
    return count > 0


def process_image(connection: sqlite3.Connection, image: Path) -> None:
    try:
        image_path = str(image.absolute())
        if should_skip(connection, image_path):
            return

        exif = read_exif(image)
        image_datetime = get_exif_datetime(exif)
        save_data(connection, image_path, image_datetime)
    except Exception as e:
        print(f"处理图像 {image} 时出错: {e}")


def read_exif(image: Path) -> dict:
    """使用 Pillow 读取 EXIF 数据，支持 HEIF/HEIC 和 PNG"""
    try:
        with Image.open(image) as img:
            exif_data = img.getexif()
            if not exif_data:
                return {}
            # 将 EXIF tag 编号转换为可读的标签名
            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = value
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
    """解析 EXIF 时间字符串，格式：YYYY:MM:DD HH:MM:SS"""
    s = raw.strip()
    m = re.match(
        r"(\d{4})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})\s+(\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})", s
    )
    if not m:
        raise ValueError(f"Invalid EXIF datetime: {raw}")
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


if __name__ == "__main__":
    typer.run(main)
