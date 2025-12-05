import re
import sqlite3
from datetime import datetime
from pathlib import Path

import exifread
import typer
from rich.progress import track

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def main(
    image_dir: Path = typer.Argument(..., help="图像文件夹"),
    db_path: Path = typer.Argument(Path(__file__).with_name("images.db"), help="数据库路径"),
) -> None:
    images = list_images(image_dir)
    with sqlite3.connect(db_path) as connection:
        create_table(connection)
        for image in track(images, description="处理图像"):
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
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_path TEXT NOT NULL,
                original_name TEXT NOT NULL,
                new_name TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def process_image(connection: sqlite3.Connection, image: Path) -> None:
    try:
        with connection:
            exif = read_exif(image)
            new_name = make_new_name(image, exif)
            if new_name:
                new_path = image.with_name(new_name)
                if new_path != image and not new_path.exists():
                    image.rename(new_path)
                    save_data(connection, image, new_name)
    except Exception as e:
        print(f"处理图像 {image} 时出错: {e}")


def should_process(connection: sqlite3.Connection, image: Path) -> bool:
    cursor = connection.execute(
        "SELECT COUNT(*) FROM images WHERE parent_path = ? AND original_name = ?",
        (str(image.parent), image.name),
    )
    return cursor.fetchone()[0] == 0


def read_exif(image: Path) -> dict:
    with open(image, "rb") as f:
        tags = exifread.process_file(f, details=False, extract_thumbnail=False)
    return tags


def make_new_name(old_image: Path, exif: dict) -> str | None:
    datetime_tag = exif.get("EXIF DateTimeOriginal") or exif.get("EXIF DateTimeDigitized") or exif.get("Image DateTime")
    if not datetime_tag:
        return None

    try:
        image_datetime = parse_exif_datetime(str(datetime_tag.values))
    except ValueError:
        return None
    image_datetime_str = image_datetime.strftime("%Y-%m-%d_%H%M%S")
    return f"{image_datetime_str}{old_image.suffix.lower()}"


def parse_exif_datetime(raw: str) -> datetime:
    s = raw.strip()
    m = re.match(r"(\d{4})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})\s+(\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})", s)
    if not m:
        raise ValueError(f"Invalid EXIF datetime: {raw}")
    y, mo, d, h, mi, se = map(int, m.groups())
    return datetime(y, mo, d, h, mi, se)


def save_data(connection: sqlite3.Connection, image: Path, new_name: str) -> None:
    connection.execute(
        "INSERT INTO images (parent_path, original_name, new_name) VALUES (?, ?, ?)",
        (str(image.parent), image.name, new_name),
    )
    connection.commit()


if __name__ == "__main__":
    typer.run(main)
