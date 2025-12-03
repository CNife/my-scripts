import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".gif", ".tiff"}


def main(
    image_dir: Path = typer.Argument(..., help="图像文件夹"),
) -> None:
    """递归查询图像文件夹，计算MD5哈希值，删除重复图像"""
    console = Console()
    images = list_images(image_dir)
    console.print(f"找到 {len(images)} 个图像文件")

    # 计算所有图像的哈希值
    hash_to_images = defaultdict(list)
    max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_image = {executor.submit(calculate_md5, image): image for image in images}

        # 使用进度条跟踪完成情况
        for future in track(as_completed(future_to_image), total=len(images), description="计算哈希值"):
            image = future_to_image[future]
            try:
                file_hash = future.result(timeout=30)  # 30秒超时
                hash_to_images[file_hash].append(image)
            except Exception as e:
                console.print(f"[red]计算 {image} 的哈希值时出错: {e}[/red]")

    # 找出重复的图像
    duplicates = {h: imgs for h, imgs in hash_to_images.items() if len(imgs) > 1}
    console.print(f"找到 {len(duplicates)} 组重复图像")

    if not duplicates:
        console.print("[green]没有找到重复图像[/green]")
        return

    # 处理重复图像
    process_duplicates(console, duplicates)


def list_images(image_dir: Path) -> list[Path]:
    """递归列出所有图像文件"""
    result_images = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            result_images.append(p)
        elif p.is_dir():
            result_images.extend(list_images(p))
    return result_images


def calculate_md5(file_path: Path) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def process_duplicates(console: Console, duplicates: dict[str, list[Path]]) -> None:
    """处理重复图像，让用户选择保留哪个"""
    for hash_value, images in duplicates.items():
        console.print(f"\n[bold]重复组 (哈希: {hash_value}):[/bold]")
        display_images_table(console, images)

        # 让用户选择保留哪个文件
        choice = get_user_choice(console, len(images))
        if choice is None:
            console.print("[yellow]跳过此组重复图像[/yellow]")
            continue

        kept_file = images[choice]
        deleted_files = [img for i, img in enumerate(images) if i != choice]

        # 确认删除
        if confirm_deletion(console, kept_file, deleted_files):
            # 删除重复文件
            for file_to_delete in deleted_files:
                try:
                    file_to_delete.unlink()
                    console.print(f"[green]已删除: {file_to_delete}[/green]")
                except Exception as e:
                    console.print(f"[red]删除 {file_to_delete} 时出错: {e}[/red]")

        else:
            console.print("[yellow]取消删除操作[/yellow]")


def display_images_table(console: Console, images: list[Path]) -> None:
    """显示重复图像表格"""
    table = Table(title="重复图像列表")
    table.add_column("序号", style="cyan", no_wrap=True)
    table.add_column("文件路径", style="magenta")
    table.add_column("文件大小", style="green")
    table.add_column("修改时间", style="yellow")

    for i, image in enumerate(images):
        try:
            size = image.stat().st_size
            mtime = image.stat().st_mtime
            from datetime import datetime

            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
        except Exception:
            size_str = "未知"
            mtime_str = "未知"

        table.add_row(str(i), str(image), size_str, mtime_str)

    console.print(table)


def get_user_choice(console: Console, num_images: int) -> int | None:
    """获取用户选择要保留哪个文件"""
    while True:
        try:
            choice = console.input(
                f"[bold cyan]请选择要保留的文件 (0-{num_images - 1})，或输入 's' 跳过: [/bold cyan]"
            ).strip()

            if choice.lower() == "s":
                return None

            choice_int = int(choice)
            if 0 <= choice_int < num_images:
                return choice_int
            else:
                console.print(f"[red]请输入 0 到 {num_images - 1} 之间的数字[/red]")
        except ValueError:
            console.print("[red]请输入有效的数字或 's'[/red]")


def confirm_deletion(console: Console, kept_file: Path, deleted_files: list[Path]) -> bool:
    """确认删除操作"""
    console.print(f"\n[bold]将保留:[/bold] {kept_file}")
    console.print("[bold]将删除:[/bold]")
    for file in deleted_files:
        console.print(f"  - {file}")

    confirmation = console.input("\n[bold yellow]确认删除？(y/N): [/bold yellow]").strip().lower()
    return confirmation == "y"


if __name__ == "__main__":
    typer.run(main)
