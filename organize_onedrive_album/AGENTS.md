# OneDrive 相册整理工具

**技术栈**: Python + Typer + SQLite + Pillow + tqdm

## 文件结构（数字前缀表示处理顺序）
```
organize_onedrive_album/
├── 1_create_database.py              # 创建数据库
├── 2_1_mark_time_by_exif.py          # EXIF 时间提取
├── 2_2_mark_time_by_filename.py      # 文件名时间提取
├── 3_1_mark_duplicate_time_images.py # 标记重复
├── 3_2_delete_marked.py              # 删除标记
├── 4_rename_images.py                # 重命名移动
└── images.db                         # SQLite 数据库
```

## 处理流程
```bash
# 1. 创建数据库
uv run python 1_create_database.py --db-path images.db

# 2. 标记时间
uv run python 2_1_mark_time_by_exif.py /path/to/images
uv run python 2_2_mark_time_by_filename.py /path/to/images  # 备用

# 3. 处理重复
uv run python 3_1_mark_duplicate_time_images.py
uv run python 3_2_delete_marked.py --preview  # 预览
uv run python 3_2_delete_marked.py            # 执行

# 4. 重命名
uv run python 4_rename_images.py /path/to/destination
```

## 数据库结构
```sql
CREATE TABLE image_datetimes (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    time TEXT,
    delete_mark INTEGER DEFAULT 0
)
```

## 输出格式
- 文件名: `YYYY-MM-DD_HHmmSS.<suffix>`
- 目录结构: `base_dir/YYYY/MM/`

## 支持格式
JPEG, PNG, WebP, HEIC/HEIF
