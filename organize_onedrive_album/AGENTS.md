# OneDrive 相册整理工具

**目录**: organize_onedrive_album/
**功能**: 多步骤 pipeline 处理图像文件：EXIF 时间提取、重复检测、文件重命名
**技术栈**: Python + Typer + SQLite + Pillow + tqdm

## 概述

该工具是一个图像处理 pipeline，包含多个脚本，通过 SQLite 数据库协调处理过程。主要功能包括：
- 从 EXIF 数据和文件名提取拍摄时间
- 检测重复时间的图像
- 重命名和组织图像到按年月分类的文件夹

## 文件结构
```
organize_onedrive_album/
├── 1_create_database.py              # 创建数据库
├── 2_1_mark_time_by_exif.py          # 从 EXIF 提取时间
├── 2_2_mark_time_by_filename.py      # 从文件名提取时间
├── 3_1_mark_duplicate_time_images.py # 标记重复时间图像
├── 3_2_delete_marked.py              # 删除标记的图像
├── 4_rename_images.py                # 重命名和移动图像
└── images.db                         # SQLite 数据库（已提交到 git）
```

## 处理流程

### 步骤 1: 创建数据库
```bash
uv run python 1_create_database.py --db-path images.db
```

### 步骤 2: 标记时间
```bash
# 从 EXIF 提取时间
uv run python 2_1_mark_time_by_exif.py /path/to/images

# 从文件名提取时间（备用方案）
uv run python 2_2_mark_time_by_filename.py /path/to/images
```

### 步骤 3: 处理重复
```bash
uv run python 3_1_mark_duplicate_time_images.py

# 查看和删除标记的图像
uv run python 3_2_delete_marked.py --preview
uv run python 3_2_delete_marked.py
```

### 步骤 4: 重命名和组织
```bash
uv run python 4_rename_images.py /path/to/destination
```

## 代码地图

### 核心函数
| 函数 | 位置 | 作用 |
|------|------|------|
| main | 1_create_database.py:8-13 | 创建数据库表 |
| create_table | 1_create_database.py:16-29 | 初始化 image_datetimes 表 |
| main | 2_1_mark_time_by_exif.py:18-37 | 处理图像文件夹 |
| list_images | 2_1_mark_time_by_exif.py:40-47 | 递归列出图像文件 |
| process_image | 2_1_mark_time_by_exif.py:58-71 | 处理单个图像的 EXIF 数据 |
| read_exif | 2_1_mark_time_by_exif.py:74-101 | 读取 EXIF 数据 |
| get_exif_datetime | 2_1_mark_time_by_exif.py:104-116 | 提取时间信息 |
| main | 4_rename_images.py:78-138 | 重命名和移动图像 |
| get_records_with_time | 4_rename_images.py:10-15 | 查询有时间记录的图像 |
| move_file_safely | 4_rename_images.py:38-58 | 安全移动文件 |
| process_record | 4_rename_images.py:61-75 | 处理单条记录 |

## 数据库结构

### image_datetimes 表
```sql
CREATE TABLE image_datetimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    time TEXT,
    delete_mark INTEGER DEFAULT 0
)
```

## 支持的图像格式

| 格式 | 扩展名 |
|------|--------|
| JPEG | .jpg, .jpeg |
| PNG | .png |
| WebP | .webp |
| HEIC/HEIF | .heic, .heif |

## 配置和参数

### 通用参数
| 参数 | 说明 | 位置 |
|------|------|------|
| --db-path | 数据库路径 | 所有脚本 |
| --preview | 预览删除操作 | 3_2_delete_marked.py |

### 处理参数
| 参数 | 说明 | 位置 |
|------|------|------|
| image_dir | 图像文件夹路径 | 2_1_mark_time_by_exif.py, 2_2_mark_time_by_filename.py |
| base_dir | 目标基础目录 | 4_rename_images.py |

## 输出格式

### 新文件名格式
```
YYYY-MM-DD_HHmmSS.<suffix>
```

### 目标目录结构
```
base_dir/
├── 2023/
│   ├── 01/
│   │   ├── 2023-01-01_103045.jpg
│   │   └── ...
│   └── 02/
└── 2024/
```

## 注意事项

1. **数据库提交**: images.db 已提交到 git，包含示例数据（2MB）
2. **预览功能**: 使用 --preview 选项预览删除操作
3. **错误处理**: 各脚本都有错误处理和统计功能
4. **进度显示**: 使用 tqdm 显示处理进度
