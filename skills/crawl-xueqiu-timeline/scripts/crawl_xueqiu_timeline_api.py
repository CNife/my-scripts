#!/usr/bin/env -S uv run
"""
爬取雪球用户时间线脚本（API 方式）
通过 agent-browser 在浏览器内执行 API 请求，自动处理 md5__1038 令牌和反爬机制
支持 --days 参数指定爬取最近 N 天（默认 3 天）
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


def run_agent_browser(args: list[str]) -> str:
    """执行 agent-browser 命令并返回输出"""
    cmd = ["agent-browser", "--cdp", "9222", *args]
    result = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=60
    )
    output = result.stdout
    lines = []
    for line in output.split("\n"):
        line = line.strip()
        if line and not line.startswith("✓") and not line.startswith("Done"):
            lines.append(line)
    return "\n".join(lines)


def open_page(url: str) -> None:
    """打开雪球用户主页"""
    print(f"正在打开页面：{url}")
    run_agent_browser(["open", url])
    time.sleep(5)


def get_api_data_in_browser(user_id: str, page: int = 1) -> dict:
    """在浏览器内通过 fetch 执行 API 请求，自动包含 md5__1038 和 Cookie"""
    js_code = f"""
    (async () => {{
        const r = await fetch('https://xueqiu.com/v4/statuses/user_timeline.json?page={page}&user_id={user_id}&count=20', {{
            method: 'GET',
            headers: {{
                'Accept': 'application/json, text/plain, */*',
                'X-Requested-With': 'XMLHttpRequest',
            }},
            credentials: 'include'
        }});
        return await r.json();
    }})()
    """
    result = run_agent_browser(["eval", js_code])

    try:
        data = json.loads(result)
        if isinstance(data, dict):
            return data
        return {}
    except json.JSONDecodeError as e:
        print(f"解析 API 响应失败：{e}, 原始输出：{result[:200]}")
        return {}


def parse_date_string(date_str: str | None, default: datetime) -> datetime:
    """解析 YYYY-MM-DD 格式日期字符串"""
    if not date_str:
        return default
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"日期格式错误，请使用 YYYY-MM-DD 格式：{date_str}") from e


def parse_timestamp(timestamp_ms: int) -> str:
    """将毫秒时间戳转换为绝对时间格式"""
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError, OSError:
        return "未知时间"


def clean_html(text: str) -> str:
    """清理 HTML 标签"""
    import re

    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&#39;", "'")
    text = text.replace("&quot;", '"')
    return text.strip()


def extract_quote_info(status: dict) -> tuple[str, str]:
    """提取转发/回复帖中的引用信息"""
    retweeted = status.get("retweeted_status", {})
    if not retweeted:
        return "", ""

    quote_user = retweeted.get("user", {}).get("screen_name", "")
    quote_text = clean_html(retweeted.get("text", ""))

    return quote_user, quote_text


def parse_status(status: dict) -> dict:
    """解析单条帖子数据"""
    post_id = status.get("id", "")
    user_id = status.get("user_id", "")
    created_at = status.get("created_at", 0)
    description = status.get("description", "")
    retweet_count = status.get("retweet_count", 0)
    reply_count = status.get("reply_count", 0)
    like_count = status.get("like_count", 0)

    content = clean_html(description)
    content = content[:500]

    quote_user, quote_content = extract_quote_info(status)

    url = f"https://xueqiu.com/{user_id}/{post_id}" if post_id else ""

    return {
        "post_id": post_id,
        "user_id": user_id,
        "post_time": parse_timestamp(created_at) if created_at else "未知时间",
        "content": content,
        "quote_user": quote_user,
        "quote_content": quote_content[:300] if quote_content else "",
        "reposts": retweet_count,
        "comments": reply_count,
        "likes": like_count,
        "url": url,
    }


def fetch_timeline_in_range(user_id: str, start_date: datetime, end_date: datetime) -> list:
    """爬取指定日期范围内的时间线，自动处理多页"""
    all_statuses = []
    page = 1
    start_ms = start_date.timestamp() * 1000
    end_ms = end_date.timestamp() * 1000

    while True:
        print(f"正在爬取第 {page} 页...")
        api_data = get_api_data_in_browser(user_id, page=page)
        statuses = api_data.get("statuses", [])

        if not statuses:
            print("没有更多数据")
            break

        # 检查是否有帖子在范围内
        in_range = [s for s in statuses if start_ms <= s.get("created_at", 0) <= end_ms]
        all_statuses.extend(in_range)

        # 检查是否可以停止：最老的帖子已早于开始日期
        oldest_timestamp = min(s.get("created_at", 0) for s in statuses)
        if oldest_timestamp < start_ms:
            print("已爬取到开始日期之前的数据，停止")
            break

        page += 1
        time.sleep(2)

    return all_statuses


def save_to_markdown(posts: list[dict], user_info: dict, output_file: str) -> None:
    """保存为 Markdown 文件"""
    content = []

    content.append(f"# 雪球时间线 (UID: {user_info.get('uid', 'N/A')})\n")
    content.append("## 时间线\n")

    for post in posts:
        content.append(f"### {post['post_time']}")

        if post["quote_user"]:
            content.append(f"回复@{post['quote_user']}: {post['content']}")
        else:
            content.append(post["content"])

        if post["quote_content"]:
            content.append(
                f"\n> @{post['quote_user'] if post['quote_user'] else ''}: {post['quote_content']}"
            )

        content.append(f"\n> {post['url']}")
        content.append("")
        content.append("---\n")

    Path(output_file).write_text("\n".join(content), encoding="utf-8")
    print(f"已保存到：{output_file}")


def get_user_info(user_id: str) -> dict:
    """获取用户信息"""
    user_info = {
        "name": "",
        "uid": user_id,
        "followers": "",
        "following": "",
        "posts": "",
    }

    import re

    lines, _ = get_snapshot()

    for line in lines:
        if "关注" in line and "link" in line:
            match = re.search(r"(\d+)\s*关注", line)
            if match:
                user_info["following"] = match.group(1)

        if "粉丝" in line and "link" in line:
            match = re.search(r"(\d+)\s*粉丝", line)
            if match:
                user_info["followers"] = match.group(1)

        if "帖子" in line and "link" in line:
            match = re.search(r"(\d+)\s*帖子", line)
            if match:
                user_info["posts"] = match.group(1)

        if "heading" in line and "level=2" in line:
            match = re.search(r'"([^"]+)"', line)
            if match:
                name = match.group(1).replace(" 设置备注", "").strip()
                if name:
                    user_info["name"] = name

    return user_info


def get_snapshot() -> tuple[list[str], dict]:
    """获取页面快照，返回快照行列表和 refs 字典"""
    cmd = ["agent-browser", "--cdp", "9222", "snapshot", "--json"]
    result = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=60
    )
    try:
        data = json.loads(result.stdout)
        snapshot_text = data.get("data", {}).get("snapshot", "")
        refs = data.get("data", {}).get("refs", {})
        return snapshot_text.split("\n"), refs
    except json.JSONDecodeError:
        print("获取快照失败，原始输出：", result.stdout[:500])
        return [], {}


def main():
    parser = argparse.ArgumentParser(description="爬取雪球用户时间线（API 方式）")
    parser.add_argument("url", nargs="?", help="雪球用户主页链接")
    parser.add_argument("--days", type=int, help="爬取最近 N 天，默认 3 天")
    parser.add_argument("--start-date", help="开始日期 (YYYY-MM-DD)，默认 3 天前")
    parser.add_argument("--end-date", help="结束日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("-o", "--output", help="输出文件名（默认自动生成）")
    args = parser.parse_args()

    if args.days and args.start_date:
        print("错误：--days 和 --start-date 参数互斥，不能同时使用")
        return

    url = args.url
    if not url:
        url = input("请输入雪球用户主页链接（如 https://xueqiu.com/u/9493911686）: ").strip()

    if not url:
        print("链接不能为空")
        return

    import re

    user_id_match = re.search(r"/u/(\d+)", url) or re.search(r"/(\d+)", url)
    if not user_id_match:
        print("无法从链接中提取用户 ID")
        return
    user_id = user_id_match.group(1)

    end_date = parse_date_string(args.end_date, datetime.now())

    if args.days:
        start_date = datetime.now() - timedelta(days=args.days)
    elif args.start_date:
        start_date = parse_date_string(args.start_date, datetime.now() - timedelta(days=3))
    else:
        start_date = datetime.now() - timedelta(days=3)

    output_file = args.output

    print("步骤 1: 打开雪球页面...")
    open_page(url)
    time.sleep(3)

    print(f"步骤 2: 获取用户 {user_id} 的时间线数据...")
    print(f"爬取时间范围：{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    statuses = fetch_timeline_in_range(user_id, start_date, end_date)

    if not statuses:
        print("未找到任何帖子数据")
        return

    print(f"获取到 {len(statuses)} 条帖子")

    print("步骤 3: 解析帖子数据...")
    posts = [parse_status(s) for s in statuses]

    print("步骤 4: 获取用户信息...")
    time.sleep(2)
    user_info = get_user_info(user_id)
    print(f"用户：{user_info.get('name', 'N/A')} (UID: {user_id})")

    if not output_file:
        user_name = user_info.get("name", "user")
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        output_file = f"{user_name}_{start_str}_{end_str}.md"

    print("步骤 5: 保存到 Markdown 文件...")
    save_to_markdown(posts, user_info, output_file)

    print("\n=== 预览前 3 条帖子 ===")
    for i, post in enumerate(posts[:3], 1):
        print(f"\n[帖子{i}] {post['post_time']}")
        if post["quote_user"]:
            print(f"回复@{post['quote_user']}: {post['content'][:50]}...")
        else:
            print(f"内容：{post['content'][:50]}...")


if __name__ == "__main__":
    main()
