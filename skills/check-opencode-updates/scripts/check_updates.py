#!/usr/bin/env python3
# ruff: noqa: S310, S603

import json
import subprocess
import urllib.request
from typing import NamedTuple


class VersionInfo(NamedTuple):
    tool_name: str
    local_version: str | None
    remote_version: str | None
    status: str
    release_notes: str


def get_releases(url: str) -> list[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/urllib"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception:
        return []


def get_remote_version(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/urllib"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("tag_name", "").lstrip("v")
    except Exception:
        return None


def get_local_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def parse_version(version: str) -> tuple[int, ...]:
    parts = version.split(".")
    result = []
    for p in parts:
        num = "".join(c for c in p if c.isdigit())
        result.append(int(num) if num else 0)
    return tuple(result)


def get_releases_between(
    releases: list[dict], local_version: str, remote_version: str
) -> list[dict]:
    result = []
    local_tuple = parse_version(local_version)
    remote_tuple = parse_version(remote_version)

    for release in releases:
        tag = release.get("tag_name", "").lstrip("v")
        if not tag:
            continue
        tag_tuple = parse_version(tag)
        if tag_tuple <= local_tuple:
            continue
        if tag_tuple > remote_tuple:
            continue
        result.append(release)

    result.sort(key=lambda r: parse_version(r.get("tag_name", "").lstrip("v")), reverse=True)
    return result


def format_release_notes(releases: list[dict], tool_name: str) -> str:
    if not releases:
        return ""

    lines = [f"\n## {tool_name} 更新日志\n"]
    for release in releases:
        tag = release.get("tag_name", "").lstrip("v")
        name = release.get("name", tag)
        body = release.get("body", "").strip()
        url = release.get("html_url", "")

        lines.append(f"### {name}")
        if body:
            preview = body[:500] + "..." if len(body) > 500 else body
            lines.append(preview)
        if url:
            lines.append(f"\n[查看完整发布说明]({url})")
        lines.append("")

    return "\n".join(lines)


def compare_versions(local: str | None, remote: str | None) -> str:
    if local is None:
        return "not installed"
    if remote is None:
        return "error"
    if local == remote:
        return "up to date"
    return "update available"


def check_tool(
    name: str,
    local_cmd: list[str],
    releases_url: str,
    latest_url: str,
) -> VersionInfo:
    remote = get_remote_version(latest_url)
    local = get_local_version(local_cmd)
    status = compare_versions(local, remote)

    release_notes = ""
    if status == "update available" and local and remote:
        releases = get_releases(releases_url)
        relevant = get_releases_between(releases, local, remote)
        release_notes = format_release_notes(relevant, name)

    return VersionInfo(name, local, remote, status, release_notes)


def main() -> None:
    tools = [
        check_tool(
            "opencode",
            ["opencode", "--version"],
            "https://api.github.com/repos/anomalyco/opencode/releases",
            "https://api.github.com/repos/anomalyco/opencode/releases/latest",
        ),
        check_tool(
            "oh-my-openagent",
            ["bunx", "oh-my-opencode-linux-x64", "--version"],
            "https://api.github.com/repos/code-yeongyu/oh-my-openagent/releases",
            "https://api.github.com/repos/code-yeongyu/oh-my-openagent/releases/latest",
        ),
    ]

    has_updates = False
    for tool in tools:
        local = tool.local_version or "N/A"
        print(f"{tool.tool_name}: {local} ({tool.status})")
        if tool.status == "update available":
            has_updates = True

    if has_updates:
        print("\n" + "=" * 50)
        for tool in tools:
            if tool.release_notes:
                print(tool.release_notes)


if __name__ == "__main__":
    main()
