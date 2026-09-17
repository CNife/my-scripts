#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["loguru"]
# ///
"""nmem 出站请求网关（单 lane）。

    nmem(background purpose) --HTTP--> 本网关 --HTTPS--> 上游 LLM 端点

每个请求固定做五件事，没有任何按 lane 的分支：

  1. 读配置（每请求重读；读/解析失败则沿用上次成功值，坏配置不打死正在跑的服务）
  2. 覆盖 ``model`` / ``max_tokens`` / ``thinking`` / ``reasoning_effort``
     （配置为 null 则不动那一个字段；``thinking`` 显式设为 null 表示删除该字段）
  3. 转发到 ``upstream_base_url`` + ``/chat/completions``（``/models`` 原样代理），带配置里的头与上游密钥
  4. 剥离响应里的推理内容（``reasoning_content`` / ``reasoning`` / ``reasoning_details``）
  5. 追加一行 JSONL 到日志

配置：同目录的 ``config.json``（不入库，权限 600），字段见 ``config.example.json``。
可用环境变量 ``GW_CONFIG`` 指向别处。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from loguru import logger

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("GW_CONFIG") or HERE / "config.json")
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details")
UPSTREAM_PATH = "/chat/completions"

_config_lock = threading.Lock()
_config: dict[str, Any] | None = None


def read_config() -> dict[str, Any]:
    """每请求重读配置；解析失败时沿用上一次成功的值。"""
    global _config
    with _config_lock:
        try:
            fresh = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(fresh, dict):
                raise ValueError("配置顶层必须是 JSON 对象")
        except Exception as exc:
            if _config is None:
                raise SystemExit(f"配置不可用 {CONFIG_PATH}: {exc}") from exc
            logger.warning("配置不可用，沿用上次成功值：{}", exc)
            return _config
        _config = fresh
        return fresh


def as_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw.decode("utf-8", "replace")


def rewrite(body: bytes, cfg: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """改写发往上游的请求体，返回 (新请求体, 本次改写的记录)。"""
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("顶层不是 JSON 对象")
    except Exception as exc:
        logger.warning("请求体不是 JSON 对象，原样转发：{}", exc)
        return body, {"rewrite": "skipped"}

    notes: dict[str, Any] = {"model_in": payload.get("model")}
    if cfg.get("upstream_model"):
        payload["model"] = cfg["upstream_model"]
    if cfg.get("max_tokens") is not None:
        payload["max_tokens"] = cfg["max_tokens"]
    if "thinking" in cfg:
        if cfg["thinking"] is None:
            payload.pop("thinking", None)
            notes["thinking"] = "removed"
        else:
            payload["thinking"] = cfg["thinking"]
            notes["thinking"] = cfg["thinking"]
    if cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg["reasoning_effort"]
        notes["reasoning_effort"] = cfg["reasoning_effort"]
    notes["model_out"] = payload.get("model")
    notes["max_tokens_out"] = payload.get("max_tokens")
    notes["stream_in"] = bool(payload.get("stream"))
    notes["stream"] = bool(payload.get("stream"))
    return json.dumps(payload, ensure_ascii=False).encode(), notes


def strip_reasoning(raw: bytes) -> tuple[bytes, list[str]]:
    """删掉响应消息里的推理内容，返回 (新响应体, 命中的字段名)。"""
    payload = as_json(raw)
    if not isinstance(payload, dict):
        return raw, []
    hit: list[str] = []
    for choice in payload.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            for key in REASONING_KEYS:
                if message.pop(key, None) is not None:
                    hit.append(key)
    if not hit:
        return raw, []
    return json.dumps(payload, ensure_ascii=False).encode(), hit


def open_upstream(
    method: str, path: str, body: bytes | None, cfg: dict[str, Any]
) -> tuple[Any | None, int, bytes, str]:
    """打开上游连接。

    成功：``(响应对象, 状态码, b"", content-type)``，由调用方负责读取与关闭。
    失败：``(None, 状态码, 错误体, content-type)``。
    """
    url = str(cfg["upstream_base_url"]).rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    headers.update(cfg.get("upstream_headers") or {})
    headers["Authorization"] = f"Bearer {cfg['upstream_api_key']}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)  # noqa: S310
    try:
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"unsupported upstream scheme: {scheme}")
        response = urllib.request.urlopen(  # noqa: S310
            request, timeout=cfg.get("timeout_seconds", 120)
        )
    except urllib.error.HTTPError as exc:
        content_type = (
            exc.headers.get("Content-Type") if exc.headers else None
        ) or "application/json"
        return None, exc.code, exc.read(), content_type
    except Exception as exc:
        logger.error("上游请求失败：{}", exc)
        failure = {
            "error": {
                "message": f"nmem-gw: upstream request failed: {exc}",
                "type": "gateway_error",
            }
        }
        return None, 502, json.dumps(failure, ensure_ascii=False).encode(), "application/json"
    return (
        response,
        response.status,
        b"",
        response.headers.get("Content-Type") or "application/json",
    )


def forward(
    method: str, path: str, body: bytes | None, cfg: dict[str, Any]
) -> tuple[int, bytes, str]:
    """非流式：把上游响应读全后返回（错误也照带）。"""
    response, status, error_body, content_type = open_upstream(method, path, body, cfg)
    if response is None:
        return status, error_body, content_type
    with response:
        return status, response.read(), content_type


class Server(ThreadingHTTPServer):
    """客户端中途断连（nmem 取消或超时后关连接）不是故障，不该刷 traceback。"""

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            logger.debug("客户端断连：{}", client_address)
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "nmem-gw/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        """我们有自己的 JSONL 记录，不要 http.server 往 stderr 刷访问日志。"""

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        body = json.dumps(
            {"error": {"message": message, "type": "gateway_error"}}, ensure_ascii=False
        ).encode()
        self._send(status, body)

    def _record(self, started: float, **fields: Any) -> None:
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "path": self.path,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "auth_in": bool(self.headers.get("Authorization")),
        }
        record.update(fields)
        logger.info(json.dumps(record, ensure_ascii=False))

    def _write_chunk(self, data: bytes, *, last: bool = False) -> None:
        if data:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        if last:
            self.wfile.write(b"0\r\n\r\n")

    def _close_stream(self) -> bool:
        """补上 chunked 结束块；客户端已经走了就返回 False，不抛栈。"""
        try:
            self._write_chunk(b"", last=True)
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    def _proxy_stream(
        self, started: float, response: Any, notes: dict[str, Any], cfg: dict[str, Any]
    ) -> None:
        """把上游 SSE 逐行透传给 nmem，顺手剥掉思考增量，并把整段内容记进日志。"""
        self.send_response(response.status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.close_connection = True

        hits: list[str] = []
        text: list[str] = []
        usage: Any = None
        finish: Any = None
        chunks = 0
        aborted: str | None = None
        strip = cfg.get("strip_reasoning", True)
        try:
            for raw in response:
                out = raw
                if raw.startswith(b"data: "):
                    data = raw[6:].strip()
                    if data and data != b"[DONE]":
                        parsed = as_json(data)
                        if isinstance(parsed, dict):
                            if isinstance(parsed.get("usage"), dict):
                                usage = parsed["usage"]
                            for choice in parsed.get("choices") or []:
                                if not isinstance(choice, dict):
                                    continue
                                if choice.get("finish_reason"):
                                    finish = choice["finish_reason"]
                                for holder in (choice.get("delta"), choice.get("message")):
                                    if not isinstance(holder, dict):
                                        continue
                                    if isinstance(holder.get("content"), str):
                                        text.append(holder["content"])
                                    if strip:
                                        for key in REASONING_KEYS:
                                            if (
                                                holder.pop(key, None) is not None
                                                and key not in hits
                                            ):
                                                hits.append(key)
                            out = (
                                b"data: " + json.dumps(parsed, ensure_ascii=False).encode() + b"\n"
                            )
                self._write_chunk(out)
                chunks += 1
        except (BrokenPipeError, ConnectionResetError):
            aborted = "client_gone"
        except Exception as exc:
            aborted = "upstream_error"
            logger.error("流式转发中断：{}", exc)
        finally:
            response.close()
            if aborted is None and not self._close_stream():
                aborted = "client_gone"

        self._record(
            started,
            method="POST",
            upstream_status=response.status,
            chunks=chunks,
            stripped=hits,
            finish_reason=finish,
            usage=usage,
            content="".join(text),
            aborted=aborted,
            **notes,
        )

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/health"):
            self._send(200, b'{"status":"ok"}')
            return
        started = time.monotonic()
        cfg = read_config()
        if self.path.rstrip("/").endswith("/models"):
            # nmem 凭据就绪后会查 provider 的模型列表，原样代理给上游
            status, body, content_type = forward("GET", "/models", None, cfg)
            self._send(status, body, content_type)
            self._record(started, method="GET", upstream_status=status, response=as_json(body))
            return
        self._send_error(405, f"nmem-gw: unsupported GET {self.path}")
        self._record(started, method="GET", upstream_status=405)

    def do_POST(self) -> None:
        started = time.monotonic()
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self._send_error(411, "nmem-gw: chunked request bodies are not supported")
            self._record(started, method="POST", upstream_status=411, error="chunked")
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        try:
            cfg = read_config()
        except SystemExit as exc:
            self._send_error(500, str(exc))
            self._record(started, method="POST", upstream_status=500, error="config")
            return

        rewritten, notes = rewrite(body, cfg)
        if notes.get("stream"):
            # nmem 的后台 lane 会带 stream=true，且校验响应必须是 text/event-stream，
            # 所以流式请求只能真流式转发（不能改写成一坨 JSON）
            response, status, error_body, content_type = open_upstream(
                "POST", UPSTREAM_PATH, rewritten, cfg
            )
            if response is None:
                self._send(status, error_body, content_type)
                self._record(
                    started,
                    method="POST",
                    upstream_status=status,
                    response=as_json(error_body),
                    **notes,
                )
                return
            self._proxy_stream(started, response, notes, cfg)
            return

        status, upstream_body, content_type = forward("POST", UPSTREAM_PATH, rewritten, cfg)
        payload, stripped = strip_reasoning(upstream_body)
        if not cfg.get("strip_reasoning", True):
            payload, stripped = upstream_body, []
        self._send(status, payload, content_type)
        self._record(
            started,
            method="POST",
            upstream_status=status,
            stripped=stripped,
            request=as_json(rewritten),
            response=as_json(payload),
            **notes,
        )


def main() -> None:
    cfg = read_config()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    log_path: Path | None = None
    log_cfg = dict(cfg.get("log") or {})
    if log_cfg.get("path"):
        log_path = Path(str(log_cfg["path"]))
        if not log_path.is_absolute():
            log_path = HERE / log_path
        logger.add(
            log_path,
            level="INFO",
            format="{message}",
            encoding="utf-8",
            rotation=int(log_cfg.get("max_bytes", 10 * 1024 * 1024)),
            retention=int(log_cfg.get("backups", 5)),
            enqueue=True,
        )

    listen = str(cfg.get("listen", "127.0.0.1:8899"))
    host, _, port = listen.rpartition(":")
    logger.info(
        "nmem-gw 启动：listen={} 上游={} 模型={} max_tokens={} thinking={} 日志={}",
        listen,
        cfg.get("upstream_base_url"),
        cfg.get("upstream_model"),
        cfg.get("max_tokens"),
        cfg.get("thinking"),
        log_path,
    )
    Server((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
