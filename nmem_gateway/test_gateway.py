"""Tests for the nmem outbound gateway.

用 stdlib 假上游复现上游 schema 漂移的响应形态（usage 里给 null），让真实网关进程处理，
断言客户端拿到的是补零后的 usage。这是 2026-09-17~18 打挂 26 次后台任务的那次回归：
nmem 把 usage 的数值字段当 usize，收到 null 会整条响应 JsonError。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

GATEWAY = Path(__file__).with_name("gateway.py")
STARTUP_TIMEOUT = 30.0

# 上游 2026-09-17~18 的真实形态：数值字段给 null，且 message/delta 里带推理内容
UPSTREAM_USAGE = {
    "prompt_tokens": 13386,
    "completion_tokens": 142,
    "total_tokens": 13528,
    "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": None},
    "completion_tokens_details": {"reasoning_tokens": 0},
}


class _Upstream(BaseHTTPRequestHandler):
    """假上游：流式吐 SSE（末 chunk 带 usage），非流式吐整包 JSON。"""

    def log_message(self, *args: object) -> None:
        """不要 http.server 往 stderr 刷访问日志。"""

    def _json(self, payload: object) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else {}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {
                    "choices": [{"index": 0, "delta": {"reasoning_content": "想一下"}}],
                    "usage": None,
                },
                {"choices": [{"index": 0, "delta": {"content": "hi"}}], "usage": None},
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": UPSTREAM_USAGE,
                },
            ):
                line = json.dumps(chunk, ensure_ascii=False).encode()
                self.wfile.write(b"data: " + line + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "hi",
                            "reasoning_content": "想一下",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": UPSTREAM_USAGE,
            }
        )


@dataclass(frozen=True)
class Gateway:
    """跑起来的网关：chat completions 地址 + 它写的 JSONL 日志路径。"""

    url: str
    log_path: Path

    def post(self, payload: dict) -> bytes:
        request = urllib.request.Request(  # noqa: S310
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=STARTUP_TIMEOUT) as response:  # noqa: S310
            return response.read()

    def records(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.startswith("{")]  # 首行是启动横幅


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sse_chunks(raw: bytes) -> list[dict]:
    out: list[dict] = []
    for line in raw.decode().splitlines():
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
            out.append(json.loads(line[6:]))
    return out


@pytest.fixture(scope="module")
def gateway(tmp_path_factory: pytest.TempPathFactory) -> Gateway:
    """启动假上游 + 真实网关进程。"""
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    workdir = tmp_path_factory.mktemp("nmem-gw")
    log_path = workdir / "requests.jsonl"
    port = _free_port()
    config_path = workdir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "listen": f"127.0.0.1:{port}",
                "upstream_base_url": f"http://127.0.0.1:{upstream.server_port}",
                "upstream_api_key": "test-key",
                "upstream_headers": {"User-Agent": "nmem-gw-test"},
                "upstream_model": "test-model",
                "max_tokens": 16384,
                "thinking": None,
                "reasoning_effort": None,
                "strip_reasoning": True,
                "timeout_seconds": 10,
                "log": {"path": str(log_path), "max_bytes": 1048576, "backups": 1},
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.Popen(  # noqa: S603
        ["uv", "run", "--no-project", str(GATEWAY)],
        env={**os.environ, "GW_CONFIG": str(config_path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    gateway = Gateway(url=f"http://127.0.0.1:{port}/v1/chat/completions", log_path=log_path)
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"网关进程提前退出：{process.stdout.read().decode()}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).close()
            break
        except urllib.error.URLError, ConnectionError:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail("网关未在超时内就绪")

    yield gateway

    process.terminate()
    process.wait(timeout=10)
    upstream.shutdown()


def test_stream_fills_null_usage_and_strips_reasoning(gateway: Gateway) -> None:
    raw = gateway.post(
        {"model": "god", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    )
    chunks = _sse_chunks(raw)
    usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))
    assert usage["prompt_tokens_details"]["cache_write_tokens"] == 0
    assert usage["prompt_tokens"] == 13386  # 只补 null，别动已有数值
    assert b"reasoning_content" not in raw
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "hi"


def test_non_stream_fills_null_usage_and_strips_reasoning(gateway: Gateway) -> None:
    payload = json.loads(
        gateway.post({"model": "god", "messages": [{"role": "user", "content": "hi"}]})
    )
    assert payload["usage"]["prompt_tokens_details"]["cache_write_tokens"] == 0
    assert payload["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    assert payload["usage"]["total_tokens"] == 13528
    message = payload["choices"][0]["message"]
    assert "reasoning_content" not in message
    assert message["content"] == "hi"


def test_log_records_filled_paths(gateway: Gateway) -> None:
    """补零要留痕：日志的 filled 字段说明上游哪次又给了 null。"""
    before = len(gateway.records())
    gateway.post({"model": "god", "messages": [{"role": "user", "content": "hi"}]})
    for _ in range(50):
        records = gateway.records()
        if len(records) > before:
            assert records[-1]["filled"] == ["usage.prompt_tokens_details.cache_write_tokens"]
            return
        time.sleep(0.1)
    pytest.fail("网关没写日志")
