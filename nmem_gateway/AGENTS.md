# nmem_gateway

**类型**: 单文件 Python 脚本（PEP 723，用 uv 管理依赖：`loguru`）
**用途**: 夹在 Nowledge Mem 与上游 LLM 端点之间的 OpenAI 兼容中转网关，**单 lane**（只服务 nmem 的 `background` 用途）。

## 为什么存在

nmem 自己拥有出站请求契约，用户无法干预：provider 的 `extra_params` 不参与请求构造，`openai_compatible` 类型不能发自定义请求头（维护者原文：`There is no custom-header support on LLM providers today.`，根因是 `llm_factory` 的 reqwest client 没有 `default_headers`）。于是把请求接管到自己的脚本里，换取三件事：

1. 按配置决定**是否思考 / 思考强度**（上游原生 `thinking`、`reasoning_effort`）；
2. 注入上游要求的自定义头（如 OpenCode Go 的 `x-opencode-session`）；
3. 完整记录出入站请求体（此前完全不可观测）。

## 请求流

```
nmem(background) --HTTP--> 127.0.0.1:8899/v1/chat/completions --> 上游 LLM 端点
```

每个请求固定五步，无分支：读配置 → 覆盖 `model`/`max_tokens`/`thinking`/`reasoning_effort` → 转发 → 剥离响应里的推理内容 → 记一行 JSONL。

`GET .../models` 原样代理给上游（nmem 凭据就绪后会查模型列表）；`GET .../health` 返回 `{"status":"ok"}`。

请求有两种形态，网关都支持：

- **流式**（后台 lane 会带 `stream: true` + `stream_options.include_usage`）：上游响应用真正的 SSE
  （`text/event-stream` + chunked）逐行透传给 nmem，同时**剥掉 `delta.reasoning_content`**；
  日志里记的是重组后的 `content` / `usage` / `chunks`。
- **非流式**（工具调用类请求，如 `classify_memory_unit_type`）：读全后按 JSON 处理。

客户端中途断连（nmem 取消或超时后直接关连接）不算故障：继续写响应会抛 `BrokenPipeError`，网关在
`_close_stream()` 与 `Server.handle_error` 里吞掉 `BrokenPipeError` / `ConnectionResetError`——不刷 traceback，
记录照写，`aborted` 字段区分 `null`（正常完成）/ `client_gone`（客户端先走）/ `upstream_error`（上游中断）。

## 配置

`config.json`（**不入库、权限 600**），字段见 `config.example.json`：

| 键 | 作用 |
| --- | --- |
| `listen` | 监听地址，默认 `127.0.0.1:8899`（只对本机开放，不做鉴权） |
| `upstream_base_url` | 上游 base（脚本会拼 `/chat/completions`、`/models`） |
| `upstream_api_key` | 上游密钥 |
| `upstream_headers` | 额外请求头（如上游要求的会话头） |
| `upstream_model` | 上游真实模型名；nmem 条目里填什么都无所谓，换模型只改这里 |
| `max_tokens` | 覆盖 nmem 传来的输出预算（给思考留出内容预算）；`null` = 不动 |
| `thinking` | 原样塞进请求体的 `thinking` 对象；`null` = 删除该字段 |
| `reasoning_effort` | 覆盖思考强度（上游支持时，如 `low`/`high`/`max`）；`null` = 不动 |
| `strip_reasoning` | 是否剥离响应中的 `reasoning_content`/`reasoning`/`reasoning_details` |
| `timeout_seconds` | 上游超时 |
| `log` | `path` / `max_bytes` / `backups`（loguru 按大小轮转） |

配置**每请求重读**：改完即生效，不用重启；写坏/写一半时沿用上次成功值并告警。

### 思考开关的取值（DeepSeek 官方 schema，上游若为 Go/DeepSeek V4 即适用）

- 关：`"thinking": {"type": "disabled"}`（`reasoning_effort: "none"` 等效）
- 开：`"thinking": {"type": "enabled"}` + 可选 `"reasoning_effort": "low"|"high"|"max"`（默认 `high`；没有 `budget_tokens` 这类字段）
- 思考开启时上游默认 `max_tokens` 是 64K，所以此时把 `max_tokens` 一起调大才不至于让思考吃光可见输出

### 两条被实验证伪的做法（别再试）

- **不要把流式请求改写成非流式**：上游会先校验 `stream_options should be set along with stream = true`（400）；
  就算同时摘掉 `stream_options`，nmem 也会用 `Invalid content type was returned: "application/json"` 拒绝
  —— 它要求响应必须是 `text/event-stream`。
- **不要以为后台 lane 是非流式的**：实测 `thread_synced`、`memory_created_review` 等任务都带 `stream: true`，
  只有工具调用类请求是非流式。


## 运行与部署

- 本机冒烟：`GW_CONFIG=/tmp/gw.json uv run --no-project gateway.py`
- 服务端：脚本与配置放 `~/.local/share/nmem-gw/`，unit 放 `~/.config/systemd/user/nmem-gw.service`，然后 `systemctl --user enable --now nmem-gw`。登出后仍运行需要一次性 `loginctl enable-linger <user>`。
- 健康检查：`curl 127.0.0.1:8899/health`。
- **上游 UA 必须显式设置**：OpenCode Go 在 Cloudflare 后面，urllib 默认的 `Python-urllib/3.x` 会被挡成 `403 error code: 1010`；`upstream_headers` 里必须带 `User-Agent`（`nmem-gw/1.0` 就够）。
- 一次性 `loginctl enable-linger <user>` 需要 sudo（不带会报 `Could not enable linger: Access denied`）。
- 改完 `gateway.py` 要重装并重启：`scp gateway.py <server>:~/.local/share/nmem-gw/` + `systemctl --user restart nmem-gw`（会中断正在进行的流；配置 `config.json` 每请求重读，不用重启）。
- **nmem 侧 provider 的 `timeout` 必须 ≥ 上游最慢请求**：`max_tokens` 抬到 16384 后单请求会跑到 55–62s，provider 里写死的 `timeout: 60.0` 会在到点时断开 SSE，nmem 侧表现为 `scheduler LLM generation timed out after 60.000s` + 任务 partial。该值在服务端 `~/.config/co.nowledge.mem.desktop/remote_llm.json` 的 `providers["openai_compatible:nmem-gw"].timeout`（现为 180.0）；CLI 的 `nmem config provider set` 没有 timeout 选项，只能改文件。

## 纪律

- **禁止**读取、打印、复制 `config.json`（含上游密钥）；密钥由用户自己填。
- 日志 `requests.jsonl` 含记忆正文，属敏感资产：只在服务端本地、不备份、不入库。
- nmem 侧配套：`openai_compatible:<id>` 条目 + `background` purpose 指向它。单 lane 是刻意设计——`ai_now` 的 AgentHarness 在思考模型上另有已知问题，不在此脚本职责内。
- 上游带 `tools` 且开启思考时，DeepSeek 要求把历史轮次的 `reasoning_content` 原样传回，否则 400；本网关默认剥离响应里的推理内容，因此**不要**用它跑 tool-loop（这是它只做 background lane 的原因之一）。

## 验证与排障（踩过的坑）

- `nmem config provider test` 由**服务端**发起探测请求，且测的是 **active provider**：想用它做端到端验证，先 `nmem config provider purpose set background --provider openai_compatible:<id> --model god`，或临时 `activate`。
- CLI 上的 `--api-url` 覆盖的是 **CLI 要连的 nmem 服务器地址**，不是 provider 的 base URL。把它指向本机网关只会得到 `Connection refused`（请求根本没到服务端）。
- `nmem config provider set <新 id>` 会把该条目**设为 active 并改写 `default` purpose**（background/ai_now 靠继承跟着走）。只想加条目、不想动路由时，加完立刻 `nmem config provider activate <原 provider>` 还原。
- `provider test` 打的是 `<base>/remote-llm/test`，**不是** chat completions；想验证真实链路，用后台任务（`POST /agent/trigger/wm-refresh`）而不是 test。
- 验证 timeout 是否生效：`nmem --json config provider list` 看条目里的 `timeout`；`journalctl -u nmem.service | grep "timed out after"` 看 nmem 日志自己打印的 effective timeout（超时行里带具体秒数）。
