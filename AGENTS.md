# Repository Guidelines

## Project Overview

FastAPI proxy on port `64130` exposing Perplexity accounts as OpenAI-compatible APIs.
No browser at runtime — `POST https://www.perplexity.ai/rest/sse/perplexity_ask` via `curl_cffi` Chrome impersonation + session cookies from `accounts.txt`.

Endpoints in `server.py`: `POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/responses/{id}`, `GET /v1/models[/{id}]`, `GET /healthz`. Streaming + multi-turn (server-side Perplexity threads) + image/file inputs supported.

## Architecture & Data Flow

Stack: `server.py` (facade) → `account_pool.py` (`AccountPool`) → `pplx_transport.py` (`PerplexityClient`) → Perplexity SSE. `db_store.py` (`Store` → `state.db` SQLite) persists threads/responses.

Request lifecycle (`server.py`):

1. `api_key_middleware` — optional `PPLX_API_KEY` Bearer on `/v1/*`. Pydantic `ChatRequest`/`ResponsesRequest` with `extra=ignore`.
2. Normalize — `_normalize_history` / `_normalize_responses_input`: last user message → `query`; earlier turns → `history` items; first system/developer → `system`. Attachments via `_attachment`; text-like files inlined as `[Attached file]` (cap `TEXT_APPEND_TOTAL_CAP`); remote `http(s)` URLs fetched to data-URLs (`FETCH_SIZE_CAP=20MB`).
3. Thread resolve — chat: `_lookup_thread` = sha256(echoed assistant answer) → `store.get_thread_by_answer`, fallback `_conversation_key(history prefix)` → `store.get_thread`. Responses: `previous_response_id` → `store.get_response()._thread`, else same fallback.
4. Ask — `_stream_ask` → `_stream_with_accounts` → `_collect_ask`: sticky `pool.get(thread.account_id)` else `pool.pick()`; per-account `asyncio.Semaphore`; `acct.client.ask(...)`. One retry-as-new-thread on `GENERIC_FAILED_RESPONSE`; `FREE_TIER_RATE_LIMITED` rotates accounts; `NoHealthyAccount` → 429.
5. Transport (`PerplexityClient.ask` in `pplx_transport.py`): `build_payload` (`query_source: followup|home`, `last_backend_uuid`/`read_write_token` on follow-ups); SSE `data:` parse → `AskEvent` (text `blocks`/`diff_block`, `sources_answer_mode.web_results`, `related_query_items`) → `done` + `ThreadState`.
6. Respond — `_chat_completion_response` / `_response_object` (usage `_est_tokens=len//4`, `_url_citations`, optional `_source_appendix` if `include_sources` / `PPLX_INCLUDE_SOURCES=1`); `store.save_response`; register `_register_thread` + `_register_answer`. Errors: `_http_from_ask_error` (429 quota/no-account, else 502), `_sse_error` mid-stream.

## Key Directories

Flat root — no `src/`, `tests/`, `pkg/`:

- `./` — all source: `server.py`, `pplx_transport.py`, `account_pool.py`, `db_store.py`.
- `./` (ops): `restart.sh`, `server.log`, `state.db`, `accounts.txt`, `README.md`, `pyproject.toml`, `requirements.txt`, `uv.lock`.
- Ignored/ephemeral: `__pycache__/`, `.ruff_cache/`, `.venv/`, `.jspace/`.

Do not add nested source dirs without reason; keep 4-module flat layout.

## Development Commands

Setup (canonical, `README.md:28-29`):

```bash
cd /home/sweetpotet/Desktop/perplexity-to-openai
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
uvicorn server:app --host 0.0.0.0 --port 64130
./restart.sh            # kill pgrep -f 'uvicorn.*server:app', nohup >> server.log, 30x curl /healthz
tail -f server.log
curl -fsS http://127.0.0.1:64130/healthz
```

Smoke (from `README.md`):

```bash
curl http://127.0.0.1:64130/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Say hi"}]}'
curl -N http://127.0.0.1:64130/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "stream": true, "messages": [{"role": "user", "content": "Hi"}]}'
curl http://127.0.0.1:64130/v1/responses -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "input": "Say hi"}'
```

Typecheck (only declared gate, unwired):

```bash
uv run --group dev ty check  # or: uvx ty check
```

No build step, no `[project.scripts]`, no Makefile/CI. Never run via `python server.py` — no `__main__`/`uvicorn.run`; entry is always `server:app`.

## Code Conventions & Common Patterns

- **Async-first, sync-isolated:** handlers/orchestration `async`; blocking SQLite via `asyncio.to_thread(store.*)`; `AccountPool._lock: asyncio.Lock`; per-account `asyncio.Semaphore(PPLX_MAX_CONCURRENT=2)`.
- **Fail-open persistence:** `db_store._safe` catches only `sqlite3.Error`, logs, returns default. Storage never fails a request.
- **Defensive untrusted-input typing:** pervasive `isinstance` narrowing on `Any` from Pydantic `extra=ignore` / Perplexity SSE before use. Follow it for new payload handling.
- **Error taxonomy drives behavior:** `AskError.code` (`FREE_TIER_RATE_LIMITED`, `GENERIC_FAILED_RESPONSE`, `NO_HEALTHY_ACCOUNT`, `HTTP_*`, `UPSTREAM_ERROR`) → rotation vs single new-thread retry vs 429/502 mapping. Preserve code strings.
- **DI = module singletons:** `pool = AccountPool(...)` + `store = Store()` in `server.py`, imported directly. `PerplexityClient` per-`Account`, reused in `AccountPool.reload()` if cookies unchanged. No framework.
- **State:** `ThreadState` (`backend_uuid`/`read_write_token`/`slug`/`account_id` + `touch()`); `Account` (`active`/`failures`/`cooldown_until`/`quota_known`); `Store` tables `threads` / `threads_by_answer` / `responses` with LRU `THREAD_LRU=1024` / `RESPONSE_LRU=512`, WAL + `synchronous=NORMAL`.
- **Naming/formatting:** `from __future__ import annotations`, `snake_case`, `_private` prefix, `log = logging.getLogger("pplx.*")`, `TEXT_*_CAP` bounds, `Optional[X]` style. Pool consts: `QUOTA_TTL=120`, `COOLDOWN=300`, `FAIL_THRESHOLD=3`.
- **Model/mode resolution** (`pplx_transport.py`): `MODEL_ALIASES`/`KNOWN_MODELS`/`MODES`, `resolve_model`/`resolve_mode`. `gpt-4o/sonar→turbo`, `sonar-pro→pplx_pro`; `pplx-copilot` default, `pplx-concise`/`pplx-internet` hints (`internet` fails on free tier).

## Important Files

- `server.py` (~1448 lines) — app, `lifespan`, routes, `_normalize_*`, `_stream_ask`/`_stream_with_accounts`/`_collect_ask`, SSE emitters, `_url_citations`/`_source_appendix`, `pool`/`store` singletons. Env: `PPLX_ACCOUNTS|PPLX_API_KEY|PPLX_MAX_CONCURRENT|PPLX_TIMEZONE|PPLX_INCLUDE_SOURCES`.
- `pplx_transport.py` (~717 lines) — `PerplexityClient` (`session`/`quota_available`/`ask`), `build_payload`, `ThreadState`/`AskEvent`/`AskError`, `ASK_URL`/`SESSION_URL`/`RATE_LIMIT_URL`.
- `account_pool.py` (~230 lines) — `parse_accounts`, `AccountPool` (`start`/`reload` mtime-based/`pick` least-active+round-robin/`get` sticky/`record_success|failure`/`status`/`close`).
- `db_store.py` (~334 lines) — `Store` (`get|save_thread`, `get|save_thread_by_answer`, `get|save_response`, `_evict`), `_safe`, `DB_PATH=PPLX_DB_PATH or state.db`.
- `pyproject.toml` — `requires-python>=3.12`, 4 runtime deps, `[tool.uv] package=false`, strict `[tool.ty.*]`.
- `requirements.txt` / `uv.lock` — pip mirror (4 lines) / pinned resolver. Keep in sync.
- `restart.sh` — kill/start/healthcheck wrapper; keep `pgrep -f 'uvicorn.*server:app'`, `0.0.0.0:64130`, `server.log`, `/healthz` loop intact.
- `README.md` — setup/curl/env-var/behavior source of truth.
- `accounts.txt` — live secrets, Netscape TSV `account N:` blocks. NEVER `read`/`grep`/`cat` values, never paste into prompts/diffs/logs. Safe signal: `accounts loaded: N` in `server.log`.
- `state.db` — SQLite WAL persistence. Do not dump; inspect via `/v1/responses/{id}` or schema-aware read only.
- `.gitignore` — bans `accounts*.txt`, `*.cookies|*.cookie|*.har`, `.venv/`, `*.log`, `*.db*`, `.jspace/`. Never `git add -f` these.

## Runtime/Tooling Preferences

- **Python `>=3.12` only** (confirmed by `__pycache__/*.cpython-312.pyc`). No Bun/Node/Docker.
- **Package managers:** `pip` (documented: `pip install -r requirements.txt`) + `uv` (`uv.lock` v1 rev 3, `[dependency-groups] dev=[ty>=0.0.78]`). If adding a dep, update `pyproject.toml` + `requirements.txt` + relock.
- **`package=false`** — run-from-source proxy, not distributable. No `[project.scripts]`/build backend.
- **`ty` is deny-all:** `[tool.ty.rules] all="error"` + `error-on-warning=true` + strict equality/generics. New code must be `ty`-clean; do not weaken config.
- Always use `https://docs.astral.sh/uv/` with everything enabled and `https://docs.astral.sh/ty/` with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them.
- **No ruff/pytest/mypy config.** Stale `.ruff_cache/` is ephemeral (own `.gitignore=*`); never commit it or `__pycache__/`.
- **Env vars:** `PPLX_ACCOUNTS=accounts.txt`, `PPLX_API_KEY` (unset=auth off), `PPLX_MAX_CONCURRENT=2`, `PPLX_TIMEZONE=UTC`, `PPLX_INCLUDE_SOURCES=0`, `PPLX_DB_PATH=state.db`. Keep `PPLX_DB_PATH` override + WAL + `_safe` semantics + LRU caps.

## Testing & QA

No tests exist — no `tests/`, `test_*.py`, `conftest.py`, pytest/unittest deps, coverage config, or CI workflows (verified via glob + grep + `pyproject.toml`/`uv.lock`/`README.md`).

- **Framework:** none. Future layout would be greenfield (`tests/` + `test_*.py` + `conftest.py`).
- **Current verification:** manual `curl` probes (`README.md:52-106`) + `./restart.sh` readiness (`curl -fsS .../healthz` ×30) + `tail -f server.log`. Healthy log shows `accounts loaded: 3`, `Uvicorn running on ...64130`, `GET /healthz 200`.
- **Coverage:** 0% / unenforced. No `[tool.coverage]`/`--cov`/codecov.
- **Lint:** none configured. Sole static bar is strict `ty` (unwired, no pre-commit/CI/README mention).
- Do not claim `pytest`/`ruff` commands work. For changes, verify with `ty check` (if available) + `./restart.sh` + `curl /healthz` + endpoint `curl` for the touched path.
