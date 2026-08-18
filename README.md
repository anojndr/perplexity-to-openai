# Perplexity → OpenAI-compatible API

Runs your Perplexity accounts behind a FastAPI server that speaks the OpenAI
Chat Completions **and** Responses APIs. No browser is involved at runtime:
requests go straight to Perplexity's private SSE endpoint with the session
cookies from `accounts.txt` (Chrome TLS impersonation via `curl_cffi`).

- Port: **64130**
- Endpoints: `/v1/chat/completions`, `/v1/responses`, `/v1/models`,
  `/v1/responses/{id}`, `/healthz`
- Streaming supported on both APIs (`stream: true`)
- Multi-turn: Perplexity keeps thread state server-side; the proxy maps
  conversation identity (message-history hash / `previous_response_id`) to
  Perplexity's thread handle (`backend_uuid` + `read_write_token`). Only the
  new user message is sent per turn — never the whole conversation.
- Files: images (data URL or remote URL) attach as image attachments
  (verified working); text-like files (`.py`, `.json`, `.txt`, …) are also
  inlined into the query so free-tier accounts can read them; binary files
  attach as file attachments (ingestion is gated by your Perplexity plan).
- Accounts: load-balanced, any number. Add more blocks to `accounts.txt` —
  the file is re-read automatically (mtime check), no restart needed.
  Quota-exhausted or failing accounts are skipped and cooled down.

## Setup

```bash
cd /home/sweetpotet/Desktop/perplexity-to-openai
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 64130
```

## accounts.txt format

Netscape cookie export, one block per account:

```
account 1:
# Netscape HTTP Cookie File
www.perplexity.ai	FALSE	/	FALSE	<expires>	pplx.visitor-id	...
.perplexity.ai	TRUE	/	TRUE	<expires>	__Secure-next-auth.session-token	...
account 2:
...
```

Export cookies (e.g. with a browser extension / DevTools) from
`https://www.perplexity.ai` while logged in, including the
`__Secure-next-auth.session-token` and `__Secure-pplx.session.*` cookies and
`cf_clearance`. Accounts are load-balanced round-robin with least-in-flight
preference.

## Usage

```bash
# Chat Completions
curl http://127.0.0.1:64130/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Say hi"}]}'

# Streaming
curl -N http://127.0.0.1:64130/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "stream": true,
       "messages": [{"role": "user", "content": "Explain entropy in one sentence"}]}'

# Responses API
curl http://127.0.0.1:64130/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "input": "Say hi"}'

# Multi-turn (send the full history like the OpenAI SDK does; only the new
# message is sent upstream — the proxy keeps the Perplexity thread)
curl http://127.0.0.1:64130/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [
         {"role": "user", "content": "My secret word is zebra"},
         {"role": "assistant", "content": "Got it."},
         {"role": "user", "content": "What is the secret word?"}]}'

# Image input (Chat Completions)
curl http://127.0.0.1:64130/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": [
         {"type": "text", "text": "What is in this image?"},
         {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}]}]}'

# File input (Responses API)
curl http://127.0.0.1:64130/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "input": [{"role": "user", "content": [
         {"type": "input_text", "text": "Summarize this file"},
         {"type": "input_file", "file_url": "https://example.com/report.pdf"}]}]}'

# Follow-up via previous_response_id
curl http://127.0.0.1:64130/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "previous_response_id": "<resp_id_from_previous_call>",
       "input": "And what about part two?"}'
```

## Models

`model` is mapped to Perplexity's `model_preference`:

- Aliases: `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `sonar`, `sonar-small`,
  `sonar-medium` → `turbo` (free tier); `sonar-pro` → `pplx_pro`;
  `sonar-reasoning` → `pplx_reasoning`
- Perplexity slugs pass through: `turbo`, `pplx_pro`, `pplx_pro_upgraded`,
  `pplx_reasoning`, `gpt5`, `gpt5_thinking`, `o3`, `claude40opusthinking`,
  `grok4`, `gemini2flash`, …
- Mode hints: `pplx-copilot` (default), `pplx-concise`, `pplx-internet` etc.
  select the answer mode. `copilot`/`concise` are verified on free accounts.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `PPLX_ACCOUNTS` | `accounts.txt` | account cookie file path |
| `PPLX_API_KEY` | *(none)* | if set, requires `Authorization: Bearer <key>` |
| `PPLX_MAX_CONCURRENT` | `2` | max parallel asks per account |
| `PPLX_TIMEZONE` | `UTC` | timezone reported to Perplexity |

## Behavior notes

- Free-tier daily quota exhaustion surfaces as OpenAI-style `429
  insufficient_quota`; the proxy moves on to the next healthy account
  automatically.
- `GENERIC_FAILED_RESPONSE` on a follow-up is retried once as a new thread
  (stale thread handle recovery).
- `internet` answer mode currently fails on free accounts (Perplexity-side);
  `copilot`/`concise` are the defaults.
- Thread state is in-memory (LRU 1024); restarting the server starts fresh
  conversations. Responses are retained in-memory (LRU 512) so
  `/v1/responses/{id}` and `previous_response_id` work within one process.
