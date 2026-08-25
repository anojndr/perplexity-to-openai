"""OpenAI-compatible API over Perplexity (Chat Completions + Responses).

FastAPI app on port 64130. No browser: transport is curl_cffi + session
cookies. Multi-turn: Perplexity keeps thread state server-side; the proxy
maps conversation identity (message-history hash or previous_response_id)
to Perplexity's backend_uuid/read_write_token handle.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional
from urllib.parse import urlparse

from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from account_pool import AccountPool
from pplx_transport import (
    AskError,
    AskEvent,
    ThreadState,
    _attachment,
    _extract_text,
    _is_text_like,
    resolve_mode,
    resolve_model,
    TEXT_APPEND_TOTAL_CAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("pplx.server")

ACCOUNTS_FILE = os.environ.get("PPLX_ACCOUNTS", "accounts.txt")
API_KEY = os.environ.get("PPLX_API_KEY")          # optional bearer enforcement
MAX_CONCURRENT = int(os.environ.get("PPLX_MAX_CONCURRENT", "2"))
THREAD_LRU = 1024
RESPONSE_LRU = 512
FETCH_SIZE_CAP = 20 * 1024 * 1024
TZ = os.environ.get("PPLX_TIMEZONE", "UTC")
INCLUDE_SOURCES = os.environ.get("PPLX_INCLUDE_SOURCES", "0").strip().lower() in ("1", "true", "yes", "on")

pool = AccountPool(ACCOUNTS_FILE, max_concurrent=MAX_CONCURRENT)
threads: OrderedDict[str, ThreadState] = OrderedDict()   # conversation key -> Perplexity thread
threads_by_answer: OrderedDict[str, ThreadState] = OrderedDict()  # sha256(last assistant text) -> thread
responses: OrderedDict[str, dict] = OrderedDict()        # resp id -> response object


def _lru(store: OrderedDict, key: str, value=None, cap: int = THREAD_LRU):
    if key in store:
        store.move_to_end(key)
    elif value is not None:
        store[key] = value
        while len(store) > cap:
            store.popitem(last=False)
    return store.get(key)


def _register_thread(key: str, thread_state: ThreadState) -> ThreadState:
    _lru(threads, key, thread_state)
    return thread_state


def _register_answer(answer: str, thread_state: ThreadState) -> None:
    """Associate a thread with its answer text so the client's echo of that
    answer on the next turn identifies the same Perplexity conversation."""
    if answer:
        _lru(threads_by_answer, hashlib.sha256(answer.encode()).hexdigest(),
             thread_state, cap=THREAD_LRU)


def _lookup_thread(messages: list) -> tuple[Optional[ThreadState], Optional[str]]:
    """Find the Perplexity thread for a messages list.

    Strategy: the OpenAI SDK resends the full history; the message right
    before the newest user message is our previous assistant answer (echoed
    verbatim). Match on its text hash first; fall back to a hash of the
    prefix for clients that do not echo assistant messages.
    """
    last_user = None
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") in ("user", "system", "developer"):
            last_user = i
    if last_user is None:
        return None, None
    prev = messages[last_user - 1] if last_user > 0 else None
    if isinstance(prev, dict) and prev.get("role") == "assistant":
        content = prev.get("content")
        if isinstance(content, list):
            content = "".join(_text_of_part(p) for p in content)
        if isinstance(content, str) and content:
            anchor = hashlib.sha256(content.encode()).hexdigest()
            found = threads_by_answer.get(anchor)
            if found is not None:
                return found, anchor
        history = _conversation_key(messages[:last_user])
        found = threads.get(history)
        if found is not None:
            return found, None
    return None, None


def _conversation_key(history: list) -> str:
    """Hash of everything except the last user message => stable thread id."""
    h = hashlib.sha256()
    for item in history:
        h.update(json.dumps(item, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _now() -> int:
    return int(time.time())


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# Request normalization
# --------------------------------------------------------------------------

def _text_of_part(part) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    t = part.get("type", "")
    if t in ("text", "input_text"):
        return part.get("text", "")
    return ""


def _iter_parts(content) -> list:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return content
    return []


def _normalize_history(messages: list) -> tuple[list, Optional[str], str, list, Optional[str]]:
    """Split OpenAI messages into: history key items, system prompt, query,
    attachments, and text injection. Only the last user message is sent to
    Perplexity; everything earlier is represented by the thread handle."""
    history: list[dict] = []
    system: Optional[str] = None
    query_parts: list[str] = []
    attachments: list[dict] = []
    injections: list[str] = []

    def handle_content(m: dict):
        nonlocal query_parts, attachments, injections
        for part in _iter_parts(m.get("content")):
            text = _text_of_part(part)
            if text:
                query_parts.append(text)
            att = _attachment(part) if isinstance(part, dict) else None
            if att is None:
                continue
            if att["type"] == "file" and _is_text_like(att):
                content_text = _extract_text(att)
                if content_text and len(content_text) <= TEXT_APPEND_TOTAL_CAP:
                    injections.append(f"[Attached file: {att['name']}]\n```\n{content_text}\n```")
                    continue
            attachments.append(att)

    last_user_index = None
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        if role in ("user", "developer", "system"):
            last_user_index = i
    if last_user_index is None:
        return [], None, "", [], []

    for i, m in enumerate(messages[:last_user_index]):
        if not isinstance(m, dict):
            continue
        history.append({
            "role": m.get("role"),
            "content": [_text_of_part(p) for p in _iter_parts(m.get("content"))],
        })
        if m.get("role") in ("system", "developer") and system is None:
            sys_text = " ".join(filter(None, (_text_of_part(p) for p in _iter_parts(m.get("content")))))
            if sys_text:
                system = sys_text

    last = messages[last_user_index]
    handle_content(last)
    query = "\n".join(filter(None, query_parts)).strip()
    if injections:
        query = query + "\n\n" + "\n\n".join(injections) if query else "\n\n".join(injections)
    return history, system, query, attachments, None


def _normalize_responses_input(data: dict) -> tuple[list, Optional[str], str, list]:
    """Responses API input -> (history key items, system, query, attachments)."""
    inp = data.get("input", "")
    system = data.get("instructions")
    history: list[dict] = []
    query_parts: list[str] = []
    attachments: list[dict] = []
    injections: list[str] = []

    if isinstance(inp, str):
        return history, system, inp, []

    items = inp if isinstance(inp, list) else []
    user_items = [it for it in items if isinstance(it, dict) and it.get("role") in ("user", "system", "developer")]
    if not user_items and items:
        user_items = items
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") == "function_call":
            continue
        role = it.get("role")
        if role == "system" and system is None:
            system = " ".join(_text_of_part(p) for p in _iter_parts(it.get("content"))) or system
            continue
    if not user_items:
        return history, system, "", []

    for it in items:
        if not isinstance(it, dict):
            continue
        role = it.get("role")
        if role == "function_call_output":
            continue
        if role not in ("user", "system", "developer"):
            continue
        is_last = it is user_items[-1]
        for part in _iter_parts(it.get("content")):
            text = _text_of_part(part)
            if text:
                if is_last:
                    query_parts.append(text)
                else:
                    history.append({"role": role, "content": text})
            att = _attachment(part) if isinstance(part, dict) else None
            if att and is_last:
                if att["type"] == "file" and _is_text_like(att):
                    ct = _extract_text(att)
                    if ct and len(ct) <= TEXT_APPEND_TOTAL_CAP:
                        injections.append(f"[Attached file: {att['name']}]\n```\n{ct}\n```")
                        continue
                attachments.append(att)
    query = "\n".join(filter(None, query_parts)).strip()
    if injections:
        query = (query + "\n\n" + "\n\n".join(injections)) if query else "\n\n".join(injections)
    return history, system, query, attachments


def _chat_completion_response(model: str, text: str, sources: list, *, request: dict,
                              prompt_tokens: int, completion_tokens: int,
                              previous_id: Optional[str] = None) -> dict:
    annotations = _url_citations(text, sources)
    message: dict = {"role": "assistant", "content": text or None, "refusal": None}
    if annotations:
        message["annotations"] = annotations
    return {
        "id": f"chatcmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0, "audio_tokens": 0,
                "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0,
            },
        },
        "system_fingerprint": None,
        "service_tier": "default",
    }


def _url_citations(text: str, sources: list) -> list:
    out = []
    if not text or not sources:
        return out
    for src in sources[:8]:
        url = src.get("url") if isinstance(src, dict) else None
        if not url:
            continue
        title = (src.get("title") if isinstance(src, dict) else None) or url
        idx = text.find(title[:60])
        if idx < 0:
            idx = text.find(url)
        if idx < 0:
            continue
        out.append({
            "type": "url_citation",
            "start_index": idx,
            "end_index": idx + len(title[:60]),
            "url": url,
            "title": title,
        })
    return out


SOURCE_APPENDIX_MAX = 50


def _source_appendix(sources: list, query: str) -> str:
    """Bridge source appendix for llmcord-go's "Show Sources" button.

    Format mirrors grok-to-openai's source-attribution.js, which llmcord-go
    parses for any provider (FinalizeXAIResponseAnswer strips it from the
    visible answer and feeds it to Show Sources):
        \n\nSources
        1. [Title](url) (domain) via `query`

        Search Queries
        1. `query`
    """
    entries: list[str] = []
    clean_query = query.replace("`", "'") if query else ""
    for src in sources[:SOURCE_APPENDIX_MAX]:
        if not isinstance(src, dict):
            continue
        url = (src.get("url") or "").strip().replace("\n", "").replace("\r", "")
        # llmcord-go's markdown-link regex stops at ')' and whitespace.
        url = url.replace(")", "%29").replace(" ", "%20")
        if not url:
            continue
        title = ((src.get("title") or "").strip().replace("[", "").replace("]", "")) or url
        entry = f"[{title}]({url})"
        host = _host_of(url)
        if title != url and host:
            entry += f" ({host})"
        if clean_query:
            entry += f" via `{clean_query}`"
        entries.append(entry)
    if not entries:
        return ""
    lines = ["Sources"]
    lines.extend(f"{i}. {entry}" for i, entry in enumerate(entries, start=1))
    if clean_query:
        lines.append("")
        lines.append("Search Queries")
        lines.append(f"1. `{clean_query}`")
    return "\n\n" + "\n".join(lines)


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def _include_sources(flag: Optional[bool]) -> bool:
    return INCLUDE_SOURCES if flag is None else flag


def _response_object(model: str, text: str, sources: list, *, request: dict,
                     prompt_tokens: int, completion_tokens: int,
                     previous_id: Optional[str] = None,
                     resp_id: Optional[str] = None) -> dict:
    resp_id = resp_id or f"resp_{secrets.token_hex(12)}"
    annotations = _url_citations(text, sources)
    content: dict = {"type": "output_text", "text": text or "", "annotations": annotations}
    return {
        "id": resp_id,
        "object": "response",
        "created_at": _now(),
        "status": "completed",
        "completed_at": _now(),
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": request.get("instructions"),
        "max_output_tokens": request.get("max_output_tokens"),
        "max_tool_calls": None,
        "model": model,
        "output": [{
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [content],
        }],
        "parallel_tool_calls": True,
        "previous_response_id": previous_id,
        "reasoning": {"effort": None, "summary": None},
        "service_tier": "default",
        "store": bool(request.get("store", True)),
        "temperature": request.get("temperature"),
        "text": {"format": request.get("text") or {"type": "text"}},
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools", []),
        "top_logprobs": 0,
        "top_p": request.get("top_p"),
        "truncation": request.get("truncation", "disabled"),
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "user": request.get("user"),
        "metadata": request.get("metadata") or {},
    }


# --------------------------------------------------------------------------
# Perplexity ask orchestration
# --------------------------------------------------------------------------

async def _fetch_remote_to_data_url(url: str) -> str:
    """Download a remote image/file so attachments stay data URLs (verified shape)."""
    if url.startswith("data:"):
        return url
    async with AsyncSession(impersonate="chrome", timeout=30) as s:
        r = await s.get(url)
        r.raise_for_status()
        body = r.content
        if len(body) > FETCH_SIZE_CAP:
            raise ValueError("attachment too large")
        mime = r.headers.get("content-type", "application/octet-stream").split(";")[0]
        return f"data:{mime};base64," + base64.b64encode(body).decode()


async def _prepare_attachments(attachments: list) -> list:
    out = []
    for att in attachments or []:
        url = att.get("url", "")
        if url.startswith("http://") or url.startswith("https://"):
            try:
                att = dict(att, url=await _fetch_remote_to_data_url(url))
            except Exception as e:
                log.warning("attachment fetch failed (%s), dropping: %s", att.get("name"), e)
                continue
        out.append(att)
    return out


class NoHealthyAccount(Exception):
    pass


async def _stream_ask(query: str, *, model: str, mode: str, attachments: list,
                      thread: Optional[ThreadState], system: Optional[str],
                      previous_id: Optional[str]) -> AsyncIterator[AskEvent]:
    """Run one ask, yielding AskEvents as they arrive.

    Handles account selection, per-account concurrency, quota marking, and
    exactly one retry-as-new-thread on GENERIC_FAILED_RESPONSE.
    """
    attachments = await _prepare_attachments(attachments)
    # Perplexity threads are account-bound: a follow-up MUST reuse the cookies
    # of the account that owns the thread, otherwise the backend starts a
    # brand-new conversation.
    if thread is not None and thread.account_id is not None:
        picked = pool.get(thread.account_id)
        if picked is None:
            picked = await pool.pick()
    else:
        picked = await pool.pick()
    if picked is None:
        raise NoHealthyAccount()
    acct, sem = picked
    acct.active += 1
    try:
        async with sem:
            attempt = 0
            while True:
                attempt += 1
                try:
                    got_text = False
                    if thread is None or thread.backend_uuid is None:
                        thread = ThreadState(model=model, mode=mode,
                                             account_id=acct.index)
                    async for ev in acct.client.ask(
                            query, model=model, mode=mode, attachments=attachments,
                            thread=thread, system=system, timezone=TZ):
                        if ev.kind == "error":
                            code = ev.data.get("error_code", "UNKNOWN")
                            if code == "FREE_TIER_RATE_LIMITED":
                                pool.record_failure(acct, code, quota=True)
                                raise AskError(code, "Perplexity free-tier daily query limit reached", retryable=False)
                            if code == "GENERIC_FAILED_RESPONSE" and attempt == 1:
                                # Stale thread handle or transient upstream failure:
                                # retry exactly once as a brand-new thread.
                                pool.record_failure(acct, code)
                                thread = ThreadState(model=model, mode=mode,
                                                     account_id=acct.index)
                                break
                            raise AskError(code, ev.data.get("message", "Perplexity upstream error"), retryable=False)
                        if ev.kind == "text":
                            got_text = True
                        if ev.kind == "done":
                            thread = ev.data.get("thread", thread)
                        yield ev
                    else:
                        if thread.backend_uuid is None and not got_text:
                            raise AskError("NO_BACKEND_UUID", "Perplexity did not return a thread handle", retryable=False)
                        pool.record_success(acct)
                        return
                    # `break` from the error branch => retry loop continues
                except AskError as e:
                    if e.retryable and attempt == 1:
                        continue
                    raise
                except Exception as e:
                    pool.record_failure(acct, f"{type(e).__name__}: {e}")
                    raise AskError("UPSTREAM_ERROR", f"Perplexity request failed: {e}", retryable=False)
    finally:
        acct.active -= 1


async def _stream_with_accounts(query: str, **kw) -> AsyncIterator[AskEvent]:
    """Yield ask events, trying the next account on quota exhaustion."""
    tried = 0
    while True:
        try:
            async for ev in _stream_ask(query, **kw):
                yield ev
            return
        except NoHealthyAccount:
            raise AskError("NO_HEALTHY_ACCOUNT",
                           "No healthy Perplexity account available (quota or cooldown)",
                           retryable=False)
        except AskError as e:
            if e.code == "FREE_TIER_RATE_LIMITED" and tried < pool.size - 1:
                tried += 1
                continue
            raise


async def _collect_ask(query: str, **kw) -> tuple[ThreadState, str, list, list]:
    """Non-streaming variant: accumulate events into a final result."""
    text_parts: list[str] = []
    sources: list = []
    related: list = []
    thread: Optional[ThreadState] = kw.get("thread")
    async for ev in _stream_with_accounts(query, **kw):
        if ev.kind == "text":
            text_parts.append(ev.text)
        elif ev.kind == "sources":
            sources = ev.data.get("results", [])
        elif ev.kind == "related":
            related = ev.data.get("items", [])
        elif ev.kind == "done":
            thread = ev.data.get("thread", thread)
    if thread is None or thread.backend_uuid is None:
        raise AskError("NO_BACKEND_UUID", "Perplexity did not return a thread handle", retryable=False)
    return thread, "".join(text_parts), sources, related


# --------------------------------------------------------------------------
# OpenAI SSE emitters
# --------------------------------------------------------------------------

def _sse(data: dict) -> str:
    if "type" in data:
        return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"
    return f"data: {json.dumps(data)}\n\n"


async def _stream_chat_completions(gen, *, completion_id: str, model: str, created: int,
                                   prompt_tokens: int, include_usage: bool,
                                   final_text: Callable[[], str] = lambda: "") -> AsyncIterator[str]:
    base = {"id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model, "system_fingerprint": None}
    first = dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""},
                                 "logprobs": None, "finish_reason": None}])
    yield _sse(first)
    completion_tokens = 0
    async for ev in gen:
        if ev.kind == "text":
            completion_tokens += _est_tokens(ev.text)
            chunk = dict(base, choices=[{"index": 0, "delta": {"content": ev.text},
                                         "logprobs": None, "finish_reason": None}])
            yield _sse(chunk)
    tail = final_text()
    if tail:
        completion_tokens += _est_tokens(tail)
        chunk = dict(base, choices=[{"index": 0, "delta": {"content": tail},
                                     "logprobs": None, "finish_reason": None}])
        yield _sse(chunk)
    yield _sse(dict(base, choices=[{"index": 0, "delta": {},
                                    "logprobs": None, "finish_reason": "stop"}]))
    if include_usage:
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        yield _sse(dict(base, choices=[], usage=usage))
    # OpenAI chat-completions streams terminate with a literal [DONE] sentinel.
    # Clients (e.g. llmcord-go) treat EOF before [DONE] as a dropped stream.
    yield "data: [DONE]\n\n"


async def _stream_responses(gen, *, model: str, request: dict, prompt_tokens: int,
                            resp_id: Optional[str] = None, msg_id: Optional[str] = None,
                            previous_id: Optional[str] = None,
                            final_text: Callable[[list], str] = lambda s: "") -> AsyncIterator[tuple[dict, str]]:
    """Stream real-time Responses API events as AskEvents arrive from upstream."""
    resp_id = resp_id or f"resp_{secrets.token_hex(12)}"
    msg_id = msg_id or f"msg_{secrets.token_hex(12)}"
    created = _now()

    resp_in_prog = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "completed_at": None,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": request.get("instructions"),
        "max_output_tokens": request.get("max_output_tokens"),
        "max_tool_calls": None,
        "model": model,
        "output": [],
        "parallel_tool_calls": True,
        "previous_response_id": previous_id,
        "reasoning": {"effort": None, "summary": None},
        "service_tier": "default",
        "store": bool(request.get("store", True)),
        "temperature": request.get("temperature"),
        "text": {"format": request.get("text") or {"type": "text"}},
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools", []),
        "top_logprobs": 0,
        "top_p": request.get("top_p"),
        "truncation": request.get("truncation", "disabled"),
        "usage": None,
        "user": request.get("user"),
        "metadata": request.get("metadata") or {},
    }

    yield resp_in_prog, _sse({"type": "response.created", "response": resp_in_prog})
    yield resp_in_prog, _sse({"type": "response.in_progress", "response": resp_in_prog})

    output_item_in_prog = {
        "id": msg_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield resp_in_prog, _sse({"type": "response.output_item.added", "output_index": 0,
                              "item": output_item_in_prog})
    yield resp_in_prog, _sse({"type": "response.content_part.added", "item_id": msg_id,
                              "output_index": 0, "content_index": 0,
                              "part": {"type": "output_text", "text": "", "annotations": []}})

    accumulated_text: list[str] = []
    sources: list = []

    async for ev in gen:
        if ev.kind == "sources":
            sources = ev.data.get("results", [])
        elif ev.kind == "text":
            if ev.text:
                accumulated_text.append(ev.text)
                yield resp_in_prog, _sse({
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": ev.text,
                })

    appendix = final_text(sources)
    if appendix:
        accumulated_text.append(appendix)
        yield resp_in_prog, _sse({
            "type": "response.output_text.delta",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "delta": appendix,
        })

    full_text = "".join(accumulated_text)
    annotations = _url_citations(full_text, sources)
    content_part = {"type": "output_text", "text": full_text, "annotations": annotations}
    output_item_done = {
        "id": msg_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [content_part],
    }

    yield resp_in_prog, _sse({
        "type": "response.output_text.done",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_text,
        "annotations": annotations,
    })
    yield resp_in_prog, _sse({
        "type": "response.content_part.done",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": content_part,
    })
    yield resp_in_prog, _sse({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": output_item_done,
    })

    completion_tokens = _est_tokens(full_text)
    resp_completed = dict(
        resp_in_prog,
        status="completed",
        completed_at=_now(),
        output=[output_item_done],
        usage={
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )

    yield resp_completed, _sse({"type": "response.completed", "response": resp_completed})


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.start()
    log.info("accounts: %d", pool.size)
    yield
    await pool.close()


app = FastAPI(title="Perplexity OpenAI-compatible API", lifespan=lifespan)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/v1/"):
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {API_KEY}":
            return JSONResponse(status_code=401, content={
                "error": {"message": "Invalid API key", "type": "invalid_request_error",
                          "code": "invalid_api_key"}})
    return await call_next(request)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "accounts": pool.size, "accounts_detail": pool.status(),
            "threads": len(threads), "responses": len(responses)}


MODEL_LIST = [
    {"id": "turbo", "object": "model", "owned_by": "perplexity"},
    {"id": "pplx_pro", "object": "model", "owned_by": "perplexity"},
    {"id": "pplx_pro_upgraded", "object": "model", "owned_by": "perplexity"},
    {"id": "pplx_reasoning", "object": "model", "owned_by": "perplexity"},
    {"id": "gpt-4o", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "gpt-4o-mini", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "sonar", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "sonar-pro", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "sonar-reasoning", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "pplx-copilot", "object": "model", "owned_by": "perplexity-mode"},
    {"id": "pplx-concise", "object": "model", "owned_by": "perplexity-mode"},
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    messages: list = []
    stream: bool = False
    stream_options: Optional[dict] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Optional[dict] = None
    include_sources: Optional[bool] = None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    model = resolve_model(req.model)
    mode = resolve_mode(req.model, model)
    history, system, query, attachments, _ = _normalize_history(req.messages)
    if not query:
        return JSONResponse(status_code=400, content={
            "error": {"message": "No user message with content found",
                      "type": "invalid_request_error", "code": "invalid_request"}})
    key = _conversation_key(history)
    thread, answer_anchor = _lookup_thread(req.messages)

    prompt_tokens = _est_tokens(query) + sum(_est_tokens(h.get("content", "")) for h in history)

    def ask_kw():
        return dict(model=model, mode=mode, attachments=attachments,
                    thread=thread, system=system, previous_id=None)

    if not req.stream:
        try:
            thread_out, text, sources, related = await _collect_ask(query, **ask_kw())
        except AskError as e:
            raise _http_from_ask_error(e)
        _register_thread(key, thread_out)
        _register_answer(text, thread_out)
        resp = _chat_completion_response(
            model, text, sources, request=req.model_dump(),
            prompt_tokens=prompt_tokens, completion_tokens=_est_tokens(text))
        if _include_sources(req.include_sources):
            appendix = _source_appendix(sources, query)
            if appendix:
                resp["choices"][0]["message"]["content"] = (text or "") + appendix
        return resp

    include_usage = bool((req.stream_options or {}).get("include_usage"))
    completion_id = f"chatcmpl-{secrets.token_hex(12)}"
    created = _now()

    async def sse_gen():
        gen = _stream_with_accounts(query, **ask_kw())
        try:
            thread_out = None
            text_parts: list[str] = []
            sources: list = []

            def appendix() -> str:
                return _source_appendix(sources, query) if _include_sources(req.include_sources) else ""

            async def forward() -> AsyncIterator[AskEvent]:
                nonlocal thread_out, text_parts, sources
                async for ev in gen:
                    if ev.kind == "text":
                        text_parts.append(ev.text)
                    elif ev.kind == "sources":
                        sources = ev.data.get("results", [])
                    elif ev.kind == "done":
                        thread_out = ev.data.get("thread")
                    yield ev

            async for chunk in _stream_chat_completions(
                    forward(), completion_id=completion_id, model=model,
                    created=created, prompt_tokens=prompt_tokens,
                    include_usage=include_usage, final_text=appendix):
                yield chunk
            if thread_out is not None:
                _register_thread(key, thread_out)
                _register_answer("".join(text_parts), thread_out)
        except AskError as e:
            yield _sse_error(e)

    return StreamingResponse(sse_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse_error(e: AskError) -> str:
    body = {"error": {"message": e.message, "type": "upstream_error",
                      "code": e.code.lower()}}
    return _sse({"type": "error", "error": body["error"]})


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    input: object = ""
    instructions: Optional[str] = None
    stream: bool = False
    previous_response_id: Optional[str] = None
    max_output_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Optional[dict] = None
    store: Optional[bool] = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: list = []
    tool_choice: object = "auto"
    text: object = None
    reasoning: object = None
    truncation: object = "disabled"
    include_sources: Optional[bool] = None


def _http_from_ask_error(e: AskError) -> HTTPException:
    if e.code == "FREE_TIER_RATE_LIMITED" or e.code == "NO_HEALTHY_ACCOUNT":
        status = 429
    elif e.code in ("NO_BACKEND_UUID", "UPSTREAM_ERROR", "GENERIC_FAILED_RESPONSE"):
        status = 502
    else:
        status = 502
    return HTTPException(status_code=status, detail={
        "message": e.message, "type": "insufficient_quota" if status == 429 else "upstream_error",
        "code": "rate_limit_exceeded" if status == 429 else e.code.lower()})


@app.post("/v1/responses")
async def create_response(req: ResponsesRequest, request: Request):
    model = resolve_model(req.model)
    mode = resolve_mode(req.model, model)
    history, system, query, attachments = _normalize_responses_input(req.model_dump())
    if req.instructions:
        system = req.instructions if system is None else f"{req.instructions}\n\n{system}"
    if not query:
        return JSONResponse(status_code=400, content={
            "error": {"message": "No user message with content found",
                      "type": "invalid_request_error", "code": "invalid_request"}})

    thread: Optional[ThreadState] = None
    previous_id = req.previous_response_id
    if previous_id:
        prev = responses.get(previous_id)
        if prev is None:
            return JSONResponse(status_code=400, content={
                "error": {"message": "Unknown previous_response_id",
                          "type": "invalid_request_error",
                          "code": "invalid_previous_response_id"}})
        thread = prev.get("_thread")
    else:
        raw_items = req.input if isinstance(req.input, list) else []
        thread, _ = _lookup_thread(raw_items)
        if thread is None:
            key = _conversation_key(history)
            thread = _lru(threads, key)

    prompt_tokens = _est_tokens(query) + sum(_est_tokens(h.get("content", "")) for h in history)

    def ask_kw():
        return dict(model=model, mode=mode, attachments=attachments,
                    thread=thread, system=system, previous_id=previous_id)

    if not req.stream:
        try:
            thread_out, text, sources, related = await _collect_ask(query, **ask_kw())
        except AskError as e:
            raise _http_from_ask_error(e)

        if thread is None:
            key = _conversation_key(history)
            _register_thread(key, thread_out)
        _register_answer(text, thread_out)
        resp = _response_object(model, text, sources, request=req.model_dump(),
                                prompt_tokens=prompt_tokens,
                                completion_tokens=_est_tokens(text),
                                previous_id=previous_id)
        if _include_sources(req.include_sources):
            appendix = _source_appendix(sources, query)
            if appendix:
                resp["output"][0]["content"][0]["text"] = (text or "") + appendix
        resp["_thread"] = thread_out
        _lru(responses, resp["id"], resp, cap=RESPONSE_LRU)
        out = dict(resp)
        out.pop("_thread", None)
        return out

    resp_id = f"resp_{secrets.token_hex(12)}"
    msg_id = f"msg_{secrets.token_hex(12)}"

    async def sse_gen():
        gen = _stream_with_accounts(query, **ask_kw())
        try:
            thread_out = None

            def appendix(srcs: list) -> str:
                return _source_appendix(srcs, query) if _include_sources(req.include_sources) else ""

            async def forward() -> AsyncIterator[AskEvent]:
                nonlocal thread_out
                async for ev in gen:
                    if ev.kind == "done":
                        thread_out = ev.data.get("thread")
                    yield ev

            completed_resp = None
            async for resp_obj, sse_chunk in _stream_responses(
                    forward(), model=model, request=req.model_dump(),
                    prompt_tokens=prompt_tokens, resp_id=resp_id,
                    msg_id=msg_id, previous_id=previous_id,
                    final_text=appendix):
                completed_resp = resp_obj
                yield sse_chunk

            if completed_resp is not None:
                final_stored = dict(completed_resp)
                final_stored["_thread"] = thread_out
                _lru(responses, resp_id, final_stored, cap=RESPONSE_LRU)

            if thread_out is not None:
                if thread is None:
                    key = _conversation_key(history)
                    _register_thread(key, thread_out)
                if completed_resp:
                    out_text = ""
                    try:
                        out_text = completed_resp["output"][0]["content"][0].get("text", "")
                    except Exception:
                        pass
                    _register_answer(out_text, thread_out)
        except AskError as e:
            yield _sse_error(e)

    return StreamingResponse(sse_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    resp = responses.get(response_id)
    if resp is None:
        raise HTTPException(status_code=404, detail={
            "message": f"No response found with id {response_id}",
            "type": "invalid_request_error", "code": "response_not_found"})
    out = dict(resp)
    out.pop("_thread", None)
    return out


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": MODEL_LIST}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    for m in MODEL_LIST:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail={
        "message": f"The model '{model_id}' does not exist", "type": "invalid_request_error",
        "code": "model_not_found"})
