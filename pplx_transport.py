"""Perplexity private-API transport (no browser).

Ground truth for this protocol was captured live from the web app (2026-08):
POST https://www.perplexity.ai/rest/sse/perplexity_ask streams SSE JSON events.
Multi-turn continuity is server-side: follow-ups carry last_backend_uuid +
read_write_token (both returned in the first event) and the thread's URL slug
as Referer. Works with curl_cffi Chrome impersonation + session cookies.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional

from curl_cffi.requests import AsyncSession

log = logging.getLogger("pplx")

BASE_URL = "https://www.perplexity.ai"
ASK_URL = f"{BASE_URL}/rest/sse/perplexity_ask"
SESSION_URL = f"{BASE_URL}/api/auth/session"
RATE_LIMIT_URL = f"{BASE_URL}/rest/rate-limit/status"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

# Free-plan-verified modes. "internet" mode fails with GENERIC_FAILED_RESPONSE
# on free accounts but works on Pro; users can opt into it via model suffix.
MODES = {"concise", "copilot", "internet", "pro", "academic", "writing", "math"}

# Requested-model aliases -> Perplexity model_preference. Anything else is
# passed through as-is (Perplexity slugs like gpt5, claude40opusthinking...).
MODEL_ALIASES = {
    "gpt-4o": "turbo",
    "gpt-4o-mini": "turbo",
    "gpt-4.1": "turbo",
    "gpt-4.1-mini": "turbo",
    "gpt-4.1-nano": "turbo",
    "gpt-4": "turbo",
    "gpt-3.5-turbo": "turbo",
    "sonar": "turbo",
    "sonar-small": "turbo",
    "sonar-medium": "turbo",
    "sonar-pro": "pplx_pro",
    "sonar-reasoning": "pplx_reasoning",
    "pplx": "pplx_pro",
}
KNOWN_MODELS = [
    "turbo", "pplx_pro", "pplx_pro_upgraded", "pplx_alpha", "pplx_beta",
    "pplx_reasoning", "pplx_research", "pplx_research_upgraded",
    "gpt41", "gpt5", "gpt5_thinking", "o3", "o3pro", "o4mini", "o1",
    "claude2", "claude37sonnetthinking", "claude40opus", "claude40opusthinking",
    "claude41opusthinking", "claude45sonnet", "claude45sonnetthinking",
    "experimental", "grok", "grok4", "gemini2flash", "gemini", "mistral",
    "llama_x_large", "r1",
]

CHUNK_RE = re.compile(r"^/chunks/(\d+)$")


def _iter_text_payloads(val) -> Iterator[tuple[Optional[str], Optional[list]]]:
    """Recursively find {text_payload: {text, chunks}} dicts anywhere in a
    patch value (instant answers arrive as one nested workflow tree)."""
    if isinstance(val, dict):
        tp = val.get("text_payload")
        if isinstance(tp, dict):
            chunks = tp.get("chunks")
            yield tp.get("text"), chunks if isinstance(chunks, list) else None
            return
        for v in val.values():
            yield from _iter_text_payloads(v)
    elif isinstance(val, list):
        for v in val:
            yield from _iter_text_payloads(v)


@dataclass
class ThreadState:
    """Server-side Perplexity conversation handle + cached session metadata."""
    backend_uuid: Optional[str] = None
    read_write_token: Optional[str] = None
    slug: Optional[str] = None
    title: Optional[str] = None
    model: Optional[str] = None
    mode: str = "copilot"
    account_id: Optional[int] = None
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()


@dataclass
class AskEvent:
    kind: str            # "meta" | "text" | "sources" | "related" | "error" | "done"
    text: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class AskError(Exception):
    code: str
    message: str
    retryable: bool = False  # True => safe to retry as a brand-new thread


def resolve_model(requested: Optional[str]) -> str:
    if not requested:
        return "turbo"
    name = requested.strip()
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if name in KNOWN_MODELS:
        return name
    if name.startswith("pplx-"):
        return name[5:]
    # Unknown names default to the free-tier model rather than erroring.
    return "turbo"


def resolve_mode(requested: Optional[str], model: str) -> str:
    """Mode comes from model hints (pplx-copilot / pplx-concise) or defaults."""
    if requested:
        low = requested.lower()
        for m in MODES:
            if low.endswith(f"-{m}") or low == m:
                return m
    return "copilot"


def _attachment(part: dict) -> Optional[dict]:
    """Normalize an OpenAI content part into a Perplexity attachment object."""
    ptype = part.get("type", "")
    if ptype == "image_url":
        url = part.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if not url:
            return None
        mime = "image/png"
        if isinstance(url, str):
            if url.startswith("data:"):
                mime = url.split(";")[0][5:] or "image/png"
            else:
                mime = "image/jpeg"
        return {"type": "image", "content_type": mime,
                "name": "image", "url": url, "size": len(url)}
    if ptype in ("input_image", "image"):
        url = part.get("image_url") or part.get("url")
        if isinstance(url, dict):
            url = url.get("url")
        if not url:
            return None
        mime = "image/png"
        if isinstance(url, str):
            if url.startswith("data:"):
                mime = url.split(";")[0][5:] or "image/png"
            else:
                mime = "image/jpeg"
        return {"type": "image", "content_type": mime,
                "name": "image", "url": url, "size": len(url)}
    if ptype in ("input_file", "file"):
        url = part.get("file_url") or part.get("url")
        name = part.get("filename") or part.get("name") or "file"
        if isinstance(url, dict):
            url = url.get("url")
        data = part.get("file_data")  # base64
        if isinstance(data, str) and data:
            mime = part.get("content_type") or "application/octet-stream"
            url = f"data:{mime};base64,{data}"
        if not url:
            return None
        mime = "application/octet-stream"
        if isinstance(url, str) and url.startswith("data:"):
            mime = url.split(";")[0][5:] or mime
        return {"type": "file", "content_type": mime, "name": name,
                "url": url, "size": len(url)}
    return None


_TEXT_MIMES = {
    "text/", "application/json", "application/xml", "application/javascript",
    "application/x-python", "application/x-sh", "application/x-yaml",
    "application/sql", "application/csv",
}
_TEXT_EXT = {".txt", ".py", ".js", ".ts", ".tsx", ".json", ".md", ".markdown",
             ".csv", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
             ".sh", ".bash", ".zsh", ".fish", ".sql", ".html", ".htm", ".css",
             ".log", ".tex", ".r", ".rb", ".go", ".rs", ".java", ".c", ".h",
             ".cpp", ".hpp", ".cs", ".php", ".lua", ".pl", ".kt", ".kts",
             ".swift", ".scala", ".dockerfile", ".env", ".gitignore", ".lock",
             ".diff", ".patch", ".ipynb", ".gradle", ".properties", ".proto",
             ".vue", ".svelte", ".astro", ".rss", ".atom"}

TEXT_APPEND_CAP = 64 * 1024  # per file
TEXT_APPEND_TOTAL_CAP = 32 * 1024  # total injected text into the query


def _is_text_like(att: dict) -> bool:
    mime = att.get("content_type", "")
    name = att.get("name", "")
    if any(mime.startswith(p) for p in _TEXT_MIMES):
        return True
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return ext in _TEXT_EXT


def _extract_text(att: dict) -> Optional[str]:
    """Decode text content from a data-URL attachment (bounded)."""
    url = att.get("url", "")
    if not url.startswith("data:"):
        return None
    try:
        head, _, payload = url.partition(",")
        if ";base64" in head:
            import base64
            raw = base64.b64decode(payload[:TEXT_APPEND_CAP])
        else:
            raw = payload.encode()[:TEXT_APPEND_CAP]
        return raw.decode("utf-8", "replace")
    except Exception:
        return None


def build_payload(query: str, *, frontend_uuid: str, frontend_context_uuid: str,
                  model: str, mode: str, attachments: Optional[list] = None,
                  thread: Optional[ThreadState] = None,
                  system: Optional[str] = None, timezone: str = "UTC") -> dict:
    if system:
        query = f"{system}\n\n{query}"
    params = {
        "attachments": attachments or [],
        "language": "en-US",
        "timezone": timezone,
        "search_focus": "internet",
        "sources": ["web"],
        "search_recency_filter": None,
        "frontend_uuid": frontend_uuid,
        "mode": mode,
        "model_preference": model,
        "is_related_query": False,
        "is_sponsored": False,
        "frontend_context_uuid": frontend_context_uuid,
        "prompt_source": "user",
        "query_source": "followup" if thread else "home",
        "is_incognito": False,
        "time_from_first_type": 0,
        "local_search_enabled": False,
        "use_schematized_api": True,
        "send_back_text_in_streaming_api": False,
        "supported_block_use_cases": [
            "answer_modes", "media_items", "knowledge_cards", "inline_entity_cards",
            "place_widgets", "finance_widgets", "sports_widgets", "news_widgets",
            "shopping_widgets", "jobs_widgets", "search_result_widgets",
            "inline_images", "inline_assets", "placeholder_cards", "diff_blocks",
            "inline_knowledge_cards", "entity_group_v2", "refinement_filters",
            "canvas_mode", "maps_preview", "answer_tabs", "price_comparison_widgets",
            "preserve_latex", "generic_onboarding_widgets", "in_context_suggestions",
            "pending_followups", "inline_claims", "unified_assets", "workflow_steps",
            "workflow_widgets", "navigation_results", "background_agents",
        ],
        "client_coordinates": None,
        "mentions": [],
        "dsl_query": query,
        "skip_search_enabled": True,
        "is_nav_suggestions_disabled": False,
        "source": "default",
        "always_search_override": False,
        "override_no_search": False,
        "should_ask_for_mcp_tool_confirmation": True,
        "supports_tool_approval_modal": True,
        "browser_agent_allow_once_from_toggle": False,
        "force_enable_browser_agent": False,
        "supported_features": ["browser_agent_permission_banner_v1.1"],
        "extended_context": False,
        "version": "2.18",
    }
    if thread is not None and thread.backend_uuid:
        params.update({
            "last_backend_uuid": thread.backend_uuid,
            "read_write_token": thread.read_write_token,
            "followup_source": "link",
        })
    return {"params": params, "query_str": query}


class PerplexityClient:
    def __init__(self, cookies: dict, *, impersonate: str = "chrome",
                 max_concurrent: int = 2, timeout: float = 180.0):
        self._cookies = dict(cookies)
        self._impersonate = impersonate
        self._timeout = timeout
        self._session: Optional[AsyncSession] = None

    async def session(self) -> AsyncSession:
        if self._session is None:
            self._session = AsyncSession(
                impersonate=self._impersonate,
                headers={
                    "User-Agent": UA,
                    "Origin": BASE_URL,
                    "Accept": "text/event-stream",
                    "x-perplexity-request-reason": "perplexity-query-state-provider",
                },
                cookies=self._cookies,
                timeout=self._timeout,
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def check_session(self) -> Optional[str]:
        """Return user id if the cookie session is valid, else None."""
        try:
            s = await self.session()
            r = await s.get(SESSION_URL, timeout=15)
            if r.status_code != 200:
                return None
            user = r.json().get("user")
            return user.get("id") if user else None
        except Exception:
            return None

    async def quota_available(self) -> Optional[bool]:
        """True/False if known, None when the status endpoint is unreachable."""
        try:
            s = await self.session()
            r = await s.get(RATE_LIMIT_URL, timeout=15)
            if r.status_code != 200:
                return None
            fq = r.json().get("free_queries") or {}
            return bool(fq.get("available"))
        except Exception:
            return None

    async def ask(self, query: str, *, model: str, mode: str,
                  attachments: Optional[list] = None,
                  thread: Optional[ThreadState] = None,
                  system: Optional[str] = None,
                  timezone: str = "UTC") -> AsyncIterator[AskEvent]:
        """Stream one Perplexity ask. Yields text/sources/related/error events."""
        frontend_uuid = str(uuid.uuid4())
        frontend_context_uuid = str(uuid.uuid4())
        if thread is None:
            thread = ThreadState(model=model, mode=mode)
        body = build_payload(
            query, frontend_uuid=frontend_uuid,
            frontend_context_uuid=frontend_context_uuid,
            model=model, mode=mode, attachments=attachments,
            thread=thread if thread.backend_uuid else None,
            system=system, timezone=timezone,
        )
        s = await self.session()
        referer = f"{BASE_URL}/search/{thread.slug}" if thread.slug else f"{BASE_URL}/"
        headers = {"Referer": referer, "x-request-id": str(uuid.uuid4())}

        # Streaming text reconstruction.
        # Segments arrive as per-index patches (/chunks/N) plus an
        # authoritative full text (.../text_payload/text). Follow-up answers
        # can skip index 0 (segment 1 arrives first), so we only emit once a
        # contiguous run from index 0 exists; otherwise we wait for the full
        # text. This guarantees deltas always concatenate to the final answer.
        segments: dict[int, str] = {}
        next_idx = 0
        pending: list[str] = []
        full_text: Optional[str] = None
        emitted = ""

        def take(idx: int, val: str) -> None:
            nonlocal next_idx, emitted
            if not val:
                return
            segments[idx] = val
            while next_idx in segments:
                seg = segments.pop(next_idx)
                next_idx += 1
                pending.append(seg)
                emitted += seg

        def reconcile(event_delta: str) -> str:
            nonlocal emitted, full_text
            if full_text is None:
                return event_delta
            if full_text.startswith(emitted):
                tail = full_text[len(emitted):]
                if tail and not event_delta.endswith(tail):
                    event_delta += tail
            elif not full_text.startswith(event_delta):
                event_delta += full_text
            emitted = full_text
            full_text = None
            return event_delta

        async with s.stream("POST", ASK_URL, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                yield AskEvent("error", data={
                    "error_code": f"HTTP_{resp.status_code}",
                    "message": f"Perplexity returned HTTP {resp.status_code}",
                    "retryable": resp.status_code >= 500,
                })
                return
            async for raw in resp.aiter_lines():
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue

                if ev.get("error_code"):
                    code = ev["error_code"]
                    retryable = code in ("GENERIC_FAILED_RESPONSE",)
                    yield AskEvent("error", data={
                        "error_code": code,
                        "message": ev.get("text") or "Perplexity upstream error",
                        "retryable": retryable,
                    })
                    return

                if ev.get("final_sse_message") is False or "blocks" in ev:
                    for key in ("backend_uuid", "read_write_token",
                                "thread_url_slug", "thread_title",
                                "display_model", "mode"):
                        val = ev.get(key)
                        if val and not getattr(thread, key, None):
                            if key == "thread_url_slug":
                                thread.slug = val
                            elif key == "display_model":
                                thread.model = val
                            else:
                                setattr(thread, key, val)
                    if ev.get("backend_uuid"):
                        thread.backend_uuid = ev["backend_uuid"]
                    if ev.get("read_write_token"):
                        thread.read_write_token = ev["read_write_token"]
                    if ev.get("thread_url_slug"):
                        thread.slug = ev["thread_url_slug"]
                    if ev.get("thread_title"):
                        thread.title = ev["thread_title"]

                    for block in ev.get("blocks", []):
                        usage = block.get("intended_usage")
                        if usage == "sources_answer_mode":
                            src = (block.get("sources_mode_block") or {}).get("web_results") or []
                            if src:
                                yield AskEvent("sources", data={"results": src})
                            continue
                        db = block.get("diff_block") or {}
                        field = db.get("field")
                        if field == "markdown_block":
                            for patch in db.get("patches", []):
                                path = patch.get("path")
                                if path == "/progress":
                                    continue
                                val = patch.get("value")
                                if isinstance(val, dict):
                                    cs = val.get("chunks")
                                    if isinstance(cs, list):
                                        for i, c in enumerate(cs):
                                            take(i, c if isinstance(c, str) else "")
                                    elif isinstance(val.get("answer"), str):
                                        take(0, val["answer"])
                                elif isinstance(val, str):
                                    m = CHUNK_RE.fullmatch(path or "")
                                    if m:
                                        take(int(m.group(1)), val)
                        elif field == "workflow_block":
                            # Answer text streams inside workflow step
                            # payloads: .../text_payload/text (full text) and
                            # .../text_payload/chunks/N (segment deltas).
                            for patch in db.get("patches", []):
                                path = patch.get("path") or ""
                                val = patch.get("value")
                                if isinstance(val, dict):
                                    for full, cs in _iter_text_payloads(val):
                                        if isinstance(full, str) and full:
                                            full_text = full
                                        if cs:
                                            for i, c in enumerate(cs):
                                                take(i, c if isinstance(c, str) else "")
                                elif isinstance(val, str):
                                    if path.endswith("/text_payload/text"):
                                        full_text = val
                                    else:
                                        m = re.search(r"/text_payload/chunks/(\d+)$", path)
                                        if m:
                                            take(int(m.group(1)), val)
                    # Emit: contiguous segment text, then reconcile against the
                    # authoritative full text so the concatenation is exact.
                    delta = reconcile("".join(pending))
                    pending.clear()
                    if delta:
                        yield AskEvent("text", text=delta)

                if ev.get("related_query_items"):
                    items = [i.get("text", "") for i in ev.get("related_query_items", [])
                             if isinstance(i, dict) and i.get("text")]
                    if items:
                        yield AskEvent("related", data={"items": items})

                if ev.get("final_sse_message") is True:
                    break

            thread.touch()
            yield AskEvent("done", data={"thread": thread})
