"""Load-balanced pool of Perplexity accounts from a Netscape cookie file.

accounts.txt holds any number of account blocks:
    account 1:
    <netscape cookie lines>
    account 2:
    ...

The file is re-read when its mtime changes, so adding accounts takes effect
without restarting the server. Accounts are skipped while quota-exhausted,
failing, or cooling down; selection prefers the least-loaded healthy account.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from pplx_transport import PerplexityClient

log = logging.getLogger("pplx.accounts")

QUOTA_TTL = 120.0       # rate-limit/status cache
COOLDOWN = 300.0        # after consecutive failures
FAIL_THRESHOLD = 3      # consecutive failures -> cooldown


@dataclass
class Account:
    index: int
    label: str
    cookies: dict
    client: PerplexityClient
    active: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    quota_known: Optional[bool] = None
    quota_checked_at: float = 0.0
    last_error: Optional[str] = None

    def healthy(self, now: float) -> bool:
        if now < self.cooldown_until:
            return False
        if self.quota_known is False:
            return False
        return True


def parse_accounts(path: str) -> list[dict]:
    """Parse the Netscape-cookie account file into cookie dicts."""
    accounts: list[dict] = []
    current: Optional[dict] = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("account"):
                current = {"label": line, "cookies": {}}
                accounts.append(current)
                continue
            if current is None or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            current["cookies"][parts[5]] = parts[6]
    return [a for a in accounts if a["cookies"]]


class AccountPool:
    def __init__(self, path: str, *, max_concurrent: int = 2):
        self.path = path
        self.max_concurrent = max_concurrent
        self._accounts: list[Account] = []
        self._rr = 0
        self._mtime: Optional[float] = None
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self.reload()

    async def reload(self) -> None:
        try:
            st = os.stat(self.path)
            if self._mtime == st.st_mtime and self._accounts:
                return
            raw = parse_accounts(self.path)
        except FileNotFoundError:
            log.error("accounts file missing: %s", self.path)
            return
        except Exception as e:
            log.error("failed to parse accounts: %s", e)
            return
        async with self._lock:
            if self._mtime is not None and self._mtime == st.st_mtime and self._accounts:
                return
            # Keep existing clients for unchanged accounts (session reuse).
            existing = {a.index: a for a in self._accounts}
            new_list = []
            for i, spec in enumerate(raw):
                old = existing.get(i)
                if old is not None and old.cookies == spec["cookies"]:
                    new_list.append(old)
                else:
                    if old is not None:
                        await old.client.close()
                    new_list.append(Account(
                        index=i, label=spec["label"], cookies=spec["cookies"],
                        client=PerplexityClient(
                            spec["cookies"],
                            max_concurrent=self.max_concurrent,
                        ),
                    ))
            self._accounts = new_list
            self._semaphores = {
                a.index: asyncio.Semaphore(self.max_concurrent)
                for a in new_list
            }
            self._mtime = st.st_mtime
            log.info("accounts loaded: %d", len(new_list))

    @property
    def size(self) -> int:
        return len(self._accounts)

    async def _refresh_quota(self, acct: Account) -> None:
        now = time.time()
        if now - acct.quota_checked_at < QUOTA_TTL:
            return
        try:
            acct.quota_known = await acct.client.quota_available()
            acct.quota_checked_at = now
        except Exception:
            pass

    async def pick(self) -> Optional[tuple[Account, asyncio.Semaphore]]:
        """Pick the healthiest account (least active, round-robin tie-break).

        Returns None when every account is unhealthy.
        """
        await self.reload()
        await asyncio.sleep(0)  # let pending health updates land
        now = time.time()
        async with self._lock:
            candidates = [a for a in self._accounts if a.healthy(now)]
            if not candidates:
                return None
            for a in candidates:
                await self._refresh_quota(a)
            candidates = [a for a in candidates if a.healthy(time.time())]
            if not candidates:
                return None
            candidates.sort(key=lambda a: (a.active, (a.index - self._rr) % len(self._accounts)))
            self._rr = (self._rr + 1) % max(len(self._accounts), 1)
            acct = candidates[0]
            return acct, self._semaphores[acct.index]

    def get(self, index: int) -> Optional[tuple[Account, asyncio.Semaphore]]:
        """Return a specific account if it exists and is healthy."""
        now = time.time()
        acct = self._accounts[index] if 0 <= index < len(self._accounts) else None
        if acct is None or not acct.healthy(now):
            return None
        return acct, self._semaphores[acct.index]

    def record_success(self, acct: Account) -> None:
        acct.consecutive_failures = 0
        acct.last_error = None
        acct.quota_known = None  # re-check on next pick

    def record_failure(self, acct: Account, error: str, *, quota: bool = False) -> None:
        acct.consecutive_failures += 1
        acct.last_error = error
        if quota:
            acct.quota_known = False
            acct.quota_checked_at = time.time()
        if acct.consecutive_failures >= FAIL_THRESHOLD:
            acct.cooldown_until = time.time() + COOLDOWN
            acct.consecutive_failures = 0
            log.warning("account %d cooling down: %s", acct.index, error)

    def status(self) -> list[dict]:
        now = time.time()
        return [{
            "index": a.index,
            "label": a.label,
            "active": a.active,
            "healthy": a.healthy(now),
            "quota_available": a.quota_known,
            "last_error": a.last_error,
            "cooldown_until": a.cooldown_until,
        } for a in self._accounts]

    async def close(self) -> None:
        for a in self._accounts:
            await a.client.close()
