"""Per-connection reply buffer: bounded (oldest dropped), one sender task, send timeout.

Worker threads post replies; a single asyncio task sends them. A client that stops reading can
therefore hold at most `maxlen` replies plus one in-flight send, and is cut off after the timeout.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

OUTBOX_MAXLEN = 8
SEND_TIMEOUT_S = 2.0


class Outbox:
    def __init__(
        self,
        send: Callable[[str], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
        maxlen: int = OUTBOX_MAXLEN,
        send_timeout_s: float = SEND_TIMEOUT_S,
    ) -> None:
        self._send = send
        self._loop = loop
        self._items: deque[str] = deque(maxlen=maxlen)
        self._wakeup = asyncio.Event()
        self._send_timeout_s = send_timeout_s
        self._closed = False
        self.dropped = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        return len(self._items)

    def post(self, text: str) -> None:
        """Queue a reply. Event-loop thread only."""
        if self._closed:
            return
        if len(self._items) == self._items.maxlen:
            self.dropped += 1
        self._items.append(text)
        self._wakeup.set()

    def post_threadsafe(self, text: str) -> None:
        """Queue a reply from a worker thread; a no-op once the connection or loop is gone."""
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self.post, text)
        except RuntimeError:  # event loop already closed (server shutting down)
            self._closed = True

    def close(self) -> None:
        self._closed = True
        self._items.clear()
        self._wakeup.set()

    async def run(self) -> None:
        """Send queued replies until closed, the client errors, or a send exceeds the timeout."""
        try:
            while not self._closed:
                await self._wakeup.wait()
                self._wakeup.clear()
                while self._items and not self._closed:
                    await asyncio.wait_for(self._send(self._items.popleft()), self._send_timeout_s)
        except TimeoutError:
            logger.warning(
                "client not reading for %.1f s; replies disabled for this connection", self._send_timeout_s
            )
        except Exception as exc:  # socket closed mid-send
            logger.debug("reply sender stopped: %s", exc)
        finally:
            self.close()
