import asyncio
import threading

from tdml.runtime.outbox import Outbox


class RecordingSender:
    """Async send() stand-in; optionally blocks until released."""

    def __init__(self, blocked: bool = False) -> None:
        self.sent: list[str] = []
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()

    async def __call__(self, text: str) -> None:
        await self.release.wait()
        self.sent.append(text)


async def _drain(outbox: Outbox, sender: RecordingSender, expected: int) -> None:
    for _ in range(200):
        if len(sender.sent) >= expected:
            return
        await asyncio.sleep(0.005)


def test_delivers_posted_messages_in_order():
    async def scenario():
        sender = RecordingSender()
        outbox = Outbox(sender, asyncio.get_running_loop())
        task = asyncio.create_task(outbox.run())
        for i in range(3):
            outbox.post(f"m{i}")
        await _drain(outbox, sender, 3)
        outbox.close()
        await task
        return sender.sent

    assert asyncio.run(scenario()) == ["m0", "m1", "m2"]


def test_keeps_only_newest_messages_while_client_is_slow():
    async def scenario():
        sender = RecordingSender(blocked=True)
        outbox = Outbox(sender, asyncio.get_running_loop(), maxlen=4, send_timeout_s=5)
        task = asyncio.create_task(outbox.run())
        outbox.post("m0")
        await asyncio.sleep(0.01)  # m0 is now in flight, blocked
        for i in range(1, 21):
            outbox.post(f"m{i}")
        sender.release.set()
        await _drain(outbox, sender, 5)
        outbox.close()
        await task
        return sender.sent, outbox.dropped

    sent, dropped = asyncio.run(scenario())
    assert sent == ["m0", "m17", "m18", "m19", "m20"]
    assert dropped == 16


def test_send_timeout_closes_outbox_and_ignores_later_posts():
    async def scenario():
        sender = RecordingSender(blocked=True)
        outbox = Outbox(sender, asyncio.get_running_loop(), send_timeout_s=0.05)
        task = asyncio.create_task(outbox.run())
        outbox.post("stuck")
        await asyncio.wait_for(task, timeout=2)
        outbox.post("late")
        return outbox.closed, outbox.pending

    closed, pending = asyncio.run(scenario())
    assert closed is True
    assert pending == 0


def test_post_threadsafe_from_worker_thread_is_delivered():
    async def scenario():
        sender = RecordingSender()
        outbox = Outbox(sender, asyncio.get_running_loop())
        task = asyncio.create_task(outbox.run())
        thread = threading.Thread(target=outbox.post_threadsafe, args=("from-thread",))
        thread.start()
        thread.join()
        await _drain(outbox, sender, 1)
        outbox.close()
        await task
        return sender.sent

    assert asyncio.run(scenario()) == ["from-thread"]


def test_post_threadsafe_after_loop_closed_does_not_raise():
    loop = asyncio.new_event_loop()
    outbox = Outbox(RecordingSender(), loop)
    loop.close()

    outbox.post_threadsafe("too late")  # must not raise
