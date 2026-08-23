import os
import structlog
import threading
from typing import Optional


logger = structlog.get_logger()

_driver_singleton = None
_driver_pid = None
_driver_lock = threading.Lock()


def _load_mofka():
    try:
        import mochi.mofka.client as mofka
    except ModuleNotFoundError as exc:
        raise RuntimeError("Mofka client is not available. Install mochi.mofka.") from exc
    return mofka


def _get_driver(group_file: str, use_progress_thread: bool):
    global _driver_singleton
    global _driver_pid
    with _driver_lock:
        if _driver_singleton is None or _driver_pid != os.getpid():
            mofka = _load_mofka()
            _driver_pid = os.getpid()
            logger.info(
                "mofka.driver.create",
                pid=_driver_pid,
                group_file=group_file,
                use_progress_thread=use_progress_thread,
            )
            _driver_singleton = mofka.MofkaDriver(
                group_file=group_file,
                use_progress_thread=use_progress_thread,
            )
        else:
            logger.debug("mofka.driver.reuse", pid=_driver_pid)
    return _driver_singleton


def open_consumer(
    group_file: str,
    topic_name: str,
    consumer_name: Optional[str] = None,
    use_progress_thread: bool = True,
):
    mofka = _load_mofka()
    driver = _get_driver(group_file, use_progress_thread)
    logger.info("mofka.topic.open.start", topic=topic_name)
    topic = driver.open_topic(topic_name)
    logger.info("mofka.topic.open.done", topic=topic_name)
    effective_name = consumer_name or f"dfdiagnoser_{os.getpid()}"
    logger.info("mofka.consumer.open", topic=topic_name, name=effective_name)
    consumer = topic.consumer(
        name=effective_name,
        thread_pool=driver.default_thread_pool,
        batch_size=mofka.AdaptiveBatchSize,
        data_allocator=mofka.ByteArrayAllocator,
        data_selector=mofka.FullDataSelector,
    )
    return driver, consumer


def open_producer(
    group_file: str,
    topic_name: str,
    use_progress_thread: bool = True,
):
    mofka = _load_mofka()
    driver = _get_driver(group_file, use_progress_thread)
    logger.info("mofka.topic.open.start", topic=topic_name)
    topic = driver.open_topic(topic_name)
    logger.info("mofka.topic.open.done", topic=topic_name)
    logger.info("mofka.producer.open", topic=topic_name)
    producer = topic.producer(
        batch_size=mofka.AdaptiveBatchSize,
        thread_pool=driver.default_thread_pool,
        ordering=mofka.Ordering.Strict,
    )
    return driver, producer


# Acknowledgement is DISABLED by default. Full rationale and measurements in
# dfoptimizer/streaming/mofka_io.py::ack_event and LIVEFLOW-STATUS sec 60.
#
# Short version: on diaspora-stream-api 0.5.1, acking an event drawn from a multi-event
# batch permanently stops delivery to that consumer. The diagnoser happens to survive it
# today only because the analyzer feeds it one event per flush; that is luck, not design,
# and it breaks the moment the analyzer flushes two windows together. Our consumers are
# PID-named against per-run, memory-backed topics, so they never resume an offset and the
# ack carries no information.
#
# Set LIVEFLOW_MOFKA_ACK=1 to restore the old behaviour.
def ack_event(event) -> None:
    if os.environ.get("LIVEFLOW_MOFKA_ACK", "0") == "1":
        event.acknowledge()
