import asyncio
import contextlib
import signal

from finsight_core.config import get_settings
from finsight_core.logging import setup_logging
from loguru import logger

settings = get_settings()


async def stream_market_data() -> None:
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting finsight-streamer service...")
    logger.info("Binance WS endpoint: {}", settings.BINANCE_WS)

    stop_event = asyncio.Event()

    def handle_stop():
        logger.info("Stopping streamer...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, handle_stop)

    try:
        while not stop_event.is_set():
            # In Diploma stage D0, this will maintain live Binance WebSocket subscriptions
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Streamer cancelled.")
    finally:
        logger.info("Streamer shut down.")


def main() -> None:
    asyncio.run(stream_market_data())


if __name__ == "__main__":
    main()
