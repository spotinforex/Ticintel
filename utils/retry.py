import asyncio
import logging
import time
from functools import wraps


def retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator for both sync and async functions.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier applied to delay after each retry (exponential backoff)
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logging.error(
                            "%s failed after %d attempts. Error: %s",
                            func.__name__, max_attempts, e
                        )
                        raise
                    logging.warning(
                        "%s attempt %d failed: %s. Retrying in %.2fs...",
                        func.__name__, attempt, e, current_delay
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logging.error(
                            "%s failed after %d attempts. Error: %s",
                            func.__name__, max_attempts, e
                        )
                        raise
                    logging.warning(
                        "%s attempt %d failed: %s. Retrying in %.2fs...",
                        func.__name__, attempt, e, current_delay
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
