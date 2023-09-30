import sys
import signal

from typing import Callable


def set_exit_handler(c: Callable) -> None:
    def signal_handler(_, __):
        c()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
