import sys
import logging

from rich.logging import RichHandler


def init_logger() -> None:
    logger = logging.getLogger("rich")

    FORMAT = "%(name)s[%(process)d] " + \
        "%(processName)s(%(threadName)s) " + \
        "%(module)s:%(lineno)d  %(message)s"

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        FORMAT,
        datefmt="%Y%m%d %H:%M:%S"
    )
    logging.basicConfig(
        level="NOTSET", format=FORMAT, handlers=[RichHandler()]
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    logger.addHandler(ch)

    logging.info("Initializing ok.")
