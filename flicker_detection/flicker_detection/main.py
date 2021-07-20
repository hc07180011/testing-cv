import os
import logging
import datetime
import coloredlogs

from argparse import ArgumentParser

from preprocessing.feature_extraction import Features
from core.flicker import Flicker


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str,
                        default="data/177193533.mp4", help="path to video, run Flask if empty")
    parser.add_argument("-l", "--log_dir", type=str, default=".log",
                        help="directory of log files")
    parser.add_argument("-dc", "--disable_cache", type=bool, default=False,
                        help="whether to disable cache function")
    parser.add_argument("-c", "--cache_dir", type=str, default=".cache",
                        help="directory of caches")
    parser.add_argument("-v", "--verbose", type=bool,
                        default=True, help="whether to print debugging logs")
    args = parser.parse_args()

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s[%(process)d] %(levelname)s %(processName)s(%(threadName)s) %(module)s:%(lineno)d  %(message)s",
        datefmt='%Y%m%d %H:%M:%S')

    ch = logging.StreamHandler()
    if args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    log_filename = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log"))
    fh = logging.FileHandler(log_filename)
    if args.verbose:
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    coloredlogs.install(
        fmt="%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(processName)s(%(threadName)s) %(module)s:%(lineno)d  %(message)s", level="DEBUG")

    logging.info("Program start ...")

    video_features = Features(
        args.data_path, not args.disable_cache, args.cache_dir)
    video_features.feature_extraction()

    flicker = Flicker(video_features.similarities, video_features.suspects,
                      video_features.horizontal_displacements, video_features.vertical_displacements)
    flicker.flicker_detection()


if __name__ == "__main__":
    main()
