import os
import cv2
import logging
import datetime
import coloredlogs
import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser

from index.colourcast import ColourCast
from index.purplefringing import PurpleFringing


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--img_dir", type=str,
        default=os.path.join("data", "old_photos"),
        help="directory of experiments .jpg"
    )
    parser.add_argument(
        "-l", "--log_dir", type=str,
        default=".log",
        help="directory of log files"
    )
    parser.add_argument(
        "-c", "--cache_dir", type=str,
        default=".cache",
        help="directory of caches"
    )
    parser.add_argument(
        "-dc", "--disable_cache", action="store_true",
        default=False,
        help="disable caching function"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        default=False,
        help="print debugging logs"
    )
    args = parser.parse_args()

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s[%(process)d]%(levelname)s " +
        "%(processName)s(%(threadName)s) %(module)s:%(lineno)d  %(message)s",
        datefmt='%Y%m%d %H:%M:%S'
    )

    ch = logging.StreamHandler()
    if args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    log_filename = os.path.join(
        args.log_dir,
        datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
    )
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
        fmt="%(asctime)s %(hostname)s %(name)s[%(process)d] " +
        "%(levelname)s %(processName)s(%(threadName)s) " +
        "%(module)s:%(lineno)d  %(message)s",
        level="DEBUG"
    )

    logging.info("Program starts")

    results_dict = dict()
    labels = list([
        "Name", "K", "Purple Fringing Pixel",
        "Purple Fringing Total Ratio",
        "Purple Fringing Denoise Pixel",
        "Purple Fringing Denoise Total Ratio",
        "Purple Fringing Ratio2",
        "Purple Fringing Ratio3",
        "Night View", "Purple Cast"

    ])
    for label in labels:
        results_dict[label] = list()

    divided_ratio = 8

    for img_filename in tqdm(sorted(os.listdir(args.img_dir))):

        if ".jpg" not in img_filename and \
                ".png" not in img_filename:
            continue

        image = cv2.imread(os.path.join(
            args.img_dir,
            img_filename
        ))

        """
        Colour Cast
        """

        colour_cast = ColourCast(image)

        results_dict["K"].append(colour_cast.color_cast())

        """
        Purple Fringing
        """

        purple_fringing = PurpleFringing(image)

        results_dict["Night View"].append(
            purple_fringing.night_view_detect()
        )
        results_dict["Purple Cast"].append(
            purple_fringing.colour_purple_cast_detect()
        )

        mask, denoise_mask = purple_fringing.purple_fringing_detect()

        results_dict["Name"].append(img_filename)
        results_dict["Purple Fringing Pixel"].append(mask.sum())
        results_dict["Purple Fringing Total Ratio"].append(mask.mean())

        results_dict["Purple Fringing Denoise Pixel"].append(
            denoise_mask.sum()
        )
        results_dict["Purple Fringing Denoise Total Ratio"].append(
            denoise_mask.mean()
        )

        h, w, _ = image.shape

        count1 = 0
        count2 = 0

        h_space = h // divided_ratio
        w_space = w // divided_ratio

        for i in range(divided_ratio):
            for j in range(divided_ratio):

                purple_ratio = np.round(
                    (mask[
                        i*h_space: (i+1)*h_space,
                        j*w_space: (j+1)*w_space
                    ]).mean() * 1e5, 3
                )

                if purple_ratio != 0:
                    count1 += 1
                if purple_ratio > 100:
                    count2 += 1

        results_dict["Purple Fringing Ratio2"].append(count1)
        results_dict["Purple Fringing Ratio3"].append(count2)

    resutls_df = pd.DataFrame(results_dict)
    resutls_df.to_csv(os.path.join(args.img_dir, "restuls.csv"))


if __name__ == "__main__":
    main()
