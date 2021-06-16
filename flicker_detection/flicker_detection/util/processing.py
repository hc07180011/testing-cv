import cv2
import logging
import numpy as np


def take_snapshots(video_path: str, limit=np.inf) -> np.ndarray:
    """Extracting frames from the given video

    Args:
        video_path (:obj:`str`): The path to the video file.
        limit (:obj:`int`, optional): Maximum frame to take. Defaults to np.inf.

    Returns:
        np.ndarray: A numpy array that includes all frames

    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    ret_images = []
    count = 0
    while success and count < limit:
        ret_images.append(image)
        success, image = vidcap.read()
        logging.debug('Parsing image: #{:04d}'.format(count))
        count += 1
    return np.array(ret_images)


def consine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If ``*args`` or ``**kwargs`` are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.
    """

    ret = np.inner(vector1, vector2) / \
        (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    # fake
    ret = -np.linalg.norm(vector1 - vector2)
    return float(ret)
