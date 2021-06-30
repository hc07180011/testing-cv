import logging
import numpy as np


def flicker_detection(similarities, suspects, horizontal_displacements, vertical_displacements, human_reaction_threshold=3):

    # mean without outlier (outside 1 (std_degree) standard error)
    std_degree = 1
    horizontal_baseline = np.mean(horizontal_displacements[abs(
        horizontal_displacements - np.mean(horizontal_displacements)) < std_degree * np.std(horizontal_displacements)])
    vertical_baseline = np.mean(vertical_displacements[abs(
        vertical_displacements - np.mean(vertical_displacements)) < std_degree * np.std(vertical_displacements)])

    """
    If continuously smaller than or greater than -> scroll
    """
    scroll_tolerant_error = human_reaction_threshold
    status_window = np.zeros(scroll_tolerant_error).astype(
        int).tolist()  # 0 -> nothing | 1 -> greater | -1 -> smaller
    current_count = 0
    current_status = 0
    for i, h in enumerate(horizontal_displacements):
        if i in suspects:
            if h - horizontal_baseline > 0.5 * np.std(vertical_displacements):
                status = 1
            elif horizontal_baseline - h > 0.5 * np.std(vertical_displacements):
                status = -1
            else:
                status = 0
        else:
            status = 0
        if status != current_status and current_status not in status_window:  # change status
            if current_status and current_count > scroll_tolerant_error * 2:
                print("End = {}, Length = {}, status = {}".format(
                    i - len(status_window) - 1, current_count - len(status_window), current_status))
            current_status = status
            current_count = 0
            for s in status_window[::-1]:
                if s == status:
                    current_count += 1
        else:
            current_count += 1
        status_window.append(status)
        status_window.pop(0)

    for i, h in enumerate(horizontal_displacements):
        if i in suspects:
            if h > horizontal_baseline:
                print(i, 1)
            else:
                print(i, -1)
