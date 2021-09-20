import cv2 as cv
import numpy as np


class PurpleFringing:

    def __init__(self, img_input: np.ndarray) -> None:
        self.img_input = img_input

    def night_view_detect(self) -> bool:
        if self.img_input.mean() < 80 and self.img_input.std() < 60:
            return True
        return False

    def colour_purple_cast_detect(self) -> bool:
        img_input_float = self.img_input.astype(np.float32) / 255.0
        np_Cb = -0.169 * img_input_float[:, :, 2] - 0.331 * \
            img_input_float[:, :, 1] + 0.5 * img_input_float[:, :, 0]
        np_Cr = 0.5 * img_input_float[:, :, 2] - 0.419 * \
            img_input_float[:, :, 1] - 0.081 * img_input_float[:, :, 0]

        # count purple distance
        np_p_diff_Cb = np_Cb - 0.331
        np_p_diff_Cr = np_Cr - 0.42

        np_p_dist = np.sqrt(np_p_diff_Cb * np_p_diff_Cb +
                            np_p_diff_Cr * np_p_diff_Cr)

        # count green distance
        np_g_diff_Cb = np_Cb + 0.331
        np_g_diff_Cr = np_Cr + 0.42

        np_g_dist = np.sqrt(np_g_diff_Cb*np_g_diff_Cb +
                            np_g_diff_Cr*np_g_diff_Cr)

        # threshold can be [0.88, 0.9]
        if (np_p_dist / np_g_dist).mean() < 0.89:
            return True
        else:
            return False

    def purple_fringing_detect(self):
        # get HSV purple region
        # ===================================
        img_HSV = cv.cvtColor(self.img_input, cv.COLOR_BGR2HSV)
        low = np.array([120, 20, 20])
        high = np.array([140, 255, 255])
        mask_HSV_purple = cv.inRange(img_HSV, low, high)

        # change BGR to YCbCr
        # get YCbCr purple region
        # ====================================
        img_input_float = self.img_input.astype(np.float32) / 255.0
        np_Y = 0.229 * img_input_float[:, :, 2] + 0.587 * \
            img_input_float[:, :, 1] + 0.114*img_input_float[:, :, 0]
        np_Cb = -0.169 * img_input_float[:, :, 2] - 0.331 * \
            img_input_float[:, :, 1] + 0.5*img_input_float[:, :, 0]
        np_Cr = 0.5 * img_input_float[:, :, 2] - 0.419 * \
            img_input_float[:, :, 1] - 0.081*img_input_float[:, :, 0]

        # according to paper, get YCbCr purple region
        # count purple distance
        np_p_diff_Cb = np_Cb - 0.331
        np_p_diff_Cr = np_Cr - 0.42
        np_p_dist = np.sqrt(np_p_diff_Cb * np_p_diff_Cb +
                            np_p_diff_Cr * np_p_diff_Cr)

        # count green distance
        np_g_diff_Cb = np_Cb + 0.331
        np_g_diff_Cr = np_Cr + 0.42
        np_g_dist = np.sqrt(np_g_diff_Cb*np_g_diff_Cb +
                            np_g_diff_Cr*np_g_diff_Cr)

        kernel = np.ones((5, 5)) / 25

        np_p_dist_2 = cv.filter2D(np_p_dist, -1, kernel)
        np_g_dist_2 = cv.filter2D(np_g_dist, -1, kernel)

        # 0.9 is a threhold [0.85, 0.95]
        mask_YCbCr_purple = (np_p_dist_2 / np_g_dist_2) < 0.9

        # intersection HSV and YCbCr purple region
        # ===================================
        mask_intersection_purple = (mask_HSV_purple == 255) & mask_YCbCr_purple

        # count the neighbor purple region
        # ===================================
        w_size = 5
        kernel = np.ones((w_size, w_size)) / (w_size * w_size)

        channel_list = []
        for c in range(3):
            img_channel = self.img_input[:, :, c] * mask_intersection_purple
            np_tmp = cv.filter2D(img_channel, -1, kernel)
            channel_list.append(np_tmp * mask_intersection_purple)

        # check if B > G & B > R, that is purple region
        mask_purple = (channel_list[1] < channel_list[0]) & (
            channel_list[1] < channel_list[2])

        # calculate NSRs
        # min is 0.7
        # ====================================
        blur = cv.GaussianBlur(np_Y, (5, 5), 0)

        window_size = 11
        kernel = np.ones((window_size, window_size))

        mean = blur.mean()
        std = blur.std()
        alpha = 1.5
        threshold = mean + alpha * std
        threshold = max(threshold, 0.7)

        mask_NSRs = blur > threshold

        mask_expand_NSRs = cv.dilate(
            mask_NSRs.astype(np.uint8), kernel, iterations=1)

        # calculate HIGH-CONTRAST NSRs
        # ===================================
        sobelx = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=5)

        M = np.sqrt(sobelx*sobelx + sobely*sobely)

        M_mean = M.mean()
        M_std = M.std()
        alpha = 3
        M_t = M_mean + alpha * M_std

        mask_NSRs_2 = M > M_t

        mask_expand_NSRs_2 = cv.dilate(
            mask_NSRs_2.astype(np.uint8), kernel, iterations=1)

        # intersection 2 HSR mask
        mask_nearNSR_purple = mask_purple & mask_expand_NSRs & mask_expand_NSRs_2

        # use connected compoent to remove few pixels regions
        # ====================================
        num_comp, comp_labels, comp_stats, comp_centroid = cv.connectedComponentsWithStats(
            mask_nearNSR_purple.astype(np.uint8))

        # threshold = total pixels * 10^-5
        threshold = self.img_input.shape[0] * \
            self.img_input.shape[1]*1e-5  # output3=1e-6*5
        # threshold = img_input.shape[0]*img_input.shape[1]*1e-6*5

        set_filter = set(np.arange(num_comp) * (comp_stats[:, 4] > threshold))
        set_filter.remove(0)
        list_filter = list(set_filter)
        final_mask = np.isin(comp_labels, list_filter)

        return mask_nearNSR_purple, final_mask

    def draw_purple_fringing(self, mask):
        imask = mask != True
        img_purple = np.dstack(((255*mask + self.input_image[:, :, 0]*imask),
                                (0*mask + self.input_image[:, :, 1]*imask),
                                (255*mask + self.input_image[:, :, 2]*imask)))
        return img_purple
