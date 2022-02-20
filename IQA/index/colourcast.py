import cv2 as cv
import numpy as np


class ColourCast:

    def __init__(self, img_input: np.ndarray) -> None:
        self.img_input = img_input

    def color_cast(self):
        # RGB to La*b*
        img_float = self.img_input.astype(np.float32) / 255.0

        np_R = img_float[:, :, 2]
        np_G = img_float[:, :, 1]
        np_B = img_float[:, :, 0]

        # RGB to CIE XYZ
        np_X = 0.412453*np_R + 0.357580*np_G + 0.180423*np_B
        np_Y = 0.212671*np_R + 0.715160*np_G + 0.072169*np_B
        np_Z = 0.019334*np_R + 0.119193*np_G + 0.950227*np_B

        # CIE XYZ to CIELab

        def f(t):
            result1 = t**(1/3)
            result2 = 7.787*t+(16/116)

            return (t > 0.008856)*result1 + (t <= 0.008856)*result2

        np_L_1 = 116 * np_Y ** (1/3) - 16
        np_L_2 = 903.3 * np_Y

        np_L = (np_Y > 0.008856)*np_L_1 + (np_Y <= 0.008856)*np_L_2

        np_a = 500*(f(np_X/0.950456) - f(np_Y))
        np_b = 200*(f(np_Y) - f(np_Z/1.088754))

        D = np.sqrt(np_a.mean()**2 + np_b.mean()**2)
        M_a = abs(np_a - np_a.mean()).mean()
        M_b = abs(np_b - np_b.mean()).mean()
        M = np.sqrt(M_a**2 + M_b**2)

        # K = D / M

        img_gray = cv.cvtColor(self.img_input, cv.COLOR_BGR2GRAY)
        A = 2000
        K_ = D / (M * max(abs(min(np.var(img_gray) - A, 1)), 1))

        return K_
