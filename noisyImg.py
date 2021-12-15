import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import pandas as pd


class noisyImg_generator:
    # ============================================================================
    def __init__(self, noiseType: str, img_path: Path, params: Tuple) -> None:
        assert noiseType.lower() in [
            "gaussian",
            "salt_pepper",
            "poisson",
        ], "the noise type does not exist"
        self.noiseType = noiseType
        self.img_orig: np.ndarray = cv2.imread(str(img_path))
        self.params = params

    # ============================================================================
    def get_noiseType(self) -> str:
        return self.noiseType

    # ============================================================================
    def get_params(self) -> Tuple:
        return self.params

    # ============================================================================
    def __call__(self) -> np.ndarray:
        # self.img_orig = self.preprocess_img(self.img_orig)

        row, col, ch = self.img_orig.shape
        if self.noiseType.lower() == "gaussain" or "salt_pepper":
            assert len(self.params) == 2, "lack of the parameters of noise"

        if self.noiseType.lower() == "gaussian":
            mean, sigma = self.params
            noise = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
            return self.img_orig + noise

        elif self.noiseType.lower() == "salt_pepper":
            s_vs_p, amount = self.params
            out = np.copy(self.img_orig)

            # Salt mode
            num_salt = np.ceil(amount * self.img_orig.size * s_vs_p)
            coords = [
                np.random.randint(0, i - 1, int(num_salt)) for i in self.img_orig.shape
            ]
            out[coords] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * self.img_orig.size * (1.0 - s_vs_p))
            coords = [
                np.random.randint(0, i - 1, int(num_pepper))
                for i in self.img_orig.shape
            ]
            out[coords] = 0
            return out

        else:
            return np.random.poisson(self.img_orig).astype(float) + self.img_orig
