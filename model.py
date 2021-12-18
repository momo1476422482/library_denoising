import numpy as np
from pathlib import Path
import cv2
from utility import LocalisedPatches
from tqdm import tqdm
from typing import Optional, Tuple, List
import pandas as pd


class NLMeans:
    # ===============================================================================================
    def __init__(
        self,
        patch_size: int,
        h: float,
        radius: int,
        sigma: Optional[float] = 0,
    ) -> None:

        self.patch_size = patch_size
        self.localised_patches = LocalisedPatches(self.patch_size)
        self.padwidth = self.patch_size // 2
        self.h = h
        self.sigma = sigma
        self.search_radius = radius
        self.path2csv = Path(__file__).parent / "result_nlmeans.csv"

    # ===============================================================================================
    def weigh_patches(
        self, patch_array_1: np.ndarray, patch_array_2: np.ndarray
    ) -> np.ndarray:
        """
        calculate the weight  between two patch arrays
        """

        assert (
            patch_array_1.shape == patch_array_2.shape
        ), f"{patch_array_1.shape} not equal {patch_array_2.shape}"

        assert len(patch_array_1.shape) == 2, "the input array should be 2-dimensional"

        weight = (
            np.sum(
                np.square(patch_array_1 - patch_array_2),
                axis=1,
            )
            - 2 * np.power(self.sigma, 2)
        )

        return np.exp(-weight / np.power(self.h, 2))

    # ===============================================================================================
    def get_patch(self, coordinate: Tuple[int, int], img: np.ndarray) -> np.ndarray:
        x, y = coordinate
        return img[
            x - self.padwidth : x + self.padwidth + 1,
            y - self.padwidth : y + self.padwidth + 1,
        ]

    # ===============================================================================================
    def set_patch(
        self, coordinate: Tuple[int, int], img: np.ndarray, value: np.ndarray
    ) -> None:

        x, y = coordinate
        img[
            x - self.padwidth : x + self.padwidth + 1,
            y - self.padwidth : y + self.padwidth + 1,
        ] = value

    # ===============================================================================================
    def img2patch(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches and the locolisation of patches from an input image
        """

        patches, coordinates = self.localised_patches.from_array(img)
        # coordinate corresponds to the center pixel of patch
        coordinates = coordinates + self.padwidth
        return patches, coordinates

    # ===============================================================================================
    def patch2img(
        self, coordinates: np.ndarray, patches: np.ndarray, img: np.ndarray
    ) -> np.ndarray:
        """
        Inverse operation of img2patch
        """
        img = np.zeros(img.shape)
        img_weight_final = np.zeros(img.shape)
        for coordinate, patch in zip(coordinates, patches):
            img_zero = np.zeros(img.shape)
            img_weight = np.zeros(img.shape)
            x, y = tuple(coordinate)
            self.set_patch(coordinate, img_zero, patch)
            img += img_zero
            self.set_patch(coordinate, img_weight, 1)
            img_weight_final += img_weight
        return np.divide(img, img_weight_final)

    # ===============================================================================================
    def get_search_area(
        self,
        coordinate: Tuple[int, int],
        img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        get the surrounded image of a given pixel
        """
        center_pixel_x, center_pixel_y = coordinate
        left_bord = max(center_pixel_x - self.search_radius, 0)
        right_bord = min(center_pixel_x + self.search_radius, img.shape[1])
        high_bord = max(center_pixel_y - self.search_radius, 0)
        low_bord = min(center_pixel_y + self.search_radius, img.shape[1])
        top_left = (left_bord, high_bord)
        return img[left_bord : right_bord + 1, high_bord : low_bord + 1], top_left

    # ===============================================================================================
    def get_weights_search_area(
        self, coordinate: Tuple[int, int], img: np.ndarray, search_area: np.ndarray
    ) -> np.ndarray:
        """
        calculate the weights between the current patch(centered at the given pixel)
        and the other patches on the search area
        """

        patches, coordinates = self.img2patch(search_area)
        center_patch = np.tile(
            self.get_patch(coordinate, img).reshape(1, -1),
            (patches.shape[0], 1),
        )
        return self.weigh_patches(center_patch, patches.reshape(patches.shape[0], -1))

    # ===============================================================================================
    def denoise_pixel(
        self,
        coordinate: Tuple[int, int],
        input_img: np.ndarray,
    ) -> np.ndarray:
        """
        process of denoising one pixel
        """
        search_area, top_left = self.get_search_area(coordinate, input_img)
        weights = self.get_weights_search_area(coordinate, input_img, search_area)
        totalweights = sum(weights)
        center_img = self.crop_img(search_area)
        return (
            np.dot(weights, center_img.reshape(-1, search_area.shape[2])) / totalweights
        )

    # ===============================================================================================
    def get_list_nearest_coordinates(
        self, coordinate: Tuple[int, int], nb_neighbors: int, img: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        get the list of the nearest patches of the given pixel
        parameter nb_neighbors :  the number of the neares patches
        """
        search_area, top_left = self.get_search_area(coordinate, img)
        patches, coordinates = self.img2patch(search_area)
        weights = self.get_weights_search_area(coordinate, img, search_area)
        coordinates_orig_img = coordinates + np.array(top_left)
        return coordinates_orig_img[np.argsort(weights)[::-1][:nb_neighbors]].tolist()

    # ===============================================================================================
    def plot_patch_selection(
        self,
        coordinate: Tuple[int, int],
        img: np.ndarray,
        path_mask: Path,
        nb_neighbors: int = 5,
    ) -> List[Tuple[int, int]]:
        """
        plot the nearest patches of a pixel
        parameter path_mask : the path for saving the plotted mask
        """
        mask = img.copy()

        # plot the nearest patches
        res: List[Tuple[int, int]] = []
        list_nearest_coordinates = self.get_list_nearest_coordinates(
            coordinate, nb_neighbors, img
        )
        for coord in list_nearest_coordinates:
            if list_nearest_coordinates.index(coord) == 0:
                self.set_patch(coord, mask, (0, 255, 0))
            self.set_patch(coord, mask, (255, 0, 0))
            res.append((np.array(coord) - self.padwidth))
        cv2.imwrite(str(path_mask), mask)
        return res

    # ===============================================================================================
    def crop_img(self, img: np.ndarray) -> np.ndarray:
        return img[
            self.padwidth : img.shape[0] - self.padwidth,
            self.padwidth : img.shape[1] - self.padwidth,
        ]

    # ===============================================================================================
    def add_noise(self, img: np.ndarray, sigma: int) -> np.ndarray:
        return img + np.random.randn(*(img.shape)) * sigma

    # ===============================================================================================
    def denoise_img(self, img: np.ndarray) -> np.ndarray:
        output_img = np.zeros_like(img)
        for i in tqdm(range(self.padwidth, img.shape[0] - self.padwidth)):
            for j in range(self.padwidth, img.shape[1] - self.padwidth):
                output_img[i, j] = self.denoise_pixel((i, j), img)

        return self.crop_img(output_img)

    # ===============================================================================================
    def get_RMSE(self, img_ref: np.ndarray, img_pred: np.ndarray) -> float:
        orig_img = self.crop_img(img_ref)
        assert img_pred.shape == orig_img.shape
        return np.sqrt(np.sum(np.power((img_pred - orig_img), 2)) / img_pred.size)

    # ===============================================================================================
    def save_to_csv(self, img_name: str, RMSE: float):
        data = {"image_name": img_name, "sigma": self.sigma, "RMSE": RMSE, "h": self.h}
        if not self.path2csv.is_file():
            data_result = pd.DataFrame.from_dict([data])
            data_result.to_csv(self.path2csv)
        else:
            data_result = pd.read_csv(self.path2csv)
            df = [data_result, pd.DataFrame.from_dict([data])]
            data_result = pd.concat(df)
            data_result.to_csv(self.path2csv, index=False)
        print(data_result.to_string())

    # ===============================================================================================
    def __call__(self, path_img: Path, sigma: int) -> np.ndarray:
        assert path_img.is_file(), f"{path_img} does not exist"
        img_orig = cv2.imread(str(path_img))
        noisy_img = self.add_noise(img_orig, sigma)
        denoised_img = self.denoise_img(noisy_img)
        cv2.imwrite(str(path_img.parent / f"denoised_{path_img.name}"), denoised_img)
        RMSE = self.get_RMSE(img_orig, denoised_img)
        self.save_to_csv(path_img.stem, RMSE)
