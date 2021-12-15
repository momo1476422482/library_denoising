import numpy as np
from pathlib import Path
import cv2
from utility import LocalisedPatches
from tqdm import tqdm
from typing import Optional, Tuple, List


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

    # ===============================================================================================
    def weight_patches(
        self, patch_array_1: np.ndarray, patch_array_2: np.ndarray
    ) -> np.ndarray:
        """
        calculate the weight between two patch arrays
        """

        assert (
            patch_array_1.shape == patch_array_2.shape
        ), f"{patch_array_1.shape} not equal {patch_array_2.shape}"

        assert len(patch_array_1.shape) == 2, "the input array should be 2-dimensional"

        euclideanDistance = (
            np.sqrt(
                np.sum(
                    np.square(patch_array_1 - patch_array_2),
                    axis=1,
                )
            )
            - 2 * np.power(self.sigma, 2)
        )
        return np.exp(-euclideanDistance / np.power(self.h, 2))

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
        Reconstruct the image with the given list of patches and its locolisation
        (inverse operation of img2patch)
        """
        img = np.zeros(img.shape)
        img_index_final = np.zeros(img.shape)
        for coordinate, patch in zip(coordinates, patches):
            img_zero = np.zeros(img.shape)
            img_index = np.zeros(img.shape)
            x, y = tuple(coordinate)
            img_zero
            img_zero[
                x - self.padwidth : x + self.padwidth + 1,
                y - self.padwidth : y + self.padwidth + 1,
            ] = patch
            img += img_zero
            img_index[
                x - self.padwidth : x + self.padwidth + 1,
                y - self.padwidth : y + self.padwidth + 1,
            ] = 1
            img_index_final += img_index
        return np.divide(img, img_index_final)

    # ===============================================================================================
    def get_search_area(
        self, coordinate: Tuple[int, int], img: np.ndarray
    ) -> np.ndarray:
        """
        get the surrounded image of a given pixel
        """
        center_pixel_x, center_pixel_y = coordinate
        left_bord = max(center_pixel_x - self.search_radius, 0)
        right_bord = min(center_pixel_x + self.search_radius, img.shape[1])
        high_bord = max(center_pixel_y - self.search_radius, 0)
        low_bord = min(center_pixel_y + self.search_radius, img.shape[1])
        return img[left_bord : right_bord + 1, high_bord : low_bord + 1]

    # ===============================================================================================
    def get_weights_search_area(
        self, coordinate: Tuple[int, int], img: np.ndarray
    ) -> np.ndarray:
        """
        calculate the weights between the concerned patch(centered at the given pixel)
        and the other patches on the search area
        """

        surrounded_img = self.get_search_area(coordinate, img)
        patches, coordinates = self.img2patch(surrounded_img)

        center_x, center_y = coordinate
        center_patch = np.tile(
            img[
                center_x - self.padwidth : center_x + self.padwidth + 1,
                center_y - self.padwidth : center_y + self.padwidth + 1,
            ].reshape(1, -1),
            (patches.shape[0], 1),
        )
        return self.weight_patches(center_patch, patches.reshape(patches.shape[0], -1))

    # ===============================================================================================
    def denoise_pixel(
        self,
        coordinate: Tuple[int, int],
        input_img: np.ndarray,
    ) -> np.ndarray:
        """
        process of denoising one pixel
        """
        weights = self.get_weights_search_area(coordinate, input_img)
        surrounded_img = self.get_search_area(coordinate, input_img)
        totalweights = sum(weights)
        center_img = surrounded_img[
            self.padwidth : surrounded_img.shape[0] - self.padwidth,
            self.padwidth : surrounded_img.shape[1] - self.padwidth,
        ]
        return np.dot(weights, center_img.reshape(-1, 3)) / totalweights

    # ===============================================================================================
    def convert_coordinate_original_img_2_search_area(
        self, coordinate: Tuple[int, int], img: np.ndarray
    ) -> np.ndarray:
        """
        convert the coordinate of the given pixel from original image to its search area
        """
        center_x, center_y = coordinate
        search_area = self.get_search_area(coordinate, img)
        patches, coordinates = self.img2patch(search_area)
        for coordinate, patch in zip(coordinates, patches):
            centerd_patch = img[
                center_x - self.padwidth : center_x + self.padwidth + 1,
                center_y - self.padwidth : center_y + self.padwidth + 1,
                :,
            ]
            res = (centerd_patch == patch).all()
            if res:
                return coordinate

    # ===============================================================================================
    def convert_coordinates_search_area_2_original_img(
        self, coordinate: Tuple[int, int], img: np.ndarray
    ) -> np.ndarray:
        """
        convert the coordinates of the surrounding patches of the given pixel
        from search area to original image
        """
        coordinate_surrounded_img = self.convert_coordinate_original_img_2_search_area(
            coordinate, img
        )
        surrounded_img = self.get_search_area(coordinate, img)
        patches, coordinates = self.img2patch(surrounded_img)
        return coordinates - coordinate_surrounded_img + np.array(coordinate)

    # ===============================================================================================
    def get_list_nearest_coordinates(
        self, coordinate: Tuple[int, int], nb_nbhd: int, img: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        get the list of the nearest patches of the given pixel
        parameter nb_nbhd :  the number of the neares patches
        """
        weights = self.get_weights_search_area(coordinate, img)
        coordinates_orig_img = self.convert_coordinates_search_area_2_original_img(
            coordinate, img
        )
        return coordinates_orig_img[np.argsort(weights)[::-1][:nb_nbhd]].tolist()

    # ===============================================================================================
    def plot_mask_nbhd(
        self,
        coordinate: Tuple[int, int],
        img: np.ndarray,
        path_mask: Path,
        nb_nbhd: int = 5,
    ) -> List[Tuple[int, int]]:
        """
        plot the nearest patches of a pixel
        parameter path_mask : the path for saving the plotted mask
        """
        mask = img.copy()

        # plot the concerned patch
        center_start_x, center_start_y = tuple(np.array(coordinate) - self.padwidth)
        center_end_x, center_end_y = tuple(np.array(coordinate) + self.padwidth)
        mask[center_start_x : center_end_x + 1, center_start_y : center_end_y + 1] = (
            0,
            255,
            0,
        )
        # plot the nearest patches
        res: List[Tuple[int, int]] = []
        list_nearest_coordinates = self.get_list_nearest_coordinates(
            coordinate, nb_nbhd, img
        )
        for coord in list_nearest_coordinates:
            coord = np.array(coord)
            start_point_x, start_point_y = tuple(np.array(coord) - self.padwidth)
            end_point_x, end_point_y = tuple(np.array(coord) + self.padwidth)
            mask[start_point_x : end_point_x + 1, start_point_y : end_point_y + 1] = (
                255,
                0,
                0,
            )
            res.append((start_point_x, start_point_y))
        cv2.imwrite(str(path_mask), mask)
        return res

    # ===============================================================================================
    def crop_img(self, img: np.ndarray) -> np.ndarray:
        return img[
            self.padwidth : img.shape[0] - self.padwidth,
            self.padwidth : img.shape[1] - self.padwidth,
        ]

    # ===============================================================================================
    def __call__(self, path_img: Path) -> np.ndarray:
        assert path_img.is_file(), f"{path_img} does not exist"
        input_img = cv2.imread(str(path_img))
        output_img = np.zeros_like(input_img)

        for i in tqdm(range(self.padwidth, input_img.shape[0] - self.padwidth)):
            for j in range(self.padwidth, input_img.shape[1] - self.padwidth):
                output_img[i, j] = self.denoise_pixel((i, j), input_img)

        return self.crop_img(output_img)
