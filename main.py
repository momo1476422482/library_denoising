import cv2
import numpy as np
from pathlib import Path
from model import NLMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
if __name__ == "__main__":

    # original test image
    # test_img = cv2.imread("lena.png")
    # test_img = test_img[:50, :50, :]
    img_width = 50
    test_img = np.ones((img_width, img_width, 3)) * 100
    # test_img[0 : img_width // 2 + 1] = 50
    path_img = Path(__file__).parent / "test_orig.png"
    cv2.imwrite(str(path_img), test_img)

    # corresponding  noisy image
    sigma = 67
    path_noisy_img = Path(__file__).parent / f"noisy_{path_img.stem}_sigma_{sigma}.png"
    test_img_noised = test_img + np.random.randn(*(test_img.shape)) * sigma

    cv2.imwrite(str(path_noisy_img), test_img_noised)

    # denoising
    patch_size = 5
    nlmeans = NLMeans(
        patch_size,
        h=150,
        radius=20,
        sigma=sigma,
    )
    orig_img = nlmeans.crop_img(test_img)
    denoised_img = nlmeans(path_noisy_img)
    print(denoised_img)

    # Test of plotting the nearest patches
    # path_mask = Path(__file__).parent / "mask.png"
    # nlmeans.plot_mask_nbhd((5, 5), test_img, path_mask, nb_nbhd=3)
    cv2.imwrite(f"output_sigma_{sigma}.png", denoised_img)

    # Evaluation of the denoising results
    assert denoised_img.shape == orig_img.shape
    MSE = np.sqrt(np.sum(np.power((denoised_img - orig_img), 2)) / denoised_img.size)
    print("MSE", MSE)

    data = {"ecart-type": sigma, "MSE": MSE}
    path_result = Path(__file__).parent / "result_syn.csv"
    if path_result.is_file() is False:
        data_result = pd.DataFrame.from_dict([data])
        data_result.to_csv(path_result)
    else:
        data_result = pd.read_csv(path_result)
        df = [data_result, pd.DataFrame.from_dict([data])]
        data_result = pd.concat(df)
        data_result.to_csv(path_result, index=False)
