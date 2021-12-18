import cv2
import numpy as np
from pathlib import Path
from model import NLMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility_plot import visualise_data


# ============================================================================
if __name__ == "__main__":

    # original test image
    image_name = "synthetic"
    img_width = 50
    if image_name == "synthetic":
        test_img = np.ones((img_width, img_width, 3)) * 100
    elif image_name == "lena":
        test_img = cv2.imread("lena.png")
        test_img = test_img[:img_width, :img_width, :]

    path_img = Path(__file__).parent / f"{image_name}.png"
    cv2.imwrite(str(path_img), test_img)
    path2csv = Path(__file__).parent / "result_nlmeans.csv"
    if image_name == "lena":
        path2csv.unlink()
    for h in range(200, 600, 100):
        for sigma in range(10, 50, 10):
            nlmeans = NLMeans(patch_size=7, h=h, radius=10, sigma=sigma)
            # denoising
            nlmeans(path_img, sigma)
    path_result = Path(__file__).parent / "result_nlmeans.csv"
    df = pd.read_csv(path_result)
    g = lambda x: x["image_name"] + "_" + str(x["h"])
    df["hue"] = df.apply(g, 1)
    vd = visualise_data(df)
    vd.line_plot("sigma", "RMSE", "hue", Path(__file__).parent / "curve.png")
