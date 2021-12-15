from model import NLMeans
import numpy as np
from pathlib import Path


def test_get_search_area():

    radius = 10
    x = 0
    y = 0
    img_x = 100
    img_y = 100
    img_test = np.random.randint(0, 255, size=(img_x, img_y, 3))
    nlmeans = NLMeans(patch_size=3, h=20, radius=radius, sigma=10)
    search_area = nlmeans.get_search_area([x, y], img_test)
    assert search_area.shape[0] == min(img_x, radius + x + 1) - max(0, x - radius)
    assert search_area.shape[1] == min(img_y, radius + y + 1) - max(0, y - radius)


def test_get_search_are():

    radius = 10
    x = 20
    y = 20
    img_x = 100
    img_y = 100
    img_test = np.random.randint(0, 255, size=(img_x, img_y, 3))
    nlmeans = NLMeans(patch_size=3, h=20, radius=radius, sigma=10)
    search_area = nlmeans.get_search_area([x, y], img_test)
    assert search_area.shape[0] == min(img_x, radius + x + 1) - max(0, x - radius)
    assert search_area.shape[1] == min(img_y, radius + y + 1) - max(0, y - radius)


def test_get_list_nearest_coordinates():
    img_test = np.zeros((10, 10, 3))
    img_test[0:5] = 255
    radius = 10
    nlmeans = NLMeans(patch_size=3, h=20, radius=radius, sigma=10)
    res = nlmeans.get_list_nearest_coordinates((5, 4), 8, img_test)
    assert np.unique(np.array(res)[:, 0]).size == 1
    assert np.unique(np.array(res)[:, 1]).size == len(res)


def test_patch2img():
    radius = 10
    img_x = 100
    img_y = 100

    img_test = np.random.randint(0, 255, size=(img_x, img_y, 3))
    nlmeans = NLMeans(patch_size=3, h=20, radius=radius, sigma=10)
    patches, coordinates = nlmeans.img2patch(img_test)
    assert (
        np.abs(nlmeans.patch2img(coordinates, patches, img_test) - img_test).max() == 0
    )
