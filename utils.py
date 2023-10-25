import numpy as np
import skimage
from skimage import io, util, filters
from skimage.filters import rank

def salt_and_pepper(obraz, procent):
    return skimage.img_as_ubyte(skimage.util.random_noise(obraz, 's&p', amount=procent / 100))

def gauss(obraz, s):
    return skimage.img_as_ubyte(skimage.util.random_noise(obraz, var=s * s))

def linear(obraz, h):
    maska = np.random.uniform(-h, h, obraz.shape)
    return np.clip(obraz + maska, 0, 255).astype(np.uint8)

def calc_nmse(original_image, noised_image):
    original_image = original_image.astype(np.float64)
    noised_image = noised_image.astype(np.float64)
    mse = np.sum(np.square(original_image - noised_image))
    mse /= np.sum(np.square(original_image))
    nmse = mse
    return nmse

def filtr_medianowy(obraz, maska):
    obraz_odszumiony = rank.median(obraz, mask=maska)
    return obraz_odszumiony

def filtr_usredniajacy(obraz, maska):
    obraz_odszumiony = rank.mean(obraz, footprint=maska)
    return obraz_odszumiony

def filtr_gaussa(obraz, sigma):
    obraz_odszumiony = filters.gaussian(obraz, sigma=sigma)
    obraz_odszumiony = skimage.img_as_ubyte(obraz_odszumiony)
    return obraz_odszumiony

def oblicz_nmse_kolorowy(oryginalny_obraz, odszumiony_obraz):
    oryginalny_obraz = oryginalny_obraz.astype(np.float64)
    odszumiony_obraz = odszumiony_obraz.astype(np.float64)

    mse_r = np.sum(np.square(oryginalny_obraz[:, :, 0] - odszumiony_obraz[:, :, 0]))
    mse_g = np.sum(np.square(oryginalny_obraz[:, :, 1] - odszumiony_obraz[:, :, 1]))
    mse_b = np.sum(np.square(oryginalny_obraz[:, :, 2] - odszumiony_obraz[:, :, 2]))
    mse = (mse_r + mse_g + mse_b) / 3
    mse /= np.sum(np.square(oryginalny_obraz))
    nmse = mse
    return nmse
