import numpy as np
import skimage
from skimage import io, util
from skimage.filters import gaussian, median
from skimage.metrics import normalized_root_mse
import matplotlib.pyplot as plt

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

image = skimage.data.camera()

sp_percentages = [5, 10, 20]
gauss_deviations = [0.05, 0.08, 0.1]
linear_ranges = [10, 20, 40]

noise_images = []

for procent in sp_percentages:
    image_saltpepper = salt_and_pepper(image, procent)
    noise_images.append(image_saltpepper)

for s in gauss_deviations:
    image_gauss = gauss(image, s)
    noise_images.append(image_gauss)

for h in linear_ranges:
    image_linear = linear(image, h)
    noise_images.append(image_linear)

nmse_results = []

for noised_image in noise_images:
    nmse = calc_nmse(image, noised_image)
    nmse_results.append(nmse)

print("Wyniki NMSE:")
for i, wynik in enumerate(nmse_results):
    print(f"Obraz {i + 1}: {wynik}")

for i, noised_image in enumerate(noise_images):
    plt.subplot(4, 3, i + 1)
    plt.imshow(noised_image, cmap='gray')
    plt.title(f"Obraz {i + 1}")

plt.subplot(4, 3, 11)
plt.imshow(image, cmap='gray')
plt.title("Obraz oryginalny")

plt.gcf().set_size_inches(10.5, 10.5)

plt.show()