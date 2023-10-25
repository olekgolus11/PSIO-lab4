import numpy as np
import skimage
from skimage import io
from skimage.filters import rank
from skimage.morphology import disk
import matplotlib.pyplot as plt
from utils import gauss, filtr_medianowy, filtr_usredniajacy, calc_nmse, filtr_gaussa

obraz = skimage.data.camera()
obraz_zaszumiony = gauss(obraz, 0.1)

rozmiary_masek = [1, 2, 3]
sigma_wartosci = [0.5, 0.8, 1.0]

wyniki_nmse = []


for maska in rozmiary_masek:
    obraz_odszumiony_median = filtr_medianowy(obraz_zaszumiony, disk(maska))
    nmse = calc_nmse(obraz, obraz_odszumiony_median)
    wyniki_nmse.append((f"Filtr Medianowy (maska {maska})", nmse))

for maska in rozmiary_masek:
    obraz_odszumiony_usredniajacy = filtr_usredniajacy(obraz_zaszumiony, disk(maska))
    nmse = calc_nmse(obraz, obraz_odszumiony_usredniajacy)
    wyniki_nmse.append((f"Filtr Uśredniający (maska {maska})", nmse))

for sigma in sigma_wartosci:
    obraz_odszumiony_gaussa = filtr_gaussa(obraz_zaszumiony, sigma)
    nmse = calc_nmse(obraz, obraz_odszumiony_gaussa)
    wyniki_nmse.append((f"Filtr Gaussa (sigma {sigma})", nmse))

wyniki_nmse.sort(key=lambda x: x[1])

najlepszy_filtr, najlepsze_nmse = wyniki_nmse[0]

print(f"Najlepszy filtr to: {najlepszy_filtr} z NMSE = {najlepsze_nmse}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(obraz_zaszumiony, cmap='gray')
plt.title("Obraz zaszumiony")

plt.subplot(1, 2, 2)
if "Filtr Medianowy" in najlepszy_filtr:
    plt.imshow(obraz_odszumiony_median, cmap='gray')
elif "Filtr Uśredniający" in najlepszy_filtr:
    plt.imshow(obraz_odszumiony_usredniajacy, cmap='gray')
else:
    plt.imshow(obraz_odszumiony_gaussa, cmap='gray')
plt.title(f"Najlepszy odszumiony obraz ({najlepszy_filtr})")

plt.show()
