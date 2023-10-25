import numpy as np
import skimage
from skimage import io
from skimage.filters import rank
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.metrics import normalized_root_mse

from utils import gauss


def vmf(obraz, maska):
    obraz_odszumiony = np.copy(obraz)
    for i in range(1, obraz.shape[0] - 1):
        for j in range(1, obraz.shape[1] - 1):
            for c in range(3):  # Dla każdego kanału koloru
                okolica = obraz[i-1:i+2, j-1:j+2, c]
                mediana = np.median(okolica[maska])
                obraz_odszumiony[i, j, c] = mediana
    return obraz_odszumiony

def oblicz_nmse(oryginalny_obraz, odszumiony_obraz):
    oryginalny_obraz = oryginalny_obraz.astype(np.float64)
    odszumiony_obraz = odszumiony_obraz.astype(np.float64)
    mse = np.mean((oryginalny_obraz - odszumiony_obraz) ** 2)
    nmse = mse / np.mean(oryginalny_obraz ** 2)
    return nmse

oryginalny_obraz = skimage.data.chelsea()
zaszumiony_obraz = gauss(oryginalny_obraz, 0.1)

procenty_szumu = [2, 5, 10]
rozmiary_masek = [disk(1), disk(2), disk(3)]
wyniki = []

for procent in procenty_szumu:
    for maska in rozmiary_masek:
        obraz_odszumiony_vmf = vmf(zaszumiony_obraz, maska)
        nmse_vmf = oblicz_nmse(oryginalny_obraz, obraz_odszumiony_vmf)
        nmse_standardowy_median = oblicz_nmse(oryginalny_obraz, rank.median(zaszumiony_obraz, mask=maska))
        wyniki.append((f"VMF (maska {maska}), {procent}% szumu", nmse_vmf, nmse_standardowy_median))

print("Wyniki NMSE:")
for nazwa, nmse_vmf, nmse_standardowy_median in wyniki:
    print(f"{nazwa}: NMSE VMF = {nmse_vmf}, NMSE standardowy median = {nmse_standardowy_median}")
