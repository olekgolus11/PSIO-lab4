import numpy as np
import skimage
from skimage import io, util
import random

def zaszum_obraz_impulsowy(obraz, procent):
    obraz_zaszumiony = np.copy(obraz)
    N = int((obraz.shape[0] * obraz.shape[1] * procent) // 100)

    for _ in range(N):
        x = random.randint(0, obraz.shape[1] - 1)
        y = random.randint(0, obraz.shape[0] - 1)
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        obraz_zaszumiony[y, x] = (R, G, B)

    return obraz_zaszumiony

# Wczytaj obraz barwny
obraz = skimage.data.chelsea()

# Zaszum obraz barwny
procent_zaszumienia = 10  # Przykładowa wartość procent
obraz_zaszumiony = zaszum_obraz_impulsowy(obraz, procent_zaszumienia)

# Wyświetl obraz zaszumiony
io.imshow(obraz_zaszumiony)
io.show()
