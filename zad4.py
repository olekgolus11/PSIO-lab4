import numpy as np
from skimage import io

from utils import oblicz_nmse_kolorowy

# Wczytaj obraz oryginalny i odszumiony
oryginalny_obraz = io.imread("sciezka/do/twojego/oryginalnego_obrazu.jpg")
odszumiony_obraz = io.imread("sciezka/do/twojego/odszumionego_obrazu.jpg")

# Oblicz NMSE
nmse = oblicz_nmse_kolorowy(oryginalny_obraz, odszumiony_obraz)

print(f"Wartość NMSE: {nmse}")