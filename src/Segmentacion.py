import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion

from config import SEGMENTATION_PARAMS

def apply_otsu_ndvi(ndvi_array):
    """
    Calcula el umbral utilizando Otsu sobre un array de NDVI
    y genera la máscara binaria.
    """
    # Filtra los valores nulos.
    ndvi_valido = ndvi_array[~np.isnan(ndvi_array)]
    
    # Calcula el valor de corte.
    umbral_otsu = threshold_otsu(ndvi_valido)
    
    # Genera una matriz de verdaderos y falsos. 
    mascara_vegetacion = ndvi_array > umbral_otsu
    
    # Aplica buffer negativo si está habilitado en config.py
    if SEGMENTATION_PARAMS.get('apply_negative_buffer', False):
        pixels = SEGMENTATION_PARAMS.get('buffer_pixels', 0)
        if pixels > 0:
            mascara_vegetacion = binary_erosion(mascara_vegetacion, iterations=pixels)
    
    # Devuelve la máscara binaria y el valor numérico del umbral.
    return mascara_vegetacion, umbral_otsu