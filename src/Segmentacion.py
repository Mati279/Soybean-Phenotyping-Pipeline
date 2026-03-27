import numpy as np
from skimage.filters import threshold_otsu

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
    
    # Devuelve la máscara binaria y el valor numérico del umbral.
    return mascara_vegetacion, umbral_otsu