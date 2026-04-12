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

def apply_mask_to_indices(indices_dict, vegetation_mask):
    """
    Aplica la máscara binaria sacada con otsu al diccionario de índices vegetativos.
    Convierte los píxeles de suelo a valores nulos.
    
    Parámetros:
    - indices_dict: Diccionario con los índices calculados (ej. {"ndvi": array, "gndvi": array}).
    - vegetation_mask: Matriz booleana donde True es vegetación y False es suelo.
    
    Retorna:
    - Un nuevo diccionario con las matrices enmascaradas.
    """
    # Crea un diccionario vacío para guardar los índices limpios sin alterar los originales.
    masked_indices = {}
    
    # Recorre cada par de nombre y matriz dentro del diccionario de índices.
    for index_name, index_array in indices_dict.items():
        # Verifica que el índice exista y no sea nulo.
        if index_array is not None:
            # Copia la matriz original para asegurar que no se modifiquen los datos base.
            masked_array = np.copy(index_array)
            
            # Se invierte la máscara con el operador y se asigna el valor np.nan a todas esas posiciones seleccionadas.
            masked_array[~vegetation_mask] = np.nan
            
            # Guarda la matriz limpia en el nuevo diccionario.
            masked_indices[index_name] = masked_array
        else:
            # Pasa el valor nulo directo si el índice no existía originalmente.
            masked_indices[index_name] = None
            
    # Devuelve el diccionario con todos los índices listos para el análisis de planta pura.
    return masked_indices