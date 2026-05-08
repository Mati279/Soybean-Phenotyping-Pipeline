import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion

from config import SEGMENTATION_PARAMS

def aplicar_otsu_indice(indice_array):
    """
    Calcula el umbral utilizando Otsu sobre una matriz de índice
    y genera la máscara binaria.
    """
    # Filtra los valores nulos.
    indice_valido = indice_array[~np.isnan(indice_array)]
    
    umbral_otsu = threshold_otsu(indice_valido)
    
    # Genera la máscara binaria.
    mascara_vegetacion = indice_array > umbral_otsu
                
    return mascara_vegetacion, umbral_otsu

def apply_mask_to_indices(indices_dict, vegetation_mask):
    """
    Aplica la máscara binaria sacada con otsu al diccionario de índices vegetativos.
    Convierte los píxeles de suelo a valores nulos.
    
    Parámetros:
    - indices_dict: Diccionario con los índices calculados.
    - vegetation_mask: Matriz booleana donde True es vegetación y False es suelo.
    
    Retorna:
    - Un nuevo diccionario con las matrices enmascaradas.
    """
    masked_indices = {}
    
    # Recorre el diccionario.
    for index_name, index_array in indices_dict.items():
        if index_array is not None:
            masked_array = np.copy(index_array)
            
            # Se invierte la máscara y se asigna el valor np.nan a todas esas posiciones seleccionadas.
            masked_array[~vegetation_mask] = np.nan
            
            # Guarda la matriz en el nuevo diccionario.
            masked_indices[index_name] = masked_array
        else:
            # Pasa el valor nulo directo si el índice no existía originalmente.
            masked_indices[index_name] = None
            
    # Devuelve el diccionario con todos los índices.
    return masked_indices