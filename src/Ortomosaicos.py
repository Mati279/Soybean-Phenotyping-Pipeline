"""
Módulo de carga y visualización de ortomosaicos

Utilidades para leer y visualizar arrays de imágenes georreferenciadas.

El flujo de trabajo se basa en la selección manual de rutas y el uso 
de la función `read_tif_array` para cargar los datos en memoria.
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Importa las rutas base y sufijos desde la configuración central.
from config import ORTOMOSAICOS_DIR, BAND_SUFFIXES

# Método para construir la ruta dinámica de los archivos.
def get_orthomosaic_path(fecha: str, banda: str) -> str:
    """
    Construye la ruta absoluta al archivo TIF solicitado de manera dinámica 
    usando las variables de config.py.
    
    Parámetros:
        fecha : str
            Carpeta correspondiente a la fecha de vuelo (ej. '10ene').
        banda : str
            Identificador de la banda o producto (ej. 'RGB', 'MS', 'NIR').
            
    Retorna:
        str: Ruta completa al archivo.
    """
    # Verifica que la banda solicitada exista en la configuración
    if banda not in BAND_SUFFIXES:
        raise ValueError(f"Error: la banda '{banda}' no está definida en config.py")
    
    # Construye el nombre exacto del archivo
    nombre_archivo = f"estanzuela_{fecha}{BAND_SUFFIXES[banda]}"
    
    # Une la ruta base, la fecha y el nombre del archivo
    ruta_completa = os.path.join(ORTOMOSAICOS_DIR, fecha, nombre_archivo)
    
    return ruta_completa

# Método de carga de geotiffs.
def read_tif_array(file_path: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Lee un archivo TIF de cualquier ruta y retorna su contenido como un array NumPy
    y su perfil geográfico. Debe ser un archivo GeoTIFF válido. Maneja también los valores NoData.
    
    Parámetros:
     file_path : str
        Ruta al archivo .tif.

    Retorna:
        1. Array con los datos de la imagen como float32.
        2. Perfil geográfico.
        Retorna (None, None) si el archivo no existe, no es GeoTIFF, o hay un error.
    """
    # Verificación de existencia del archivo
    if not os.path.exists(file_path):
        print(f"Error: archivo no encontrado en: {file_path}")
        return None, None

    try:
        # Abre el archivo TIF con rasterio
        with rasterio.open(file_path) as src:
            # Lee los datos, casteando a float32 para la Normalización Radiométrica
            data = src.read(out_dtype=np.float32)
            
            # Obtiene el valor definido como NoData en el archivo
            nodata_val = src.nodata
            
            # Si existe un valor NoData, convierte esos píxeles a NaN para evitar errores en cálculos
            if nodata_val is not None:
                data[data == nodata_val] = np.nan
            
            # Copia el perfil geográfico
            profile = src.profile.copy()
            
            # Chequeo de GeoTIFF válido
            if profile.get('crs') is None:
                print(f"Error: el archivo '{os.path.basename(file_path)}' no es un geotiff válido.")
                return None, None
                
            return data, profile
            
    except Exception as e:
        print(f"Error al leer el archivo {file_path}: {e}")
        return None, None


# Método de visualización de ortomosaicos.
def show_orthomosaic(orthomosaic: np.ndarray, title: str = " ") -> None:
    """
    Visualiza un ortomosaico (como array de NumPy).
    
    Esta función asume que los datos ya están cargados como np.ndarray.
    Maneja arrays para la visualización.

    Parámetros:
        orthomosaic : np.ndarray
        Array de NumPy ya cargado en memoria.
    title : str, opcional
        Título del gráfico.
    """
    # Chequeo de tipo
    if not isinstance(orthomosaic, np.ndarray) or orthomosaic.size == 0:
        print("Error: se esperaba un array NumPy no vacío para visualizar.")
        return
        
    data = orthomosaic
    
    plt.figure(figsize=(8, 8))

    # Imagen con color
    if data.shape[0] >= 3:
        # Si es el MS tomamos solo las primeras 3 (R, G, B/NIR) para que Matplotlib sepa qué dibujar.
        display_data = data[:3]
        
        # Transponemos: De (3, Alto, Ancho) a (Alto, Ancho, 3) para matplotlib
        rgb_img = np.transpose(display_data, (1, 2, 0)).astype(np.float32)
        
        # Normalizamos rápido para visualización
        max_val = np.nanmax(rgb_img)
        if max_val > 0:
            rgb_img /= max_val     
        plt.imshow(np.clip(rgb_img, 0, 1))

    # Imagen blanco y negro (1 banda, DSM)
    elif data.shape[0] == 1:
        # Mostramos la única banda con escala de grises
        plt.imshow(data[0], cmap="gray")
        
    else:
        # Si tiene 2 bandas, mostramos la primera
        plt.imshow(data[0], cmap="viridis")

    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()