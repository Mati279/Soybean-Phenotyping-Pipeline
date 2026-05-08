"""
Módulo de carga y visualización de ortomosaicos.

El flujo de trabajo se basa en la selección manual de rutas (para dev) y el uso 
de la función read_tif_array para cargar los datos en memoria.
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from rasterio.features import geometry_mask

# Importa las rutas base y sufijos desde Config.py.
from config import ORTOMOSAICOS_DIR, BAND_SUFFIXES

# Construye la ruta dinámica de los archivos.
def get_orthomosaic_path(fecha: str, banda: str) -> str:
    """
    Construye la ruta absoluta al archivo TIF  
    usando las variables de config.py.
    
    Parámetros:
        fecha : str
            Carpeta correspondiente a la fecha de vuelo.
        banda : str
            Identificador de la banda (ej. 'RGB', 'MS', 'NIR').
            
    Retorna:
        str: Ruta completa al archivo.
    """
    # Verifica que la banda solicitada exista.
    if banda not in BAND_SUFFIXES:
        raise ValueError(f"Error: la banda '{banda}' no está definida en config.py")
    
    # Construye el nombre exacto del archivo
    nombre_archivo = f"estanzuela_{fecha}{BAND_SUFFIXES[banda]}"
    
    # Une la ruta base, la fecha y el nombre del archivo.
    ruta_completa = os.path.join(ORTOMOSAICOS_DIR, fecha, nombre_archivo)
    
    return ruta_completa

# Método de carga de geotiffs.
def read_tif_array(file_path: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Lee un archivo TIF y retorna su contenido como un array NumPy
    y su perfil geográfico. 
    
    Parámetros:
     file_path : str
        Ruta al archivo .tif.

    Retorna:
        1. Array con los datos de la imagen como float32.
        2. Perfil geográfico.
        Retorna (None, None) si el archivo no existe, no es GeoTIFF, o hay un error.
    """
    # Verificación de existencia del archivo.
    if not os.path.exists(file_path):
        print(f"Error: archivo no encontrado en: {file_path}")
        return None, None

    try:
        # Abre el archivo TIF.
        with rasterio.open(file_path) as src:
            # Lee los datos casteando a float32 para la Normalización Radiométrica.
            data = src.read(out_dtype=np.float32)
            
            # Obtiene el valor definido como NoData.
            nodata_val = src.nodata
            
            # Si existe un valor NoData convierte esos píxeles a NaN.
            if nodata_val is not None:
                data[data == nodata_val] = np.nan
            
            # Copia el perfil geográfico.
            profile = src.profile.copy()
            
            # Chequeo de que sea un GeoTIFF válido.
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
    Visualiza un ortomosaico como array de NumPy.
    
    Esta función asume que los datos ya están cargados como np.ndarray.

    Parámetros:
        orthomosaic : np.ndarray
        Array de NumPy ya cargado en memoria.
    title : str, op cional
        Título del gráfico.
    """
    # Chequeo de tipo.
    if not isinstance(orthomosaic, np.ndarray) or orthomosaic.size == 0:
        print("Error: se esperaba un array NumPy no vacío para visualizar.")
        return
        
    data = orthomosaic
    
    plt.figure(figsize=(8, 8))

    # Imagen con color.
    if data.shape[0] >= 3:
        # Toma solo las primeras 3 bandas (para tener el RGB).
        display_data = data[:3]
        
        # Transpone de (3, Alto, Ancho) a (Alto, Ancho, 3) para matplotlib.
        rgb_img = np.transpose(display_data, (1, 2, 0)).astype(np.float32)
        
        # Normalizamos para el gráfico.
        max_val = np.nanmax(rgb_img)
        if max_val > 0:
            rgb_img /= max_val     
        plt.imshow(np.clip(rgb_img, 0, 1))

    # Imagen blanco y negro (1 banda, DSM).
    elif data.shape[0] == 1:
        plt.imshow(data[0], cmap="gray")
        
    else:
        # Si tiene 2 bandas mostramos la primera.
        plt.imshow(data[0], cmap="viridis")

    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def aplicar_mascara_poligono(arreglo: np.ndarray, perfil: dict, poligono: Polygon, nombre_capa: str = "Capa") -> np.ndarray:
    """
    Recorta un array usando Shapely.

    """
    geometrias = [poligono]
    
    mascara = geometry_mask(
        geometries=geometrias,                                
        out_shape=(perfil['height'], perfil['width']),        
        transform=perfil['transform'],                        
        invert=True                                           
    )
    
    arreglo_recortado = arreglo.copy()
    
    # 4. Iteramos sobre cada banda.
    for banda in range(arreglo_recortado.shape[0]):
        arreglo_recortado[banda][~mascara] = 0
        
    res_x = abs(perfil['transform'][0])
    res_y = abs(perfil['transform'][4])
    
    print(f"[INFO] Recorte aplicado a: {nombre_capa} | Resol: {res_x:.4f}x{res_y:.4f} m/px")
    
    return arreglo_recortado