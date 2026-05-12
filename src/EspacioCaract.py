import numpy as np
import pandas as pd

class EspacioCaracteristicas:
    """
    Clase dedicada exclusivamente a la extracción y construcción del espacio de características.
    Recibe los tensores espaciales procesados y devuelve el DataFrame.
    """

    @staticmethod
    def construir_espacio(resultados_pipeline: dict, nombres_bandas_ms: list = None, nombres_bandas_rgb: list = None) -> pd.DataFrame:
        # Extraemos los componentes necesarios del diccionario generado por el pipeline
        mascara = resultados_pipeline['mascara_binaria']
        ms_final = resultados_pipeline['ms_final']
        rgb_final = resultados_pipeline['rgb_final']
        indices = resultados_pipeline['indices_enmascarados']
        chm = resultados_pipeline.get('chm')
        rugosidades = resultados_pipeline.get('rugosidades', {})

        # Extrae las coordenadas espaciales de los píxeles clasificados como vegetación.
        coords_veg = np.where(mascara)
        rows, cols = coords_veg

        # Inicializamos el diccionario base con la posición.
        feature_dict = {
            'pos_x': cols,
            'pos_y': rows
        }

        # Agregamos las Bandas MS crudas.
        # Si no se pasan nombres explícitos, generamos nombres genéricos.
        nombres_ms = nombres_bandas_ms if nombres_bandas_ms else [f"MS_B{i+1}" for i in range(ms_final.shape[0])]
        # Iteramos sobre las bandas.
        for idx, nombre in enumerate(nombres_ms):
            # Extrae solo los valores que coinciden con coords_veg.
            feature_dict[nombre] = ms_final[idx][coords_veg]

        # Agrega las Bandas RGB crudas.
        nombres_rgb = nombres_bandas_rgb if nombres_bandas_rgb else [f"RGB_B{i+1}" for i in range(rgb_final.shape[0])]
        for idx, nombre in enumerate(nombres_rgb):
            # Extrae los valores correspondientes al espectro visible.
            feature_dict[nombre] = rgb_final[idx][coords_veg]

        # Agregamos los Índices Vegetativosprecalculados.
        for nombre_idx, matriz_idx in indices.items():
            if matriz_idx is not None:
                feature_dict[nombre_idx] = matriz_idx[coords_veg]

        # Agregamos las variables estructurales.
        if chm is not None:
            # Extraemos CHM.
            val_chm = chm[coords_veg]
            feature_dict['chm'] = val_chm
            
            # MÉTRICA DE INGENIERÍA: Vigor relativo a la altura (NDVI / CHM)
            # Detecta relaciones anómalas entre biomasa y estructura física.
            if 'ndvi' in indices and indices['ndvi'] is not None:
                feature_dict['ratio_ndvi_chm'] = indices['ndvi'][coords_veg] / (val_chm + 0.001)

        if rugosidades:
            # Iteramos sobre el diccionario de mapas de rugosidad.
            for nombre_rug, matriz_rug in rugosidades.items():
                nombre_limpio = nombre_rug.split()[0]
                feature_dict[f'rug_{nombre_limpio}'] = matriz_rug[coords_veg]

        # Crea el DataFrame.
        df_features = pd.DataFrame(feature_dict)
        
        # Elimina NaN.
        df_features = df_features.dropna()

        print(f"Espacio de Características listo: {df_features.shape[0]:,} píxeles x {df_features.shape[1]} columnas.")
        
        return df_features