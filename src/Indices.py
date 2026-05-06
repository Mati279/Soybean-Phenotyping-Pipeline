import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Asume que VALIDATED_INDICES viene de tu archivo config
from config import VALIDATED_INDICES

class VegetationIndices:
    """
    Calculadora de índices vegetativos.
    
    Trabaja con los outputs normalizados de "Normalizacion.process_session".
    
    Mapeo de Bandas MS:
    0: RED
    1: GREEN
    2: NIR
    3: RED EDGE
    4: ALPHA
    """
    
    B_MS_RED = 0
    B_MS_GREEN = 1
    B_MS_NIR = 2
    B_MS_RED_EDGE = 3
    
    B_RGB_RED = 0
    B_RGB_GREEN = 1
    B_RGB_BLUE = 2

    def __init__(self, ms_norm_array, rgb_norm_array=None): 
        """
        Inicializa la calculadora con arrays ya normalizados.
        """
        # Descarta la banda ALPHA si el archivo MS contiene 5 bandas.
        if ms_norm_array.shape[0] == 5:
            print("Detectadas 5 bandas en MS. Asigna solo las primeras 4 (ignora ALPHA).")
            self.ms_array = ms_norm_array[:4, :, :]
        else:
            self.ms_array = ms_norm_array
        
        self.rgb_array = rgb_norm_array
        
        self.rgb_red = None
        self.rgb_green = None
        self.rgb_blue = None
        
        # Extrae y asigna las bandas multiespectrales a variables individuales.
        self.red = self.ms_array[self.B_MS_RED, :, :]
        self.green = self.ms_array[self.B_MS_GREEN, :, :]
        self.nir = self.ms_array[self.B_MS_NIR, :, :]
        self.red_edge = self.ms_array[self.B_MS_RED_EDGE, :, :]
        
        # Extrae y asigna las bandas RGB si el arreglo fue provisto.
        if self.rgb_array is not None:
            self.rgb_red = self.rgb_array[self.B_RGB_RED, :, :]
            self.rgb_green = self.rgb_array[self.B_RGB_GREEN, :, :]
            self.rgb_blue = self.rgb_array[self.B_RGB_BLUE, :, :]

    def _safe_divide(self, numerator, denominator):
        """
        Realiza la división matemática. Evita errores por división por cero.
        """
        # Genera una máscara booleana donde el denominador es cero o extremadamente pequeño.
        mask = (denominator == 0) | (np.abs(denominator) < 1e-10)
        
        # Crea un arreglo base lleno de valores nulos (NaN).
        result = np.full_like(denominator, np.nan, dtype=np.float64)

        # Ejecuta la división únicamente en los píxeles donde la máscara es falsa.
        if np.isscalar(numerator):
            result[~mask] = numerator / denominator[~mask]
        else:
            result[~mask] = numerator[~mask] / denominator[~mask]

        return result

    def calculate_ndvi(self):
        """NDVI = (NIR - RED) / (NIR + RED)"""
        res = self._safe_divide(self.nir - self.red, self.nir + self.red)
        # Limita los valores matemáticos al rango estricto [-1, 1].
        return np.clip(res, -1, 1)

    def calculate_ndre(self):
        """NDRE = (NIR - RedEdge) / (NIR + RedEdge)."""
        res = self._safe_divide(self.nir - self.red_edge, self.nir + self.red_edge)
        return np.clip(res, -1, 1)

    def calculate_savi(self, L=0.5): 
        """SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)"""
        numerator = self.nir - self.red
        denominator = self.nir + self.red + L 
        res = self._safe_divide(numerator, denominator) * (1 + L)
        return res
    
    def calculate_gndvi(self):
        """GNDVI = (NIR - Green) / (NIR + Green)"""
        res = self._safe_divide(self.nir - self.green, self.nir + self.green)
        return np.clip(res, -1, 1)

    def calculate_vari(self):
        """VARI = (Green - Red) / (Green + Red - Blue)."""
        if self.rgb_blue is None:
            return None
        numerator = self.rgb_green - self.rgb_red
        denominator = self.rgb_green + self.rgb_red - self.rgb_blue
        res = self._safe_divide(numerator, denominator)
        return np.clip(res, -1, 1)

    def calculate_exg(self):
        """ExG = 2*Green - Red - Blue."""
        if self.rgb_blue is None: 
            return None
        return 2 * self.rgb_green - self.rgb_red - self.rgb_blue

    def calculate_gi(self):
        """
        Calcula el Green Index..
        El GI evalúa el grado de verdor con base en una escala empírica de 0 a 255.
        Convierte los valores normalizados temporalmente a la escala 0-255.
        """
        if self.rgb_blue is None: 
            return None
        # Des-normaliza temporalmente las matrices al rango 0-255 esperado por la fórmula empírica.
        red_255 = self.rgb_red * 255.0
        green_255 = self.rgb_green * 255.0
        blue_255 = self.rgb_blue * 255.0
        
        # Calcula la proximidad de cada canal a los valores óptimos de verde definidos en el paper.
        # Usa el valor absoluto para medir la distancia respecto a G=165, R=37.5 y B=37.5.
        componente_g = 255.0 - np.abs(green_255 - 165.0)
        componente_r = 255.0 - np.abs(red_255 - 37.5)
        componente_b = 255.0 - np.abs(blue_255 - 37.5)
        
        # Suma los componentes y normaliza por el máximo teórico (3 * 255).
        pre_gi = (componente_g + componente_r + componente_b) / (3.0 * 255.0)
        
        # Calcula el denominador final restando el pre-GI a 1.
        denominador = 1.0 - pre_gi
        
        # Aplica la división segura para evitar errores y aplica el factor de escalado del paper (1/12).
        gi_final = self._safe_divide(pre_gi, denominador) / 12.0
        
        # Devuelve la matriz matemática calculada al diccionario principal.
        return gi_final
    
    def calculate_evi_hybrid(self):
        """EVI ajustado para datos de reflectancia previamente normalizados."""
        numerator = self.nir - self.red
        denominator = self.nir + 2.4 * self.red + 1
        return 2.5 * self._safe_divide(numerator, denominator)

    def calculate_main_indices(self):
        """
        Ejecuta el cálculo masivo de los índices habilitados en la configuración.
        Devuelve un diccionario con las matrices resultantes.
        """
        indices = {}
        
        if 'NDVI' in VALIDATED_INDICES:
            indices["ndvi"] = self.calculate_ndvi()
        if 'NDRE' in VALIDATED_INDICES:
            indices["ndre"] = self.calculate_ndre()
        if 'SAVI' in VALIDATED_INDICES:
            indices["savi"] = self.calculate_savi()
        if 'GNDVI' in VALIDATED_INDICES:
            indices["gndvi"] = self.calculate_gndvi()
            
        if self.rgb_array is not None:
            if 'VARI' in VALIDATED_INDICES:
                indices["vari"] = self.calculate_vari()
            if 'ExG' in VALIDATED_INDICES:
                indices["exg"] = self.calculate_exg()
            if 'GI' in VALIDATED_INDICES:
                indices["gi"] = self.calculate_gi()
            if 'EVI' in VALIDATED_INDICES:
                indices["evi"] = self.calculate_evi_hybrid()
                
        return indices
    
    def plot_index(self, index_array, title="Mapa de Índice", cmap='RdYlGn', auto_scale=True):
        """
        Genera la visualización dual: mapa de calor e histograma coloreado.
        El parámetro auto_scale fuerza el cálculo de percentiles para estirar el contraste visual.
        """
        # Aplana la matriz a un vector 1D y elimina los valores nulos (fondo).
        valid_data = index_array.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]

        # Cancela la ejecución si no hay datos válidos en el recorte.
        if valid_data.size == 0:
            print("No hay datos válidos para graficar.")
            return

        # Calcula los percentiles 2 y 98 para descartar outliers y estirar la escala.
        if auto_scale:
            vmin = np.nanpercentile(valid_data, 2)
            vmax = np.nanpercentile(valid_data, 98)
        else:
            # Fallback a los límites matemáticos absolutos.
            vmin, vmax = -1, 1

        # Prepara la figura con dos columnas: mapa (izquierda) e histograma (derecha).
        fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        # Renderiza el mapa espacial.
        im = ax_map.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_map.set_title(title)
        ax_map.axis('off')
        
        # Agrega la barra de referencias al mapa.
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label='Valor del Índice')

        # Dibuja el histograma base y captura los componentes generados (bins y barras).
        n, bins, patches = ax_hist.hist(valid_data, bins=50, edgecolor='black', linewidth=0.5, range=(vmin, vmax))
        
        # Configura el normalizador para mapear los valores de los bins a colores exactos.
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
        
        # Itera sobre cada barra del histograma para asignarle el color correspondiente.
        for bin_val, patch in zip(bins, patches):
            color = colormap(norm(bin_val))
            patch.set_facecolor(color)

        # Ajusta las etiquetas y el estilo del histograma.
        ax_hist.set_title("Distribución de Valores (Auto-ajustado)")
        ax_hist.set_xlabel("Valor del Índice")
        ax_hist.set_ylabel("Cantidad de Píxeles")
        ax_hist.grid(axis='y', linestyle='--', alpha=0.5)

        # Renderiza el lienzo completo.
        plt.tight_layout()
        plt.show()
    
    def exportar_metricas_indices(diccionario_resultados, fecha, directorio_salida="../outputs"):
        """
        Calcula estadísticas descriptivas para cada índice y las exporta a un archivo JSON.
        Ignora los valores nulos (NaN) provenientes del fondo.
        """
        # Crea el directorio de salida si no existe
        os.makedirs(directorio_salida, exist_ok=True)
        
        resumen_metricas = {}
        
        print("Calculando métricas para exportación...")
        
        # Itera sobre cada índice calculado en el diccionario de resultados
        for nombre_indice, matriz in diccionario_resultados.items():
            # Aplana la matriz y filtra los valores NaN
            datos_validos = matriz.flatten()
            datos_validos = datos_validos[~np.isnan(datos_validos)]
            
            # Verifica que existan datos válidos antes de calcular estadísticas
            if len(datos_validos) > 0:
                resumen_metricas[nombre_indice.upper()] = {
                    "Minimo": float(np.min(datos_validos)),
                    "Maximo": float(np.max(datos_validos)),
                    "Media": float(np.mean(datos_validos)),
                    "Desviacion_Estandar": float(np.std(datos_validos)),
                    "Percentil_25": float(np.percentile(datos_validos, 25)),
                    "Mediana_50": float(np.percentile(datos_validos, 50)),
                    "Percentil_75": float(np.percentile(datos_validos, 75))
                }
            else:
                resumen_metricas[nombre_indice.upper()] = "Sin datos válidos"

        # Define la ruta del archivo utilizando la fecha para mantener orden
        ruta_archivo = os.path.join(directorio_salida, f"metricas_indices_{fecha}.json")
        
        # Exporta el diccionario a un archivo JSON con formato legible (indent=4)
        with open(ruta_archivo, 'w') as f:
            json.dump(resumen_metricas, f, indent=4)
            
        print(f"Métricas exportadas exitosamente en: {ruta_archivo}")