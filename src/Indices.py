import numpy as np

import matplotlib.pyplot as plt

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
    
    # Bandas MS
    B_MS_RED = 0
    B_MS_GREEN = 1
    B_MS_NIR = 2
    B_MS_RED_EDGE = 3
    
    # Bandas RGB
    B_RGB_RED = 0
    B_RGB_GREEN = 1
    B_RGB_BLUE = 2

    def __init__(self, ms_norm_array, rgb_norm_array=None): 
        """
        Inicializa la calculadora con arrays ya normalizados.
        """
        # Si el MS tiene 5 bandas, usar solo las primeras 4 (ignorar ALPHA)
        if ms_norm_array.shape[0] == 5:
            print("Detectadas 5 bandas en MS, usando solo las primeras 4 (ignorando ALPHA)")
            self.ms_array = ms_norm_array[:4, :, :]  # Toma solo bandas 0-3
        else:
            self.ms_array = ms_norm_array
        
        self.rgb_array = rgb_norm_array
        
        self.rgb_red = None
        self.rgb_green = None
        self.rgb_blue = None
        
        # Extrae las bandas del MS
        self.red = self.ms_array[self.B_MS_RED, :, :]
        self.green = self.ms_array[self.B_MS_GREEN, :, :]
        self.nir = self.ms_array[self.B_MS_NIR, :, :]
        self.red_edge = self.ms_array[self.B_MS_RED_EDGE, :, :]
        
        # Extrae las bandas del RGB (si está)
        if self.rgb_array is not None:
            self.rgb_red = self.rgb_array[self.B_RGB_RED, :, :]
            self.rgb_green = self.rgb_array[self.B_RGB_GREEN, :, :]
            self.rgb_blue = self.rgb_array[self.B_RGB_BLUE, :, :]

    def _safe_divide(self, numerator, denominator):
        """
        Realiza la división manejando denominadores cero.
        """
        mask = (denominator == 0) | (np.abs(denominator) < 1e-10)
        result = np.full_like(denominator, np.nan, dtype=np.float64)

        # Manejar caso escalar en numerator
        if np.isscalar(numerator):
            result[~mask] = numerator / denominator[~mask]
        else:
            result[~mask] = numerator[~mask] / denominator[~mask]

        return result

    # Índices principales (usando el MS)   
    def calculate_ndvi(self):
        """NDVI = (NIR - RED) / (NIR + RED)"""
        res = self._safe_divide(self.nir - self.red, self.nir + self.red)
        # Asegura que el resultado esté en el rango [-1, 1]
        return np.clip(res, -1, 1)

    def calculate_ndre(self):
        """NDRE = (NIR - RedEdge) / (NIR + RedEdge)."""
        res = self._safe_divide(self.nir - self.red_edge, self.nir + self.red_edge)
        # Asegura que el resultado esté en el rango [-1, 1]
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

    # Índices adicionales (requiere el blue del RGB)
    def calculate_vari(self):
        """VARI = (Green - Red) / (Green + Red - Blue)."""
        if self.rgb_blue is None:
            return None
        numerator = self.rgb_green - self.rgb_red
        denominator = self.rgb_green + self.rgb_red - self.rgb_blue
        return self._safe_divide(numerator, denominator)

    def calculate_exg(self):
        """ExG = 2*Green - Red - Blue."""
        if self.rgb_blue is None: return None
        return 2 * self.rgb_green - self.rgb_red - self.rgb_blue

    def calculate_evi_hybrid(self):
        """
        EVI simplificado para valores normalizados (0-1).
        
        EVI_simplified = 2.5 * ((NIR - RED) / (NIR + 2.4*RED + 1))
        
        Esta versión ajustada funciona mejor cuando BLUE está normalizado.
        """
        if self.rgb_blue is None: 
            return None
        
        numerator = self.nir - self.red
        # Coeficientes ajustados para valores 0-1
        denominator = self.nir + 2.4 * self.red + 1
        
        return 2.5 * self._safe_divide(numerator, denominator)

    def calculate_main_indices(self):
        """Calcula y devuelve todos los índices principales."""
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
                indices["exg"] = self.calculate_exg() # type: ignore
            if 'EVI' in VALIDATED_INDICES:
                indices["evi"] = self.calculate_evi_hybrid() # type: ignore
            
        return indices
    
    def plot_index(self, index_array, title="Mapa de Índice", cmap='RdYlGn', vmin=None, vmax=None):
        """
        Muestra un mapa de calor de un índice y su histograma.
        Permite definir vmin y vmax..
        """

        # Aplanar y filtrar NaNs para estadísticas y visualización
        valid_data = index_array.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]

        # Establece los límites visuales si no se proveen
        if vmin is None and vmax is None:
            if cmap == 'RdYlGn':
                # Para índices normalizados, usar el rango [-1, 1]
                vmin, vmax = -1, 1
            else:
                # Para otros índices usar percentiles para evitar outliers
                if valid_data.size > 0:
                    vmin = np.nanpercentile(valid_data, 2)
                    vmax = np.nanpercentile(valid_data, 98)
                else:
                    vmin, vmax = 0, 1  # Fallback si no hay datos

        # Configuración de la figura
        fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        # Mapa
        im = ax_map.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_map.set_title(title)
        ax_map.axis('off')
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label='Valor')

        # Histograma
        ax_hist.hist(valid_data, bins=50, color='gray', alpha=0.7, edgecolor='white', range=(vmin, vmax))
        ax_hist.set_title("Distribución de Valores")
        ax_hist.set_xlabel("Valor")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()