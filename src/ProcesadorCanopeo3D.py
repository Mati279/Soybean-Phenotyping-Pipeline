import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, uniform_filter
from scipy.interpolate import griddata

class ProcesadorCanopeo3D:
    def __init__(self, dsm_array, mascara_binaria, percentil_terreno=1.0, indice_banda=0):
        """
        Inicializa el procesador aislando los datos válidos (dejados por el recorte poligonal previo)
        y calculando el nivel base del terreno de forma adaptativa.
        """
        # Extraemos la banda y aseguramos float64 para evitar errores de precisión
        self.dsm_original = dsm_array[indice_banda].copy().astype(np.float64)
        self.mascara_binaria = mascara_binaria
        
        # Máscara de datos válidos: ignora el fondo (NoData) generado por el recorte del polígono
        self.mascara_datos_validos = (self.dsm_original != 0) & (~np.isnan(self.dsm_original))
        
        self.dsm_work = self.dsm_original.copy()
        self.dsm_work[~self.mascara_datos_validos] = np.nan
        
        # El umbral de terreno se calcula estadísticamente sobre los datos reales del lote
        valores_validos = self.dsm_work[self.mascara_datos_validos]
        if len(valores_validos) > 0:
            self.umbral_minimo_terreno = np.percentile(valores_validos, percentil_terreno)
        else:
            self.umbral_minimo_terreno = -np.inf # Fallback de seguridad

    def obtener_mascaras_dilatadas(self, dilation_iterations=5):
        """
        Dilata la máscara de vegetación para asegurar que los bordes del canopeo no contaminen el suelo.
        """
        mascara_veg_valida = self.mascara_binaria & self.mascara_datos_validos
        mascara_dilatada = binary_dilation(mascara_veg_valida, iterations=dilation_iterations)
        mascara_dilatada = mascara_dilatada & self.mascara_datos_validos
        return mascara_veg_valida, mascara_dilatada

    def perforar_dsm(self, mascara_dilatada):
        """
        Elimina los píxeles de vegetación y filtra ruido por debajo del nivel base calculado.
        """
        dsm_perforado = self.dsm_work.copy()
        dsm_perforado[mascara_dilatada] = np.nan
        
        # Usamos el umbral dinámico para limpiar artefactos del sensor (outliers bajos)
        dsm_perforado_clean = dsm_perforado.copy()
        mask_outliers_bajos = (dsm_perforado < self.umbral_minimo_terreno) & ~np.isnan(dsm_perforado)
        dsm_perforado_clean[mask_outliers_bajos] = np.nan
        
        mascara_suelo = ~np.isnan(dsm_perforado_clean) & self.mascara_datos_validos
        return dsm_perforado_clean, mascara_suelo

    def generar_dtm_interpolado(self, dsm_perforado_clean, mascara_suelo):
        """
        Interpola el terreno bajo el canopeo usando Delaunay lineal y rellena huecos con nearest-neighbor.
        """
        rows, cols = self.dsm_work.shape
        grid_c, grid_r = np.meshgrid(np.arange(cols), np.arange(rows))
        
        puntos_xy = np.column_stack([grid_c[mascara_suelo], grid_r[mascara_suelo]])
        valores_z = dsm_perforado_clean[mascara_suelo]
        puntos_query = np.column_stack([grid_c[self.mascara_datos_validos], grid_r[self.mascara_datos_validos]])
        
        # Interpolación principal
        dtm_valores = griddata(puntos_xy, valores_z, puntos_query, method='linear')
        
        dtm = np.full(self.dsm_work.shape, np.nan, dtype=np.float64)
        dtm[self.mascara_datos_validos] = dtm_valores
        
        # Relleno de polígonos cóncavos/bordes
        mascara_huecos = np.isnan(dtm) & self.mascara_datos_validos
        huecos_count = np.sum(mascara_huecos)
        
        if huecos_count > 0:
            dtm_nn = griddata(puntos_xy, valores_z, puntos_query, method='nearest')
            dtm_nn_grid = np.full(self.dsm_work.shape, np.nan)
            dtm_nn_grid[self.mascara_datos_validos] = dtm_nn
            dtm[mascara_huecos] = dtm_nn_grid[mascara_huecos]
            
        return dtm, huecos_count, len(valores_z)

    def suavizar_dtm(self, dtm, gaussian_sigma=15, min_weight_threshold=0.01):
        """
        Aplica un filtro gaussiano al DTM considerando los valores NaN.
        """
        arr = dtm.copy()
        arr[np.isnan(arr)] = 0.0
        pesos = (~np.isnan(dtm)).astype(float)
        
        num = gaussian_filter(arr, sigma=gaussian_sigma)
        den = gaussian_filter(pesos, sigma=gaussian_sigma)
        
        dtm_gaus = np.divide(num, den, out=np.full_like(num, np.nan), where=den > min_weight_threshold)
        dtm_gaus[~self.mascara_datos_validos] = np.nan
        return dtm_gaus

    def calcular_chm(self, dtm_suavizado, chm_clip_min=0):
        """
        Calcula el Modelo de Altura de Canopeo restando el DTM al DSM y filtrando por la máscara original.
        """
        chm = self.dsm_work - dtm_suavizado
        chm[~self.mascara_binaria] = np.nan
        chm = np.clip(chm, chm_clip_min, None)
        return chm

    @staticmethod
    def calcular_rugosidad_local(chm_2d, ventana=5, valor_nan_sustituto=0.0):
        """
        Calcula la desviación estándar local (rugosidad) mediante filtros uniformes.
        """
        chm_seguro = np.nan_to_num(chm_2d, nan=valor_nan_sustituto)
        
        media_cuadrados = uniform_filter(chm_seguro ** 2, size=ventana)
        cuadrado_media = uniform_filter(chm_seguro, size=ventana) ** 2
        
        varianza = np.clip(media_cuadrados - cuadrado_media, 0, None)
        return np.sqrt(varianza)

    def calcular_multiples_rugosidades(self, chm, ventanas_dict, valor_nan_sustituto=0.0):
        """
        Genera un diccionario de arreglos de rugosidad para distintas escalas espaciales.
        """
        rugosidades = {}
        for nombre, v in ventanas_dict.items():
            rug = self.calcular_rugosidad_local(chm, ventana=v, valor_nan_sustituto=valor_nan_sustituto)
            rug[~self.mascara_binaria] = np.nan
            rugosidades[nombre] = rug
        return rugosidades