"""
PipelineOrtomosaico.py
---------------
Centraliza en un único método el procesamiento completo de un ortomosaico
(carga → recorte ROI → normalización → índices → Otsu → CHM → rugosidad)
tal como se hace en el notebook de análisis exploratorio.
 
Uso mínimo
----------
    from PipelineOrtomosaico import PipelineOrtomosaico
 
    resultado = PipelineOrtomosaico.procesar(
        fecha="17ene",
        poligono=None,          # None → usa COORDENADAS_DEFAULT
        mostrar_graficos=True
    )
 
El diccionario devuelto contiene todas las matrices y métricas
listas para el EDA.
"""
 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
 
# Agrega src al path si es necesario (ajustar según estructura del proyecto).
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
 
from Normalizacion import process_session
from Ortomosaicos import get_orthomosaic_path, read_tif_array, show_orthomosaic, aplicar_mascara_poligono
from Indices import VegetationIndices
from Segmentacion import aplicar_otsu_indice, apply_mask_to_indices
from SelectorROI import SelectorROI
from ProcesadorCanopeo3D import ProcesadorCanopeo3D
 
 
# ---------------------------------------------------------------------------
# Coordenadas de fallback para desarrollo (modo sin selector interactivo).
# ---------------------------------------------------------------------------
COORDENADAS_DEFAULT = [
    (436866.364641592, 6200265.648429024),
    (436868.3868145593, 6200247.737754172),
    (436901.3193457407, 6200251.782100106),
    (436898.21386582666, 6200269.909436349),
    (436866.364641592, 6200265.648429024),
]
 
 
class PipelineOrtomosaico:
    """
    Pipeline de procesamiento completo para un ortomosaico dado.
    Todas las etapas del notebook quedan encapsuladas en `procesar()`.
    """
 
    # ------------------------------------------------------------------
    # Parámetros con valores por defecto (se pueden sobreescribir).
    # ------------------------------------------------------------------
    INDICE_OTSU: str = "ndvi"
    TAMANO_MINIMO_PIXELES: int = 50
    PERCENTIL_TERRENO: float = 1.5
    DILATION_ITERATIONS: int = 5
    GAUSSIAN_SIGMA: int = 15
    MIN_WEIGHT_THRESHOLD: float = 0.01
    CHM_CLIP_MIN: float = 0.0
    VENTANAS_RUGOSIDAD: dict = {
        "25cm (hoja)": 5,
        "55cm (planta)": 11,
        "105cm (surco)": 21,
    }
    PALETAS_INDICES: dict = {
        "ndvi": "RdYlGn",
        "ndre": "viridis",
        "savi": "inferno",
        "gndvi": "YlGn",
        "exg": "plasma",
        "gi": "cividis",
    }
 
    @staticmethod
    def procesar(
        fecha: str,
        poligono=None,
        coordenadas_default: list = None,
        mostrar_graficos: bool = True,
        exportar_metricas: bool = True,
        factor_reduccion_roi: int = 4,
        # Parámetros de Otsu y limpieza
        indice_otsu: str = None,
        tamano_minimo_pixeles: int = None,
        # Parámetros 3D
        percentil_terreno: float = None,
        dilation_iterations: int = None,
        gaussian_sigma: int = None,
        chm_clip_min: float = None,
        ventanas_rugosidad: dict = None,
    ) -> dict:
        """
        Ejecuta el pipeline completo sobre la fecha indicada.
 
        Parámetros
        ----------
        fecha : str
            Identificador de la sesión de vuelo (ej. "17ene").
        poligono : shapely.geometry.Polygon, opcional
            Polígono de recorte ya construido. Si es None se usará el
            SelectorROI interactivo (o las coordenadas por defecto).
        coordenadas_default : list, opcional
            Lista de tuplas (x, y) en CRS del raster para usar como
            fallback sin interacción. Si es None usa COORDENADAS_DEFAULT.
        mostrar_graficos : bool
            Si False suprime todas las visualizaciones (útil en CI/batch).
        exportar_metricas : bool
            Si True exporta el JSON de métricas de índices.
        factor_reduccion_roi : int
            Factor de reducción para la previsualización del SelectorROI.
        indice_otsu, tamano_minimo_pixeles, percentil_terreno,
        dilation_iterations, gaussian_sigma, chm_clip_min,
        ventanas_rugosidad : opcionales
            Sobreescriben los valores por defecto de la clase.
 
        Retorna
        -------
        dict con claves:
            rgb_final, ms_final, dsm_final,
            indices_crudos, indices_enmascarados,
            mascara_binaria, chm, rugosidades,
            procesador_3d, calculadora_indices
        """
 
        # -- Resolución de parámetros con fallback a defaults de clase -----
        _coords_default = coordenadas_default or COORDENADAS_DEFAULT
        _indice_otsu = indice_otsu or PipelineOrtomosaico.INDICE_OTSU
        _min_px = tamano_minimo_pixeles if tamano_minimo_pixeles is not None else PipelineOrtomosaico.TAMANO_MINIMO_PIXELES
        _pct_terreno = percentil_terreno if percentil_terreno is not None else PipelineOrtomosaico.PERCENTIL_TERRENO
        _dil_iter = dilation_iterations if dilation_iterations is not None else PipelineOrtomosaico.DILATION_ITERATIONS
        _gauss = gaussian_sigma if gaussian_sigma is not None else PipelineOrtomosaico.GAUSSIAN_SIGMA
        _chm_min = chm_clip_min if chm_clip_min is not None else PipelineOrtomosaico.CHM_CLIP_MIN
        _ventanas = ventanas_rugosidad or PipelineOrtomosaico.VENTANAS_RUGOSIDAD
 
        # ==================================================================
        # 1. CARGA DE ORTOMOSAICOS
        # ==================================================================
        print(f"\n{'='*60}")
        print(f"  PIPELINE — Fecha: {fecha}")
        print(f"{'='*60}\n")
 
        ruta_rgb = get_orthomosaic_path(fecha, "RGB")
        ruta_ms = get_orthomosaic_path(fecha, "MS")
 
        ruta_dsm = None
        if fecha != "10ene":
            ruta_dsm = get_orthomosaic_path(fecha, "DSM")
 
        ortomosaico_rgb_array, ortomosaico_rgb_perfil = read_tif_array(ruta_rgb)
        ortomosaico_ms_array, ortomosaico_ms_perfil = read_tif_array(ruta_ms)
 
        if ruta_dsm is not None:
            ortomosaico_dsm_array, ortomosaico_dsm_perfil = read_tif_array(ruta_dsm)
        else:
            ortomosaico_dsm_array, ortomosaico_dsm_perfil = None, None
 
        print("Ortomosaicos cargados.")
 
        # ==================================================================
        # 2. SELECCIÓN / RECORTE DE ROI
        # ==================================================================
        if poligono is None:
            # Intenta usar el selector interactivo; si no hay display activo
            # cae directo a las coordenadas default.
            try:
                selector = SelectorROI(ruta_rgb, factor_reduccion=factor_reduccion_roi)
                if mostrar_graficos:
                    selector.mostrar_interfaz()
                poligono_final = selector.obtener_poligono_shapely(
                    coordenadas_default=_coords_default
                )
            except Exception:
                from shapely.geometry import Polygon
                print("SelectorROI no disponible — usando coordenadas default.")
                poligono_final = Polygon(_coords_default)
        else:
            poligono_final = poligono
 
        rgb_recortado = aplicar_mascara_poligono(
            ortomosaico_rgb_array, ortomosaico_rgb_perfil, poligono_final, "RGB"
        )
        ms_recortado = aplicar_mascara_poligono(
            ortomosaico_ms_array, ortomosaico_ms_perfil, poligono_final, "Multiespectral"
        )
        dsm_recortado = None
        if ortomosaico_dsm_array is not None:
            dsm_recortado = aplicar_mascara_poligono(
                ortomosaico_dsm_array, ortomosaico_dsm_perfil, poligono_final, "DSM"
            )
 
        print("Recorte de ROI finalizado.")
 
        # ==================================================================
        # 3. ALINEACIÓN Y NORMALIZACIÓN
        # ==================================================================
        sesion_normalizada = process_session(
            ms_data=ms_recortado,
            ms_profile=ortomosaico_ms_perfil,
            rgb_data=rgb_recortado,
            rgb_profile=ortomosaico_rgb_perfil,
            dsm_data=dsm_recortado,
            dsm_profile=ortomosaico_dsm_perfil,
        )
 
        ms_final = sesion_normalizada["ms"]
        rgb_final = sesion_normalizada["rgb"]
        dsm_final = sesion_normalizada.get("dsm", None)
 
        print(f"MS:  {ms_final.shape}  |  rango [{np.nanmin(ms_final):.3f}, {np.nanmax(ms_final):.3f}]")
        print(f"RGB: {rgb_final.shape}  |  rango [{np.nanmin(rgb_final):.3f}, {np.nanmax(rgb_final):.3f}]")
        if dsm_final is not None:
            print(f"DSM: {dsm_final.shape}")
 
        # ==================================================================
        # 4. CÁLCULO DE ÍNDICES
        # ==================================================================
        ms_final_nan = np.where(ms_final == 0, np.nan, ms_final)
        rgb_final_nan = np.where(rgb_final == 0, np.nan, rgb_final)
 
        calculadora = VegetationIndices(
            ms_norm_array=ms_final_nan, rgb_norm_array=rgb_final_nan
        )
        indices_crudos = calculadora.calculate_main_indices()
        print(f"Índices calculados: {list(indices_crudos.keys())}")
 
        # ==================================================================
        # 5. SEGMENTACIÓN CON OTSU + LIMPIEZA
        # ==================================================================
        indice_array = indices_crudos[_indice_otsu]
        mascara_binaria, umbral_otsu = aplicar_otsu_indice(indice_array)
 
        # Elimina objetos pequeños (malezas aisladas / ruido).
        mascara_binaria = remove_small_objects(mascara_binaria, min_size=_min_px)
 
        print(f"Otsu ({_indice_otsu.upper()}): umbral={umbral_otsu:.4f} | "
              f"píxeles vegetación={np.sum(mascara_binaria):,}")
 
        # Aplica la máscara a todos los índices.
        indices_enmascarados = apply_mask_to_indices(indices_crudos, mascara_binaria)
 
        if exportar_metricas:
            VegetationIndices.exportar_metricas_indices(indices_enmascarados, fecha)
 
        # ==================================================================
        # 6. DSM → DTM → CHM → RUGOSIDAD  (solo si hay DSM)
        # ==================================================================
        chm = None
        rugosidades = {}
        procesador_3d = None
 
        if dsm_final is not None:
            procesador_3d = ProcesadorCanopeo3D(
                dsm_array=dsm_final,
                mascara_binaria=mascara_binaria,
                percentil_terreno=_pct_terreno,
            )
            print(f"Umbral terreno dinámico: {procesador_3d.umbral_minimo_terreno:.2f}m")
 
            mascara_veg_valida, mascara_dilatada = procesador_3d.obtener_mascaras_dilatadas(
                dilation_iterations=_dil_iter
            )
 
            dsm_perforado_clean, mascara_suelo = procesador_3d.perforar_dsm(mascara_dilatada)
 
            dtm, huecos, pts_control = procesador_3d.generar_dtm_interpolado(
                dsm_perforado_clean, mascara_suelo
            )
            print(f"DTM: {pts_control:,} puntos de control | {huecos:,} huecos residuales")
 
            dtm = procesador_3d.suavizar_dtm(
                dtm,
                gaussian_sigma=_gauss,
                min_weight_threshold=0.01,
            )
 
            chm = procesador_3d.calcular_chm(dtm, chm_clip_min=_chm_min)
            print(f"CHM — Media: {np.nanmean(chm):.4f}m | Max: {np.nanmax(chm):.3f}m")
 
            rugosidades = procesador_3d.calcular_multiples_rugosidades(
                chm, _ventanas, valor_nan_sustituto=0.0
            )
            print(f"Rugosidades calculadas: {list(rugosidades.keys())}")
        else:
            print("Sin DSM — etapas 3D omitidas.")
 
        # ==================================================================
        # 7. VISUALIZACIONES (opcionales)
        # ==================================================================
        if mostrar_graficos:
            PipelineOrtomosaico._graficar_resumen(
                fecha=fecha,
                rgb_final=rgb_final,
                ms_final=ms_final,
                dsm_final=dsm_final,
                indices_crudos=indices_crudos,
                indices_enmascarados=indices_enmascarados,
                mascara_binaria=mascara_binaria,
                umbral_otsu=umbral_otsu,
                indice_otsu=_indice_otsu,
                chm=chm,
                rugosidades=rugosidades,
                calculadora=calculadora,
            )
 
        print(f"\n{'='*60}")
        print(f"  Pipeline completado — {fecha}")
        print(f"{'='*60}\n")
 
        return {
            "rgb_final": rgb_final,
            "ms_final": ms_final,
            "dsm_final": dsm_final,
            "indices_crudos": indices_crudos,
            "indices_enmascarados": indices_enmascarados,
            "mascara_binaria": mascara_binaria,
            "chm": chm,
            "rugosidades": rugosidades,
            "procesador_3d": procesador_3d,
            "calculadora_indices": calculadora,
        }
 
    # ------------------------------------------------------------------
    # Visualizaciones internas
    # ------------------------------------------------------------------
    @staticmethod
    def _graficar_resumen(
        fecha, rgb_final, ms_final, dsm_final,
        indices_crudos, indices_enmascarados, mascara_binaria, umbral_otsu,
        indice_otsu, chm, rugosidades, calculadora,
    ):
        """Genera las visualizaciones de resumen del pipeline."""
 
        show_orthomosaic(rgb_final, title=f"RGB Normalizado — {fecha}")
        show_orthomosaic(ms_final, title=f"MS Normalizado — {fecha}")
 
        # DSM
        if dsm_final is not None:
            dsm_vis = dsm_final[0].copy()
            dsm_vis[dsm_vis == 0] = np.nan
            vals_validos = dsm_vis[dsm_vis > 1] if np.any(dsm_vis > 1) else dsm_vis[~np.isnan(dsm_vis)]
            vmin = np.percentile(vals_validos, 2) if len(vals_validos) else None
            vmax = np.percentile(vals_validos, 98) if len(vals_validos) else None
            plt.figure(figsize=(10, 6))
            plt.imshow(dsm_vis, cmap="terrain", vmin=vmin, vmax=vmax)
            plt.title(f"DSM — {fecha}")
            plt.colorbar(label="Elevación (m)")
            plt.axis("off")
            plt.show()
 
        # Índices crudos
        for nombre, matriz in indices_crudos.items():
            if matriz is not None:
                paleta = PipelineOrtomosaico.PALETAS_INDICES.get(nombre, "viridis")
                calculadora.plot_index(matriz, title=f"Índice: {nombre.upper()} — {fecha}", cmap=paleta)
 
        # Máscara Otsu
        indice_array = indices_crudos[indice_otsu]
        indice_valido = indice_array[~np.isnan(indice_array)]
        vmin_v = np.percentile(indice_valido, 2)
        vmax_v = np.percentile(indice_valido, 98)
 
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        im0 = axs[0].imshow(indice_array, cmap="viridis", vmin=vmin_v, vmax=vmax_v)
        axs[0].set_title(f"{indice_otsu.upper()} original")
        axs[0].axis("off")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        axs[1].imshow(mascara_binaria, cmap="gray")
        axs[1].set_title(f"Máscara Otsu (umbral={umbral_otsu:.3f})")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
 
        # NDVI enmascarado
        if "ndvi" in indices_enmascarados:
            calculadora.plot_index(
                indices_enmascarados["ndvi"],
                title=f"NDVI Enmascarado — {fecha}",
                cmap="YlGn",
            )
 
        # CHM
        if chm is not None:
            plt.figure(figsize=(10, 8))
            chm_vmax = np.nanpercentile(chm, 98)
            im = plt.imshow(chm, cmap="YlGn", vmin=0, vmax=chm_vmax)
            plt.title(
                f"CHM | Media: {np.nanmean(chm):.3f}m | P95: {np.nanpercentile(chm, 95):.3f}m — {fecha}"
            )
            plt.axis("off")
            plt.colorbar(im, label="Altura canopeo (m)")
            plt.tight_layout()
            plt.show()
 
        # Rugosidades
        if rugosidades:
            fig, axs = plt.subplots(1, len(rugosidades), figsize=(6 * len(rugosidades), 5))
            if len(rugosidades) == 1:
                axs = [axs]
            for ax, (nombre, rug) in zip(axs, rugosidades.items()):
                vals = rug[~np.isnan(rug)]
                vmax_r = np.percentile(vals, 98) if len(vals) else 1
                im = ax.imshow(rug, cmap="magma", vmin=0, vmax=vmax_r)
                ax.set_title(f"Rugosidad {nombre}\nMedia: {np.mean(vals):.4f}m")
                ax.axis("off")
                plt.colorbar(im, ax=ax, label="Rugosidad (m)")
            plt.suptitle(f"Rugosidad del Canopeo — {fecha}", fontsize=13)
            plt.tight_layout()
            plt.show()