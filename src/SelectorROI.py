import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from shapely.geometry import Polygon

class SelectorROI:
    """
    Clase para manejar la selección interactiva de Regiones de Interés en ortomosaicos.
  
    """
    def __init__(self, ruta_raster: str, factor_reduccion: int = 4):
        self.coordenadas = []
        
        with rasterio.open(ruta_raster) as src:
            out_shape = (src.count, int(src.height / factor_reduccion), int(src.width / factor_reduccion))
            
            self.imagen_baja_res = src.read(out_shape=out_shape)
            
            self.transformacion = src.transform * src.transform.scale(
                (src.width / self.imagen_baja_res.shape[-1]),
                (src.height / self.imagen_baja_res.shape[-2])
            )
            
    def _al_seleccionar(self, vertices):
        self.coordenadas.clear()
        self.coordenadas.extend(vertices)
        print("ROI definida.")

    def mostrar_interfaz(self):
        # Grafica.
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Mostramos el raster ajustado espacialmente.
        show(self.imagen_baja_res, transform=self.transformacion, ax=self.ax, 
             title="Haz clic en los bordes para delimitar el lote.\nPresiona 'Enter' al terminar.")
        
        # Instancia el widget.
        self.selector = PolygonSelector(self.ax, self._al_seleccionar)
        plt.show()

    def obtener_poligono_shapely(self, coordenadas_default: list = None) -> Polygon:
        if len(self.coordenadas) < 3 and coordenadas_default:
            print("Usando coordenadas predeterminadas (modo desarrollo).")
            return Polygon(coordenadas_default)
        elif len(self.coordenadas) < 3:
            raise ValueError("No se ha definido un polígono válido y no hay coordenadas por defecto.")
        
        return Polygon(self.coordenadas)