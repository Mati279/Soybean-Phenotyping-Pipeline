import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(BASE_DIR, 'data')
ORTOMOSAICOS_DIR = os.path.join(DATA_DIR, 'Ortomosaicos')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

BAND_SUFFIXES = {
    'RGB': '_rgb_orthophoto.tif',
    'MS': '_MS_orthophoto.tif',
    'RED': '_RED_orthophoto.tif',
    'NIR': '_NIR_orthophoto.tif'
}


SEGMENTATION_PARAMS = {
    'method': 'otsu',
    'primary_index': 'NDVI',
    'apply_negative_buffer': True,
    'buffer_pixels': 2  
}

VALIDATED_INDICES = [
    'NDVI',
    'NDRE',
    'GNDVI',
    'EVI',
    'VARI',
    'ExG',
    'SAVI'
]

FEATURE_EXTRACTION_STATS = [
    'min',
    'max',
    'mean',
    'std'
]