import ee
import os
import streamlit as st
from datetime import datetime, date
import matplotlib.pyplot as plt
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import geemap as geemap
import geopandas as gpd
from shapely.geometry import shape, Polygon
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import zipfile
import io
import calendar
from scipy import stats
import requests
import json
from dateutil.relativedelta import relativedelta
from typing import Optional, Tuple, Dict, Any, List
from branca.element import Template, MacroElement

# ==========================================================
# CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Analyse Bassin Versant Lac Itasy",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_footer = """
<style>
    footer { visibility: hidden; }
    button[title="View fullscreen"] { display: none; }
    .stActionButton { display: none; }
</style>
"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

# ==========================================================
# CONSTANTES
# ==========================================================
LAKE_ITASY_COORDS = [
    [46.72699751247, -19.047433273407638],
    [46.741760347593825, -19.049055887592797],
    [46.746910173732175, -19.03737271064461],
    [46.76682283509766, -19.04516158656546],
    [46.76304629592128, -19.060089243784287],
    [46.73832713026754, -19.069823948671107],
    [46.74656685231409, -19.09805135842389],
    [46.78948207028636, -19.11037906153418],
    [46.80184165310149, -19.094807073290696],
    [46.82965071453079, -19.076313434305625],
    [46.82690414064118, -19.06722808333309],
    [46.85162330637077, -19.0639831946587],
    [46.84922005441708, -19.040942657787507],
    [46.81317127093361, -19.045810643141134],
    [46.780899026728754, -19.024065868888933],
    [46.744506921393665, -19.01887266602623],
    [46.729744086332644, -19.033802687134983]
]

NDVI_THRESHOLD_DEFAULT = 0.3
LAKE_REFERENCE_AREA_KM2 = 35.0
MONTHS_FR = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
             'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

# ==========================================================
# INITIALISATION EARTH ENGINE
# ==========================================================
@st.cache_resource
def initialize_ee() -> bool:
    try:
        creds_json = st.secrets["EE_CREDENTIALS"]
        creds = service_account.Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials=creds, project='lac-itasy')
        return True
    except Exception as e:
        st.error(f"Erreur d'initialisation Earth Engine: {e}")
        return False

# ==========================================================
# FONCTIONS UTILITAIRES (MODIFIÉES POUR LES MOIS)
# ==========================================================
def generate_months_list(years: List[int], start_month: int, end_month: int, current_year: int, current_month: int) -> List[Tuple[int, int]]:
    """
    Génère la liste des (année, mois) à traiter.
    Pour la première année : mois >= start_month
    Pour la dernière année : mois <= end_month
    Pour les années intermédiaires : tous les mois (1 à 12 ou jusqu'au mois courant si année en cours)
    """
    months = []
    for i, year in enumerate(years):
        if year > current_year:
            continue
        max_month = 12 if year < current_year else current_month
        min_month = start_month if i == 0 else 1
        max_month = min(max_month, end_month if i == len(years)-1 else 12)
        for month in range(min_month, max_month + 1):
            months.append((year, month))
    return months

def create_temp_shapefile(uploaded_file) -> Optional[str]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file) as z:
                    z.extractall(tmpdir)
                shapefile_path = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.shp')), None)
            else:
                shapefile_path = os.path.join(tmpdir, uploaded_file.name)
                with open(shapefile_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            return shapefile_path if shapefile_path and os.path.exists(shapefile_path) else None
    except Exception as e:
        st.error(f"Erreur création shapefile temporaire: {e}")
        return None

def load_watershed_shapefile(shapefile_path: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[gpd.GeoDataFrame]]:
    try:
        if shapefile_path.endswith('.shp'):
            gdf = gpd.read_file(shapefile_path)
            if gdf.empty:
                st.error("Shapefile vide")
                return None, None, None, None
            watershed_fc = geemap.gdf_to_ee(gdf, geodesic=False)
            watershed_geom = watershed_fc.geometry()
            aoi = watershed_geom.bounds()
            return watershed_fc, watershed_geom, aoi, gdf
        else:
            st.error("Format de fichier non supporté")
            return None, None, None, None
    except Exception as e:
        st.error(f"Erreur chargement shapefile: {e}")
        return None, None, None, None

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    scl = image.select('SCL')
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(mask)

def calculate_ndvi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

def calculate_mndwi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(['B3', 'B11']).rename('MNDWI')

# ==========================================================
# FONCTIONS ANALYSE VÉGÉTATION (NDVI)
# ==========================================================
def compute_monthly_ndvi_area(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry, ndvi_threshold: float) -> Optional[float]:
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterDate(start, end)
                      .filterBounds(aoi)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                      .map(mask_s2_clouds)
                      .map(calculate_ndvi))
        if collection.size().getInfo() == 0:
            return None
        ndvi_median = collection.median()
        veg_mask = ndvi_median.gt(ndvi_threshold).selfMask()
        area_img = veg_mask.multiply(ee.Image.pixelArea())
        area = area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=watershed_geom, scale=10, maxPixels=1e13).get('NDVI')
        area_val = area.getInfo()
        return area_val / 1e6 if area_val else None
    except Exception as e:
        st.warning(f"Erreur calcul NDVI {year}-{month}: {e}")
        return None

def get_ndvi_image(year: int, month: int, watershed_geom: ee.Geometry) -> Optional[ee.Image]:
    """Récupère une image NDVI médiane pour un mois donné."""
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterDate(start, end)
                      .filterBounds(watershed_geom.bounds())
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                      .map(mask_s2_clouds)
                      .map(calculate_ndvi))
        if collection.size().getInfo() == 0:
            return None
        return collection.median().clip(watershed_geom)
    except Exception as e:
        st.error(f"Erreur récupération image NDVI: {e}")
        return None

# ==========================================================
# FONCTIONS ANALYSE SÉDIMENTAIRE
# ==========================================================
def monthly_rainfall(year: int, month: int, watershed_geom: ee.Geometry) -> Optional[float]:
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .filterBounds(watershed_geom.bounds())
                  .sum()
                  .rename('P'))
        rain = chirps.reduceRegion(reducer=ee.Reducer.mean(), geometry=watershed_geom, scale=5000, maxPixels=1e13).get('P')
        return rain.getInfo()
    except Exception as e:
        st.warning(f"Erreur calcul pluie {year}-{month}: {e}")
        return None

def monthly_sediment_index(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry) -> Optional[float]:
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        dem = ee.Image('USGS/SRTMGL1_003').clip(watershed_geom)
        slope = ee.Terrain.slope(dem)
        LS = slope.divide(30).clamp(0, 1).rename('LS')
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(start, end)
              .filterBounds(aoi)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
              .map(mask_s2_clouds)
              .map(calculate_ndvi))
        if s2.size().getInfo() == 0:
            return None
        ndvi = s2.median().rename('NDVI')
        rainfall = monthly_rainfall(year, month, watershed_geom)
        rainfall_val = rainfall if rainfall is not None else 0
        sediment_index = ee.Image.constant(rainfall_val).multiply(LS).multiply(ee.Image(1).subtract(ndvi))
        sed = sediment_index.reduceRegion(reducer=ee.Reducer.mean(), geometry=watershed_geom, scale=30, maxPixels=1e13).get('constant')
        return sed.getInfo()
    except Exception as e:
        st.warning(f"Erreur calcul sédiment {year}-{month}: {e}")
        return None

# ==========================================================
# FONCTIONS ANALYSE LAVAKAS
# ==========================================================
def calculate_ndti(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(['B11', 'B12']).rename('NDTI')

def calculate_brightness_index(image: ee.Image) -> ee.Image:
    brightness = image.select(['B2', 'B3', 'B4', 'B8']).reduce(ee.Reducer.mean())
    return brightness.rename('Brightness')

def detect_lavakas(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry, scale: int = 30) -> Tuple[Optional[ee.Image], Optional[ee.Image], float, int]:
    """Détecte les lavakas avec une meilleure gestion des erreurs."""
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterDate(start, end)
                         .filterBounds(aoi)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                         .map(mask_s2_clouds))
        
        # Vérification rapide de la présence d'images (évite size().getInfo())
        count_img = s2_collection.count()
        count = count_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=aoi,
            scale=1000,
            bestEffort=True
        ).get('constant')
        
        try:
            nb_images = count.getInfo()
            if nb_images is None or nb_images == 0:
                return None, None, 0.0, 0
        except:
            # Fallback : méthode classique mais avec timeout court
            if s2_collection.size().getInfo() == 0:
                return None, None, 0.0, 0
        
        # Image médiane
        s2_image = s2_collection.median().clip(watershed_geom)
        
        # Indices
        ndti = calculate_ndti(s2_image)
        brightness = calculate_brightness_index(s2_image)
        dem = ee.Image('USGS/SRTMGL1_003').clip(watershed_geom)
        slope = ee.Terrain.slope(dem)
        ndvi = calculate_ndvi(s2_image)
        
        # Normalisation
        ndti_norm = ndti.clamp(-1, 1)
        brightness_norm = brightness.divide(10000).clamp(0, 1)
        ndvi_norm = ndvi.clamp(-1, 1)
        
        # Score
        lavaka_score = (ndti_norm.multiply(0.4)
                        .add(brightness_norm.multiply(0.3))
                        .add(slope.divide(90).multiply(0.3))
                        .subtract(ndvi_norm.multiply(0.5))).rename('lavaka_score')
        
        lavaka_mask = lavaka_score.gt(0.4).selfMask()
        
        # Surface
        area_img = lavaka_mask.multiply(ee.Image.pixelArea())
        area_result = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=watershed_geom,
            scale=scale,
            maxPixels=1e13,
            bestEffort=True
        )
        area_m2 = area_result.get('lavaka_score')
        area_km2 = area_m2.getInfo() / 1e6 if area_m2 else 0.0
        
        # Estimation du nombre de lavakas (évite connectedPixelCount)
        # Surface moyenne estimée d'un lavaka : 500 m² (ajustable)
        estimated_num = int(area_km2 * 1e6 / 500) if area_km2 > 0 else 0
        num_lavakas = estimated_num
        
        # Optionnel : comptage précis si zone petite
        if area_km2 < 10 and area_km2 > 0:
            try:
                objects = lavaka_mask.connectedPixelCount(20, True)
                max_count = objects.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=watershed_geom,
                    scale=scale,
                    bestEffort=True,
                    maxPixels=1e13
                ).get('lavaka_score')
                if max_count:
                    num_lavakas = max_count.getInfo()
            except:
                pass
        
        return lavaka_mask.clip(watershed_geom), lavaka_score, area_km2, num_lavakas
        
    except Exception as e:
        st.warning(f"Erreur détection lavakas {year}-{month}: {str(e)[:100]}")
        return None, None, 0.0, 0

def analyze_lavakas_trend(lavaka_data: pd.DataFrame) -> Optional[Dict]:
    if lavaka_data.empty:
        return None
    df = lavaka_data.copy()
    if 'area_km2' in df.columns and len(df) > 1:
        x = np.arange(len(df))
        y = df['area_km2'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        annual_change = slope * 12
        return {'slope': slope, 'annual_change': annual_change, 'r_squared': r_value**2, 'p_value': p_value, 'total_change': y[-1] - y[0] if len(y) > 0 else 0}
    return None

# ==========================================================
# FONCTIONS ANALYSE ÉROSION
# ==========================================================
def plot_erosion_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for year in years[-3:]:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['total_erosion_km2'], mode='lines+markers', name=str(year), line=dict(width=2), marker=dict(size=8)))
    fig.update_layout(title=dict(text='Évolution mensuelle des zones d\'érosion totale', font=dict(size=16)), xaxis=dict(title='Mois', tickmode='array', tickvals=list(range(1,13)), ticktext=MONTHS_FR), yaxis=dict(title='Surface d\'érosion (km²)'), hovermode='x unified', template='plotly_white', height=500)
    return fig

def calculate_monthly_erosion(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry) -> Dict[str, float]:
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterDate(start, end)
                         .filterBounds(aoi)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                         .map(mask_s2_clouds))
        if s2_collection.size().getInfo() > 0:
            ndvi = s2_collection.map(calculate_ndvi).median().clip(watershed_geom).rename('NDVI')
        else:
            ndvi = ee.Image.constant(0.4).clip(watershed_geom).rename('NDVI')
        dem = ee.Image('USGS/SRTMGL1_003').clip(watershed_geom)
        slope = ee.Terrain.slope(dem)
        high_erosion = slope.gt(15).And(ndvi.lt(0.3))
        moderate_erosion = (slope.gte(5).And(slope.lte(15)).Or(ndvi.lt(0.5).And(ndvi.gte(0.3)))).And(high_erosion.Not())
        low_erosion = high_erosion.Not().And(moderate_erosion.Not())
        def get_area_km2(mask: ee.Image) -> float:
            try:
                area_image = mask.multiply(ee.Image.pixelArea())
                area_result = area_image.reduceRegion(reducer=ee.Reducer.sum(), geometry=watershed_geom, scale=30, maxPixels=1e13, bestEffort=True)
                area_dict = area_result.getInfo()
                if area_dict:
                    area_m2 = list(area_dict.values())[0]
                    return area_m2 / 1e6 if area_m2 else 0.0
                return 0.0
            except:
                return 0.0
        return {'year': year, 'month': month, 'low_erosion_km2': get_area_km2(low_erosion), 'moderate_erosion_km2': get_area_km2(moderate_erosion), 'high_erosion_km2': get_area_km2(high_erosion), 'total_erosion_km2': get_area_km2(high_erosion.Or(moderate_erosion))}
    except Exception as e:
        st.warning(f"Erreur calcul érosion {year}-{month}: {e}")
        return {'year': year, 'month': month, 'low_erosion_km2': None, 'moderate_erosion_km2': None, 'high_erosion_km2': None, 'total_erosion_km2': None}

def calculate_erosion_zones(watershed_geom: ee.Geometry) -> Tuple[ee.Image, Dict[str, float]]:
    try:
        dem = ee.Image('USGS/SRTMGL1_003').clip(watershed_geom)
        slope = ee.Terrain.slope(dem)
        end_date = ee.Date(datetime.now().strftime("%Y-%m-%d"))
        start_date = end_date.advance(-1, 'month')
        ndvi_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                           .filterDate(start_date, end_date)
                           .filterBounds(watershed_geom.bounds())
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)))
        if ndvi_collection.size().getInfo() > 0:
            ndvi = ndvi_collection.map(mask_s2_clouds).map(calculate_ndvi).median().clip(watershed_geom).rename('NDVI')
        else:
            ndvi = ee.Image.constant(0.4).clip(watershed_geom).rename('NDVI')
        high_erosion = slope.gt(15).And(ndvi.lt(0.3))
        moderate_erosion = (slope.gte(5).And(slope.lte(15)).Or(ndvi.lt(0.5).And(ndvi.gte(0.3)))).And(high_erosion.Not())
        low_erosion = high_erosion.Not().And(moderate_erosion.Not())
        erosion_image = (low_erosion.multiply(0).add(moderate_erosion.multiply(1)).add(high_erosion.multiply(2))).rename('erosion_class').clip(watershed_geom)
        def get_area_km2(mask: ee.Image) -> float:
            try:
                area_image = mask.multiply(ee.Image.pixelArea())
                area_result = area_image.reduceRegion(reducer=ee.Reducer.sum(), geometry=watershed_geom, scale=30, maxPixels=1e13, bestEffort=True)
                area_dict = area_result.getInfo()
                if area_dict:
                    area_m2 = list(area_dict.values())[0]
                    return area_m2 / 1e6 if area_m2 else 0.0
                return 0.0
            except:
                return 0.0
        area_low = get_area_km2(low_erosion)
        area_moderate = get_area_km2(moderate_erosion)
        area_high = get_area_km2(high_erosion)
        try:
            area_total_m2 = watershed_geom.area(maxError=1).getInfo()
            area_total = area_total_m2 / 1e6
        except:
            area_total = area_low + area_moderate + area_high or 1.0
        return erosion_image, {'low': area_low, 'moderate': area_moderate, 'high': area_high, 'total': area_total}
    except Exception as e:
        st.error(f"Erreur calcul zones érosion: {e}")
        return ee.Image.constant(0).clip(watershed_geom).rename('erosion_class'), {'low': 0.0, 'moderate': 0.0, 'high': 0.0, 'total': 0.0}

# ==========================================================
# FONCTIONS ANALYSE LAC ITASY
# ==========================================================
def calculate_lake_metrics(year: int, month: int, aoi: ee.Geometry) -> Dict[str, Any]:
    try:
        current_date = datetime.now()
        if year > current_date.year or (year == current_date.year and month > current_date.month):
            return {'status': 'future_date'}
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterDate(start, end)
                      .filterBounds(aoi)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                      .map(mask_s2_clouds))
        if collection.size().getInfo() == 0:
            return {'status': 'no_images'}
        mndwi_collection = collection.map(calculate_mndwi)
        mndwi_image = mndwi_collection.median().clip(aoi)
        water_mask = mndwi_image.gt(0.3)
        area_img = water_mask.multiply(ee.Image.pixelArea())
        area_result = area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13, bestEffort=True)
        area_m2 = area_result.get('MNDWI')
        area_m2_value = area_m2.getInfo() if area_m2 else None
        mean_result = mndwi_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=10, maxPixels=1e13, bestEffort=True)
        mean_mndwi = mean_result.get('MNDWI')
        mean_mndwi_value = mean_mndwi.getInfo() if mean_mndwi else None
        if area_m2_value is None:
            surface_km2 = 0.0
            status = 'no_water_detected'
        else:
            surface_km2 = area_m2_value / 1e6
            status = 'success'
        water_percentage = min(100, (surface_km2 / LAKE_REFERENCE_AREA_KM2) * 100) if LAKE_REFERENCE_AREA_KM2 > 0 else 0
        return {'surface_km2': surface_km2, 'mean_mndwi': mean_mndwi_value if mean_mndwi_value is not None else -1.0, 'water_percentage': water_percentage, 'status': status}
    except Exception as e:
        return {'status': f'error: {str(e)[:80]}'}

def analyze_lake_surface_trend(surface_data: pd.DataFrame) -> Tuple:
    df = surface_data.copy()
    df = df[df['surface_km2'] > 0]
    if len(df) < 2:
        return None, None, None, None, None, None, None
    x = np.arange(len(df))
    y = df['surface_km2'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    annual_change = slope * 12
    total_change = (df['surface_km2'].iloc[-1] - df['surface_km2'].iloc[0])
    percent_change = (total_change / df['surface_km2'].iloc[0]) * 100 if df['surface_km2'].iloc[0] > 0 else 0
    return slope, intercept, annual_change, percent_change, r_value, p_value, std_err

# ==========================================================
# FONCTIONS DE VISUALISATION CARTE (NOUVELLES)
# ==========================================================
def create_erosion_map(watershed_gdf: gpd.GeoDataFrame, erosion_zones: Optional[ee.Image] = None) -> folium.Map:
    """Carte interactive des zones d'érosion (originale)."""
    try:
        center = [watershed_gdf.geometry.centroid.y.mean(), watershed_gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.GeoJson(watershed_gdf.__geo_interface__, name="Bassin Versant", style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)
        if erosion_zones is not None:
            erosion_viz = {'min': 0, 'max': 2, 'palette': ['green', 'yellow', 'red']}
            map_id_dict = erosion_zones.getMapId(erosion_viz)
            folium.TileLayer(tiles=map_id_dict['tile_fetcher'].url_format, attr='Google Earth Engine', name="Zones d'érosion", overlay=True, control=True).add_to(m)
        folium.TileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', name='OpenStreetMap', attr='© OpenStreetMap contributors', control=True).add_to(m)
        folium.TileLayer('https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png', name='CartoDB Light', attr='© CartoDB', control=True).add_to(m)
        folium.LayerControl().add_to(m)
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; background:white; border:2px solid grey; padding:8px; z-index:9999; font-size:14px;">
            <b>Légende</b><br>
            <i style="background:green; width:15px; height:15px; display:inline-block;"></i> Faible érosion<br>
            <i style="background:yellow; width:15px; height:15px; display:inline-block;"></i> Érosion modérée<br>
            <i style="background:red; width:15px; height:15px; display:inline-block;"></i> Forte érosion
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        return m
    except Exception as e:
        st.error(f"Erreur création carte érosion: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)

def create_ndvi_map(ndvi_image: ee.Image, watershed_gdf: gpd.GeoDataFrame, year: int, month: int) -> folium.Map:
    """Carte interactive du NDVI."""
    try:
        center = [watershed_gdf.geometry.centroid.y.mean(), watershed_gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.GeoJson(watershed_gdf.__geo_interface__, name="Bassin Versant", style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)
        ndvi_viz = {'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green']}
        map_id = ndvi_image.getMapId(ndvi_viz)
        folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='Google Earth Engine', name=f'NDVI ({year}-{month:02d})', overlay=True, control=True).add_to(m)
        folium.LayerControl().add_to(m)
        legend_html = """
        <div style="position: fixed; bottom: 50px; right: 50px; background:white; border:2px solid grey; padding:8px; z-index:9999;">
            <b>NDVI</b><br>
            <i style="background:red; width:15px; height:15px; display:inline-block;"></i> Sol nu / eau<br>
            <i style="background:yellow; width:15px; height:15px; display:inline-block;"></i> Végétation clairsemée<br>
            <i style="background:green; width:15px; height:15px; display:inline-block;"></i> Végétation dense
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        return m
    except Exception as e:
        st.error(f"Erreur création carte NDVI: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)

def create_lake_map(mndwi_image: ee.Image, watershed_gdf: gpd.GeoDataFrame, year: int, month: int) -> folium.Map:
    """Carte interactive de l'extension du lac (MNDWI)."""
    try:
        center = [watershed_gdf.geometry.centroid.y.mean(), watershed_gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.GeoJson(watershed_gdf.__geo_interface__, name="Bassin Versant", style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)
        water_mask = mndwi_image.gt(0.3).selfMask()
        water_viz = {'palette': ['blue']}
        map_id = water_mask.getMapId(water_viz)
        folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='Google Earth Engine', name=f'Plan d\'eau ({year}-{month:02d})', overlay=True, control=True).add_to(m)
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Erreur création carte lac: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)

def create_lavaka_map(lavaka_score: ee.Image, watershed_gdf: gpd.GeoDataFrame, year: int, month: int) -> folium.Map:
    """Carte interactive des lavakas (probabilité)."""
    if lavaka_score is None:
            st.warning("Image lavaka non disponible")
            return folium.Map(location=[-19.0, 46.8], zoom_start=11)
    try:
        center = [watershed_gdf.geometry.centroid.y.mean(), watershed_gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.GeoJson(watershed_gdf.__geo_interface__, name="Bassin Versant", style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)
        lavaka_viz = {'min': 0, 'max': 0.8, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']}
        map_id = lavaka_score.getMapId(lavaka_viz)
        folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='Google Earth Engine', name=f'Probabilité lavaka ({year}-{month:02d})', overlay=True, control=True).add_to(m)
        folium.LayerControl().add_to(m)
        legend_html = """
        <div style="position: fixed; bottom: 50px; right: 50px; background:white; border:2px solid grey; padding:8px;">
            <b>Légende - Lavakas</b><br>
            <i style="background:#ffffb2; width:15px; height:15px; display:inline-block;"></i> Faible probabilité<br>
            <i style="background:#fd8d3c; width:15px; height:15px; display:inline-block;"></i> Probabilité moyenne<br>
            <i style="background:#bd0026; width:15px; height:15px; display:inline-block;"></i> Forte probabilité
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        return m
    except Exception as e:
        st.error(f"Erreur création carte lavaka: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)

def create_rainfall_map(year: int, month: int, watershed_geom: ee.Geometry, watershed_gdf: gpd.GeoDataFrame) -> folium.Map:
    """Carte des précipitations mensuelles CHIRPS."""
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .filterBounds(watershed_geom.bounds())
                  .sum()
                  .clip(watershed_geom)
                  .rename('precipitation'))
        center = [watershed_gdf.geometry.centroid.y.mean(), watershed_gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.GeoJson(watershed_gdf.__geo_interface__, name="Bassin Versant", style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)
        rain_viz = {'min': 0, 'max': 300, 'palette': ['white', 'lightblue', 'blue', 'darkblue']}
        map_id = chirps.getMapId(rain_viz)
        folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='Google Earth Engine', name=f'Précipitations {year}-{month:02d} (mm)', overlay=True, control=True).add_to(m)
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Erreur création carte pluie: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)

# ==========================================================
# FONCTIONS MÉTÉOROLOGIQUES (inchangées)
# ==========================================================
def get_openmeteo_historical(latitude: float, longitude: float, start_date: str, end_date: str) -> Optional[Dict]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": latitude, "longitude": longitude, "start_date": start_date, "end_date": end_date,
              "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum",
                        "rain_sum", "snowfall_sum", "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
                        "shortwave_radiation_sum", "et0_fao_evapotranspiration"], "timezone": "auto"}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur récupération données Open-Meteo: {e}")
        return None

def get_openmeteo_forecast(latitude: float, longitude: float) -> Optional[Dict]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude,
              "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "snowfall_sum",
                        "windspeed_10m_max", "windgusts_10m_max", "shortwave_radiation_sum", "et0_fao_evapotranspiration"],
              "timezone": "auto", "forecast_days": 7}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"⚠️ Erreur récupération prévisions Open-Meteo: {e}")
        return None

def process_meteo_data(meteo_data: Dict) -> Optional[pd.DataFrame]:
    if not meteo_data or "daily" not in meteo_data:
        return None
    daily_data = meteo_data["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily_data["time"]), "temp_max": daily_data.get("temperature_2m_max", []),
                       "temp_min": daily_data.get("temperature_2m_min", []), "temp_mean": daily_data.get("temperature_2m_mean", []),
                       "precipitation": daily_data.get("precipitation_sum", []), "rain": daily_data.get("rain_sum", []),
                       "snowfall": daily_data.get("snowfall_sum", []), "wind_speed_max": daily_data.get("windspeed_10m_max", []),
                       "wind_gust_max": daily_data.get("windgusts_10m_max", []), "wind_direction": daily_data.get("winddirection_10m_dominant", []),
                       "radiation": daily_data.get("shortwave_radiation_sum", []), "evapotranspiration": daily_data.get("et0_fao_evapotranspiration", [])})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df

def calculate_meteo_statistics(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or df.empty:
        return None
    stats = {}
    monthly_stats = df.groupby(["year", "month"]).agg({"precipitation": "sum", "temp_mean": "mean", "temp_max": "max",
                                                       "temp_min": "min", "radiation": "sum", "evapotranspiration": "sum"}).reset_index()
    monthly_stats["date"] = pd.to_datetime(monthly_stats["year"].astype(str) + "-" + monthly_stats["month"].astype(str) + "-01")
    annual_stats = df.groupby("year").agg({"precipitation": "sum", "temp_mean": "mean", "temp_max": "max",
                                           "temp_min": "min", "radiation": "sum", "evapotranspiration": "sum"}).reset_index()
    stats["monthly"] = monthly_stats
    stats["annual"] = annual_stats
    stats["daily"] = df
    return stats

def analyze_meteo_trends(meteo_stats: Dict) -> Optional[Dict]:
    if meteo_stats is None:
        return None
    trends = {}
    annual_df = meteo_stats["annual"]
    if len(annual_df) > 1:
        x = annual_df["year"]
        y_precip = annual_df["precipitation"]
        slope_precip, intercept_precip, _, _, _ = stats.linregress(x, y_precip)
        y_temp = annual_df["temp_mean"]
        slope_temp, intercept_temp, _, _, _ = stats.linregress(x, y_temp)
        trends["precipitation_slope"] = slope_precip
        trends["temperature_slope"] = slope_temp
        trends["precip_change_percent"] = (slope_precip * len(x)) / annual_df["precipitation"].mean() * 100
        trends["temp_change_total"] = slope_temp * len(x)
        mean_precip = annual_df["precipitation"].mean()
        annual_df["precip_anomaly"] = annual_df["precipitation"] - mean_precip
        mean_temp = annual_df["temp_mean"].mean()
        annual_df["temp_anomaly"] = annual_df["temp_mean"] - mean_temp
        trends["anomaly_data"] = annual_df[["year", "precip_anomaly", "temp_anomaly"]]
    return trends

# ==========================================================
# FONCTIONS DE VISUALISATION GRAPHIQUES (inchangées)
# ==========================================================
def plot_ndvi_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for year in years:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['surface_km2'], mode='lines+markers', name=str(year), line=dict(width=2), marker=dict(size=8)))
    fig.update_layout(title=dict(text='Évolution mensuelle de la végétation saine (NDVI > seuil)', font=dict(size=16)), xaxis=dict(title='Mois', tickmode='array', tickvals=list(range(1,13)), ticktext=MONTHS_FR), yaxis=dict(title='Surface (km²)'), hovermode='x unified', template='plotly_white', height=500)
    return fig

def plot_sediment_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for year in years:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['sediment_index'], mode='lines+markers', name=str(year), line=dict(width=2), marker=dict(size=8, symbol='square')))
    fig.update_layout(title=dict(text='Indice mensuel de contribution sédimentaire', font=dict(size=16)), xaxis=dict(title='Mois', tickmode='array', tickvals=list(range(1,13)), ticktext=MONTHS_FR), yaxis=dict(title='Indice sédimentaire'), hovermode='x unified', template='plotly_white', height=500)
    return fig

def plot_lake_surface_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Surface mensuelle du lac', 'Tendance et moyenne mobile'), vertical_spacing=0.15)
    for year in years:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['surface_km2'], mode='lines+markers', name=str(year), line=dict(width=2), marker=dict(size=6)), row=1, col=1)
    df_sorted = df.sort_values(['year', 'month'])
    df_sorted['date'] = pd.to_datetime(df_sorted['year'].astype(str) + '-' + df_sorted['month'].astype(str) + '-01')
    df_sorted['moving_avg'] = df_sorted['surface_km2'].rolling(window=6, min_periods=1).mean()
    if len(df_sorted) > 1:
        x = np.arange(len(df_sorted))
        y = df_sorted['surface_km2'].values
        slope, intercept = np.polyfit(x, y, 1)
        trend_line = intercept + slope * x
        fig.add_trace(go.Scatter(x=df_sorted['date'], y=df_sorted['surface_km2'], mode='markers', name='Données mensuelles', marker=dict(size=4, color='lightblue'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_sorted['date'], y=df_sorted['moving_avg'], mode='lines', name='Moyenne mobile (6 mois)', line=dict(color='orange', width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_sorted['date'], y=trend_line, mode='lines', name='Tendance linéaire', line=dict(color='red', width=2, dash='dash')), row=2, col=1)
    fig.update_xaxes(title_text='Mois', tickmode='array', tickvals=list(range(1,13)), ticktext=MONTHS_FR, row=1, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=1)
    fig.update_layout(height=700, hovermode='x unified', template='plotly_white', title_text="Évolution de la surface du Lac Itasy", title_font_size=18)
    return fig

def plot_lavakas_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Surface totale des lavakas', 'Nombre de lavakas', 'Évolution mensuelle', 'Tendance annuelle'), vertical_spacing=0.15, horizontal_spacing=0.15)
    for year in years[-3:]:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['area_km2'], mode='lines+markers', name=str(year), line=dict(width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['num_lavakas'], mode='lines+markers', name=str(year), line=dict(width=2), showlegend=False), row=1, col=2)
    monthly_avg = df.groupby('month').agg({'area_km2': 'mean', 'num_lavakas': 'mean'}).reset_index()
    fig.add_trace(go.Bar(x=monthly_avg['month'], y=monthly_avg['area_km2'], name='Surface moyenne', marker_color='sienna', opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=monthly_avg['month'], y=monthly_avg['num_lavakas'], mode='lines+markers', name='Nombre moyen', line=dict(color='darkred', width=3), yaxis='y2'), row=2, col=1)
    yearly_trend = df.groupby('year').agg({'area_km2': 'sum', 'num_lavakas': 'sum'}).reset_index()
    fig.add_trace(go.Scatter(x=yearly_trend['year'], y=yearly_trend['area_km2'], mode='lines+markers', name='Surface annuelle', line=dict(color='brown', width=3), marker=dict(size=10)), row=2, col=2)
    fig.update_xaxes(title_text='Mois', tickmode='array', tickvals=list(range(1,13)), ticktext=MONTHS_FR, row=1, col=1)
    fig.update_xaxes(title_text='Mois', tickmode='array', tickvals=list(range(1,13)), ticktext=MONTHS_FR, row=1, col=2)
    fig.update_xaxes(title_text='Mois', tickmode='array', tickvals=list(range(1,13)), ticktext=MONTHS_FR, row=2, col=1)
    fig.update_xaxes(title_text='Année', row=2, col=2)
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=1)
    fig.update_yaxes(title_text='Nombre', row=1, col=2)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=1)
    fig.update_yaxes(title_text='Nombre', row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=2)
    fig.update_layout(height=800, hovermode='x unified', template='plotly_white', title_text="Analyse temporelle des lavakas", title_font_size=18, showlegend=True)
    return fig

def create_lake_dashboard(lake_data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Surface mensuelle', 'Variation saisonnière', 'Tendance annuelle', 'Cycle annuel'), vertical_spacing=0.15)
    years = sorted(lake_data['year'].unique())
    for year in years[-3:]:
        sub = lake_data[lake_data['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['surface_km2'], mode='lines+markers', name=str(year)), row=1, col=1)
    seasonal_data = lake_data.groupby('month')['surface_km2'].agg(['mean', 'std']).reset_index()
    fig.add_trace(go.Bar(x=seasonal_data['month'], y=seasonal_data['mean'], error_y=dict(type='data', array=seasonal_data['std']), name='Moyenne ± écart-type', marker_color='skyblue'), row=1, col=2)
    yearly_data = lake_data.groupby('year')['surface_km2'].mean().reset_index()
    fig.add_trace(go.Scatter(x=yearly_data['year'], y=yearly_data['surface_km2'], mode='lines+markers', name='Surface moyenne annuelle', line=dict(color='green', width=3), marker=dict(size=10)), row=2, col=1)
    monthly_stats = lake_data.groupby('month')['surface_km2'].agg(['mean', 'std', 'min', 'max']).reset_index()
    fig.add_trace(go.Scatter(x=monthly_stats['month'], y=monthly_stats['mean'], mode='lines', name='Cycle moyen annuel', line=dict(color='darkblue', width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=list(monthly_stats['month']) + list(monthly_stats['month'][::-1]), y=list(monthly_stats['mean'] + monthly_stats['std']) + list((monthly_stats['mean'] - monthly_stats['std'])[::-1]), fill='toself', fillcolor='rgba(0,100,255,0.2)', line=dict(color='rgba(255,255,255,0)'), name='±1 écart-type', showlegend=True), row=2, col=2)
    fig.update_xaxes(title_text='Mois', row=1, col=1)
    fig.update_xaxes(title_text='Mois', row=1, col=2)
    fig.update_xaxes(title_text='Année', row=2, col=1)
    fig.update_xaxes(title_text='Mois', row=2, col=2)
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=2)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=2)
    fig.update_layout(height=800, showlegend=True, template='plotly_white', title_text="Tableau de bord - Surveillance du Lac Itasy", title_font_size=20)
    return fig

def create_dashboard(df_ndvi: pd.DataFrame, df_sediment: pd.DataFrame, years: List[int]) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Surface végétale saine', 'Contribution sédimentaire', 'Comparaison annuelle', 'Corrélation NDVI-Sédiments'), vertical_spacing=0.15, horizontal_spacing=0.15)
    for year in years:
        sub = df_ndvi[df_ndvi['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['surface_km2'], mode='lines+markers', name=f'NDVI {year}', legendgroup='ndvi', showlegend=True), row=1, col=1)
    for year in years:
        sub = df_sediment[df_sediment['year'] == year]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['month'], y=sub['sediment_index'], mode='lines+markers', name=f'Sédiments {year}', legendgroup='sed', showlegend=True), row=1, col=2)
    yearly_ndvi = df_ndvi.groupby('year')['surface_km2'].mean().reset_index()
    yearly_sed = df_sediment.groupby('year')['sediment_index'].mean().reset_index()
    fig.add_trace(go.Bar(x=yearly_ndvi['year'], y=yearly_ndvi['surface_km2'], name='NDVI moyen', marker_color='green', opacity=0.7), row=2, col=1)
    fig.add_trace(go.Bar(x=yearly_sed['year'], y=yearly_sed['sediment_index'], name='Sédiments moyen', marker_color='brown', opacity=0.7, yaxis='y2'), row=2, col=1)
    merged_df = pd.merge(df_ndvi, df_sediment, on=['year', 'month'])
    fig.add_trace(go.Scatter(x=merged_df['surface_km2'], y=merged_df['sediment_index'], mode='markers', name='Corrélation', marker=dict(size=8, color=merged_df['month'], colorscale='Viridis', showscale=True, colorbar=dict(title="Mois")), text=[f"Mois {m}" for m in merged_df['month']]), row=2, col=2)
    fig.update_xaxes(title_text="Mois", row=1, col=1)
    fig.update_xaxes(title_text="Mois", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=2, col=1)
    fig.update_xaxes(title_text="Surface végétale (km²)", row=2, col=2)
    fig.update_yaxes(title_text="Surface (km²)", row=1, col=1)
    fig.update_yaxes(title_text="Indice sédimentaire", row=1, col=2)
    fig.update_yaxes(title_text="Surface (km²)", row=2, col=1)
    fig.update_yaxes(title_text="Indice sédimentaire", row=2, col=2, secondary_y=True)
    fig.update_layout(height=800, showlegend=True, template='plotly_white', title_text="Tableau de bord d'analyse du bassin versant", title_font_size=20)
    return fig

def create_multi_year_meteo_comparison(meteo_stats: Dict, years: List[int]) -> go.Figure:
    if meteo_stats is None or "monthly" not in meteo_stats:
        return None
    monthly_df = meteo_stats["monthly"]
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Température moyenne par mois', 'Précipitations totales par mois', 'Température max par année', 'Précipitations totales par année', 'Radiation solaire par mois', 'Évapotranspiration par mois'), vertical_spacing=0.15, horizontal_spacing=0.15)
    for year in sorted(monthly_df['year'].unique()):
        year_data = monthly_df[monthly_df['year'] == year]
        fig.add_trace(go.Scatter(x=year_data['month'], y=year_data['temp_mean'], mode='lines+markers', name=f'T° {year}', line=dict(width=2), marker=dict(size=6)), row=1, col=1)
    for year in sorted(monthly_df['year'].unique())[-3:]:
        year_data = monthly_df[monthly_df['year'] == year]
        fig.add_trace(go.Bar(x=year_data['month'], y=year_data['precipitation'], name=f'Précip {year}', opacity=0.7), row=1, col=2)
    annual_stats = meteo_stats.get("annual")
    if annual_stats is not None and not annual_stats.empty:
        fig.add_trace(go.Bar(x=annual_stats['year'], y=annual_stats['temp_max'], name='Température max annuelle', marker_color='red', opacity=0.7), row=2, col=1)
        fig.add_trace(go.Bar(x=annual_stats['year'], y=annual_stats['precipitation'], name='Précipitations annuelles', marker_color='blue', opacity=0.7), row=2, col=2)
    monthly_avg = monthly_df.groupby('month').agg({'radiation': 'mean', 'evapotranspiration': 'mean'}).reset_index()
    fig.add_trace(go.Scatter(x=monthly_avg['month'], y=monthly_avg['radiation'], mode='lines+markers', name='Radiation solaire (moyenne)', line=dict(color='orange', width=3), marker=dict(size=8, symbol='square')), row=3, col=1)
    fig.add_trace(go.Scatter(x=monthly_avg['month'], y=monthly_avg['evapotranspiration'], mode='lines+markers', name='Évapotranspiration (moyenne)', line=dict(color='green', width=3), marker=dict(size=8, symbol='circle')), row=3, col=2)
    fig.update_xaxes(title_text="Mois", row=1, col=1)
    fig.update_xaxes(title_text="Mois", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=2, col=1)
    fig.update_xaxes(title_text="Année", row=2, col=2)
    fig.update_xaxes(title_text="Mois", row=3, col=1)
    fig.update_xaxes(title_text="Mois", row=3, col=2)
    fig.update_yaxes(title_text="Température (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Précipitations (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Température (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Précipitations (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Radiation (MJ/m²)", row=3, col=1)
    fig.update_yaxes(title_text="ET₀ (mm)", row=3, col=2)
    fig.update_layout(height=900, showlegend=True, template='plotly_white', title_text="Comparaison météorologique multi-années", title_font_size=18, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_meteo_dashboard(meteo_stats: Dict, trends: Dict, forecast_data: Optional[Dict] = None) -> go.Figure:
    if meteo_stats is None:
        return None
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Précipitations mensuelles', 'Température moyenne mensuelle', 'Évolution annuelle des précipitations', 'Évolution annuelle de la température', 'Anomalies de précipitations', 'Prévisions à 7 jours'), vertical_spacing=0.12, horizontal_spacing=0.15)
    monthly_df = meteo_stats["monthly"]
    annual_df = meteo_stats["annual"]
    fig.add_trace(go.Bar(x=monthly_df["date"], y=monthly_df["precipitation"], name="Précipitations mensuelles", marker_color="blue", opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=monthly_df["date"], y=monthly_df["temp_mean"], mode="lines+markers", name="Température moyenne", line=dict(color="red", width=2), marker=dict(size=4)), row=1, col=2)
    fig.add_trace(go.Bar(x=annual_df["year"], y=annual_df["precipitation"], name="Précipitations annuelles", marker_color="navy"), row=2, col=1)
    if trends:
        x_trend = annual_df["year"]
        y_trend = trends["precipitation_slope"] * (x_trend - x_trend.min()) + annual_df["precipitation"].mean()
        fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode="lines", name="Tendance précipitations", line=dict(color="orange", width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=annual_df["year"], y=annual_df["temp_mean"], mode="lines+markers", name="Température moyenne annuelle", line=dict(color="darkred", width=3), marker=dict(size=8, symbol="circle")), row=2, col=2)
    if trends:
        x_temp_trend = annual_df["year"]
        y_temp_trend = trends["temperature_slope"] * (x_temp_trend - x_temp_trend.min()) + annual_df["temp_mean"].mean()
        fig.add_trace(go.Scatter(x=x_temp_trend, y=y_temp_trend, mode="lines", name="Tendance température", line=dict(color="orange", width=2, dash="dash")), row=2, col=2)
    if trends and "anomaly_data" in trends:
        anomaly_df = trends["anomaly_data"]
        colors = ["red" if x < 0 else "blue" for x in anomaly_df["precip_anomaly"]]
        fig.add_trace(go.Bar(x=anomaly_df["year"], y=anomaly_df["precip_anomaly"], name="Anomalies précipitations", marker_color=colors, opacity=0.7), row=3, col=1)
    if forecast_data and "daily" in forecast_data:
        forecast_daily = forecast_data["daily"]
        forecast_df = pd.DataFrame({"date": pd.to_datetime(forecast_daily["time"]), "temp_max": forecast_daily.get("temperature_2m_max", []), "temp_min": forecast_daily.get("temperature_2m_min", []), "precipitation": forecast_daily.get("precipitation_sum", [])})
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["temp_max"], mode="lines+markers", name="Température max (prévision)", line=dict(color="red", width=2), marker=dict(size=6)), row=3, col=2)
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["temp_min"], mode="lines+markers", name="Température min (prévision)", line=dict(color="blue", width=2), marker=dict(size=6)), row=3, col=2)
        fig.add_trace(go.Bar(x=forecast_df["date"], y=forecast_df["precipitation"], name="Précipitations (prévision)", marker_color="lightblue", opacity=0.6, yaxis="y2"), row=3, col=2)
        fig.update_layout(yaxis5=dict(title="Température (°C)", title_font=dict(color="red")), yaxis6=dict(title="Précipitations (mm)", title_font=dict(color="blue"), overlaying="y5", side="right"))
    else:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="Prévisions non disponibles", showarrow=False, font=dict(size=16), row=3, col=2)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=2, col=1)
    fig.update_xaxes(title_text="Année", row=2, col=2)
    fig.update_xaxes(title_text="Année", row=3, col=1)
    fig.update_xaxes(title_text="Date (prévision)", row=3, col=2)
    fig.update_yaxes(title_text="Précipitations (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Température (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Précipitations (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Température (°C)", row=2, col=2)
    fig.update_yaxes(title_text="Anomalie (mm)", row=3, col=1)
    fig.update_layout(height=1000, showlegend=True, template="plotly_white", title_text="Tableau de bord météorologique - Bassin versant Lac Itasy", title_font_size=18, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ==========================================================
# INTERFACE PRINCIPALE
# ==========================================================
def main():
    st.title("🌿 Analyse du Bassin Versant du Lac Itasy")
    st.markdown("Analyse de la couverture végétale, érosion et sédimentation avec données satellitaires.")
    
    st.sidebar.header("⚙️ Paramètres d'analyse")
    if not initialize_ee():
        st.sidebar.warning("Authentification Earth Engine requise")
        return
    
    # 1. Données géographiques
    st.sidebar.subheader("1. Données géographiques")
    use_default = st.sidebar.checkbox("Utiliser shapefile par défaut", value=True)
    watershed_data = None
    if use_default:
        try:
            shapefile_path = 'data/BV_SBV_Lac_Itasy_20mars.shp'
            if os.path.exists(shapefile_path):
                watershed_data = load_watershed_shapefile(shapefile_path)
                if all(v is not None for v in watershed_data):
                    st.sidebar.success("✅ Shapefile par défaut chargé")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur shapefile par défaut: {str(e)[:50]}...")
    if not watershed_data:
        uploaded_file = st.sidebar.file_uploader("📤 Uploader shapefile", type=['shp', 'zip'])
        if uploaded_file:
            with st.sidebar.spinner("Chargement..."):
                temp_path = create_temp_shapefile(uploaded_file)
                if temp_path:
                    watershed_data = load_watershed_shapefile(temp_path)
                    if watershed_data and all(watershed_data):
                        st.sidebar.success("✅ Shapefile uploadé chargé")
    if not watershed_data or any(v is None for v in watershed_data):
        st.info("👈 Veuillez charger un shapefile valide dans la sidebar")
        show_welcome_page()
        return
    watershed_fc, watershed_geom, aoi, watershed_gdf = watershed_data
    
    # 2. Période d'analyse (années ET mois)
    st.sidebar.subheader("2. Période d'analyse")
    current_year = datetime.now().year
    years_options = list(range(2015, current_year + 1))
    
    year_selection_mode = st.sidebar.radio("Mode de sélection des années", ["Plage d'années", "Années spécifiques"], index=0)
    if year_selection_mode == "Plage d'années":
        col1, col2 = st.sidebar.columns(2)
        start_year = col1.selectbox("Année début", years_options, index=years_options.index(2020) if 2020 in years_options else 0)
        end_year = col2.selectbox("Année fin", years_options, index=years_options.index(current_year-1) if (current_year-1) in years_options else -1)
        if start_year > end_year:
            st.sidebar.error("❌ L'année début doit être ≤ année fin")
            return
        years_to_analyze = list(range(start_year, end_year+1))
    else:
        selected_years = st.sidebar.multiselect("Sélectionnez les années", years_options, default=[2020,2021,2022,2023] if current_year>=2023 else years_options[-4:])
        if not selected_years:
            st.sidebar.warning("⚠️ Veuillez sélectionner au moins une année")
            return
        years_to_analyze = sorted(selected_years)
    
    # Nouveaux sélecteurs de mois
    st.sidebar.subheader("Plage de mois")
    col1, col2 = st.sidebar.columns(2)
    start_month = col1.selectbox("Mois début", list(range(1,13)), format_func=lambda x: MONTHS_FR[x-1], index=0)
    end_month = col2.selectbox("Mois fin", list(range(1,13)), format_func=lambda x: MONTHS_FR[x-1], index=11)
    if start_month > end_month:
        st.sidebar.error("Le mois début doit être ≤ mois fin")
        return
    
    # Génération de la liste des mois à traiter
    months_to_process = generate_months_list(years_to_analyze, start_month, end_month, current_year, datetime.now().month)
    if not months_to_process:
        st.sidebar.error("Aucun mois valide dans la période sélectionnée")
        return
    st.sidebar.success(f"📅 {len(months_to_process)} mois à analyser")
    
    # 3. Options d'analyse
    st.sidebar.subheader("3. Options d'analyse")
    with st.sidebar.expander("🌱 Végétation et sédiments", expanded=True):
        analyze_ndvi = st.checkbox("Analyse végétation (NDVI)", value=True)
        if analyze_ndvi:
            ndvi_threshold = st.slider("Seuil NDVI", 0.0, 1.0, NDVI_THRESHOLD_DEFAULT, 0.05)
        else:
            ndvi_threshold = NDVI_THRESHOLD_DEFAULT
        analyze_sediment = st.checkbox("Analyse sédimentaire", value=True)
    with st.sidebar.expander("⚠️ Érosion", expanded=True):
        analyze_erosion_zones = st.checkbox("Analyser zones d'érosion", value=True)
        if analyze_erosion_zones:
            erosion_analysis_type = st.radio("Type d'analyse érosion", ["État actuel", "Évolution temporelle"], index=0)
    with st.sidebar.expander("🌊🌋 Hydrologie", expanded=False):
        analyze_lake = st.checkbox("Surveiller surface lac", value=True)
        analyze_lavakas = st.checkbox("Détecter lavakas", value=False)
    with st.sidebar.expander("🌤️ Météorologie", expanded=False):
        analyze_meteo = st.checkbox("Analyse météorologique", value=False)
        if analyze_meteo:
            meteo_range = st.radio("Période des données météo", ["Même que l'analyse", "Personnalisée", "Dernières 5 années"], index=0)
            if meteo_range == "Personnalisée":
                col1, col2 = st.columns(2)
                meteo_start_year = col1.selectbox("Début météo", years_options, index=years_options.index(years_to_analyze[0]) if years_to_analyze[0] in years_options else 0)
                meteo_end_year = col2.selectbox("Fin météo", years_options, index=years_options.index(years_to_analyze[-1]) if years_to_analyze[-1] in years_options else -1)
            elif meteo_range == "Dernières 5 années":
                meteo_start_year = current_year - 5
                meteo_end_year = current_year
            else:
                meteo_start_year = years_to_analyze[0]
                meteo_end_year = years_to_analyze[-1]
    with st.sidebar.expander("⚙️ Paramètres avancés", expanded=False):
        analysis_resolution = st.select_slider("Résolution d'analyse", options=["Rapide (30m)", "Standard (20m)", "Détaillée (10m)"], value="Standard (20m)")
        resolution_map = {"Rapide (30m)": 30, "Standard (20m)": 20, "Détaillée (10m)": 10}
        analysis_scale = resolution_map[analysis_resolution]
        if len(years_to_analyze) > 3:
            limit_months = st.checkbox("Limiter aux mois clés", value=False)
        else:
            limit_months = False
    
    # Récapitulatif et lancement
    st.sidebar.markdown("---")
    analyses_selected = sum([analyze_ndvi, analyze_sediment, analyze_erosion_zones, analyze_lake, analyze_lavakas, analyze_meteo])
    st.sidebar.subheader(f"📋 Récapitulatif ({analyses_selected} analyse(s))")
    st.sidebar.info(f"⏱️ Estimation: {analyses_selected * 2} min")
    
    if st.sidebar.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True):
        # Préparation des paramètres
        advanced_params = {'scale': analysis_scale, 'limit_months': limit_months, 'resolution': analysis_resolution}
        meteo_params = None
        if analyze_meteo:
            meteo_params = {'start_year': meteo_start_year, 'end_year': meteo_end_year, 'use_custom_range': (meteo_range == "Personnalisée"), 'is_last_5_years': (meteo_range == "Dernières 5 années")}
        erosion_params = None
        if analyze_erosion_zones:
            erosion_params = {'analysis_type': erosion_analysis_type, 'is_temporal': (erosion_analysis_type == "Évolution temporelle")}
        
        run_analysis(
            watershed_geom=watershed_geom, aoi=aoi, watershed_gdf=watershed_gdf,
            months_list=months_to_process, years=years_to_analyze,
            ndvi_threshold=ndvi_threshold,
            analyze_ndvi=analyze_ndvi, analyze_sediment=analyze_sediment,
            analyze_erosion=analyze_erosion_zones, analyze_lake=analyze_lake,
            analyze_lavakas=analyze_lavakas, analyze_meteo=analyze_meteo,
            meteo_params=meteo_params, erosion_params=erosion_params,
            advanced_params=advanced_params
        )
    else:
        show_welcome_page()

def run_analysis(watershed_geom, aoi, watershed_gdf, months_list, years, ndvi_threshold,
                 analyze_ndvi, analyze_sediment, analyze_erosion, analyze_lake,
                 analyze_lavakas, analyze_meteo, meteo_params, erosion_params, advanced_params):
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_analyses = sum([analyze_ndvi, analyze_sediment, analyze_erosion, analyze_lake, analyze_lavakas, analyze_meteo])
    if total_analyses == 0:
        st.warning("Veuillez sélectionner au moins une analyse")
        return
    current_analysis = 0
    
    # ==========================================================
    # ANALYSE NDVI
    # ==========================================================
    if analyze_ndvi:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Analyse végétation (NDVI)...")
        st.header("🌱 Analyse de la couverture végétale (NDVI)")
        with st.spinner("Calcul des surfaces végétales..."):
            ndvi_rows = []
            for i, (year, month) in enumerate(months_list):
                progress = (current_analysis - 1 + i/len(months_list)) / total_analyses
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Traitement NDVI {year}-{month:02d}...")
                area = compute_monthly_ndvi_area(year, month, watershed_geom, aoi, ndvi_threshold)
                ndvi_rows.append({'year': year, 'month': month, 'surface_km2': area if area is not None else None})
            df_ndvi = pd.DataFrame(ndvi_rows)
        # Affichage
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Données NDVI")
            st.dataframe(df_ndvi.round(3), use_container_width=True)
            csv = df_ndvi.to_csv(index=False)
            st.download_button("📥 Télécharger NDVI", csv, f"ndvi_{years[0]}_{years[-1]}.csv", "text/csv")
        with col2:
            st.subheader("Statistiques")
            if not df_ndvi.empty:
                stats_df = df_ndvi.groupby('year').agg({'surface_km2': ['mean','min','max','sum']}).round(2)
                stats_df.columns = ['Moyenne','Minimum','Maximum','Total']
                st.dataframe(stats_df, use_container_width=True)
        st.subheader("📈 Évolution temporelle")
        fig_ndvi = plot_ndvi_timeseries(df_ndvi, years)
        st.plotly_chart(fig_ndvi, use_container_width=True)
        
        # Carte NDVI pour le dernier mois disponible
        if not df_ndvi.empty:
            last_valid = df_ndvi.dropna(subset=['surface_km2']).iloc[-1]
            year_last, month_last = int(last_valid['year']), int(last_valid['month'])
            ndvi_img = get_ndvi_image(year_last, month_last, watershed_geom)
            if ndvi_img is not None:
                st.subheader(f"🗺️ Carte NDVI - {year_last}-{month_last:02d}")
                m_ndvi = create_ndvi_map(ndvi_img, watershed_gdf, year_last, month_last)
                st_folium(m_ndvi, width=800, height=500, key=f"ndvi_map_{year_last}_{month_last}")
    
    # ==========================================================
    # ANALYSE SÉDIMENTAIRE
    # ==========================================================
    if analyze_sediment:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Analyse sédimentaire...")
        st.header("🏔️ Analyse de l'érosion et sédimentation")
        with st.spinner("Calcul des indices sédimentaires..."):
            sediment_rows = []
            for i, (year, month) in enumerate(months_list):
                progress = (current_analysis - 1 + i/len(months_list)) / total_analyses
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Traitement sédiments {year}-{month:02d}...")
                sed_val = monthly_sediment_index(year, month, watershed_geom, aoi)
                sediment_rows.append({'year': year, 'month': month, 'sediment_index': sed_val if sed_val is not None else None})
            df_sediment = pd.DataFrame(sediment_rows)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Données sédimentaires")
            st.dataframe(df_sediment.round(4), use_container_width=True)
            csv_sed = df_sediment.to_csv(index=False)
            st.download_button("📥 Télécharger sédiments", csv_sed, f"sediment_{years[0]}_{years[-1]}.csv", "text/csv")
        with col2:
            st.subheader("Statistiques")
            if not df_sediment.empty:
                stats_sed = df_sediment.groupby('year').agg({'sediment_index': ['mean','min','max','std']}).round(4)
                stats_sed.columns = ['Moyenne','Minimum','Maximum','Écart-type']
                st.dataframe(stats_sed, use_container_width=True)
        st.subheader("📉 Évolution des indices sédimentaires")
        fig_sed = plot_sediment_timeseries(df_sediment, years)
        st.plotly_chart(fig_sed, use_container_width=True)
    
    # ==========================================================
    # ANALYSE ÉROSION
    # ==========================================================
    if analyze_erosion:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        is_temporal = erosion_params.get('is_temporal', False) if erosion_params else False
        if is_temporal:
            status_text.text("📈 Analyse évolution temporelle de l'érosion...")
            st.header("📈 Évolution Temporelle des Zones d'Érosion")
            with st.spinner("Calcul de l'évolution de l'érosion..."):
                erosion_rows = []
                for i, (year, month) in enumerate(months_list):
                    progress = (current_analysis - 1 + i/len(months_list)) / total_analyses
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Érosion {year}-{month:02d}...")
                    data = calculate_monthly_erosion(year, month, watershed_geom, aoi)
                    erosion_rows.append(data)
                df_erosion = pd.DataFrame(erosion_rows)
                df_valid = df_erosion.dropna(subset=['total_erosion_km2'])
            if not df_valid.empty:
                st.subheader("📊 Résumé statistique")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Érosion forte moyenne", f"{df_valid['high_erosion_km2'].mean():.1f} km²")
                col2.metric("Maximum érosion forte", f"{df_valid['high_erosion_km2'].max():.1f} km²")
                col3.metric("Période analysée", f"{len(df_valid['year'].unique())} an(s)")
                trend_text = "📈 Hausse" if df_valid['high_erosion_km2'].iloc[-1] > df_valid['high_erosion_km2'].iloc[0] else "📉 Baisse"
                col4.metric("Tendance globale", trend_text)
                st.subheader("📈 Visualisations")
                fig_temporal = go.Figure()
                fig_temporal.add_trace(go.Scatter(x=pd.to_datetime(df_valid['year'].astype(str)+'-'+df_valid['month'].astype(str)+'-01'), y=df_valid['high_erosion_km2'], mode='lines+markers', name='Érosion forte', line=dict(color='red')))
                fig_temporal.add_trace(go.Scatter(x=pd.to_datetime(df_valid['year'].astype(str)+'-'+df_valid['month'].astype(str)+'-01'), y=df_valid['moderate_erosion_km2'], mode='lines', name='Érosion modérée', line=dict(color='orange'), fill='tonexty'))
                fig_temporal.add_trace(go.Scatter(x=pd.to_datetime(df_valid['year'].astype(str)+'-'+df_valid['month'].astype(str)+'-01'), y=df_valid['low_erosion_km2'], mode='lines', name='Érosion faible', line=dict(color='green'), fill='tonexty'))
                fig_temporal.update_layout(title='Évolution des surfaces érodées', xaxis_title='Date', yaxis_title='Surface (km²)', hovermode='x unified', height=500)
                st.plotly_chart(fig_temporal, use_container_width=True)
                csv_detailed = df_valid.to_csv(index=False)
                st.download_button("📥 Données détaillées", csv_detailed, f"erosion_temporelle_{years[0]}_{years[-1]}.csv", "text/csv")
            else:
                st.warning("Aucune donnée valide")
        else:
            status_text.text("⚠️ Analyse zones d'érosion (état actuel)...")
            st.header("⚠️ Zones à Risque d'Érosion - État Actuel")
            with st.spinner("Calcul des zones d'érosion actuelles..."):
                erosion_zones, stats = calculate_erosion_zones(watershed_geom)
                if stats['total'] > 0:
                    percent_low = stats['low']/stats['total']*100
                    percent_moderate = stats['moderate']/stats['total']*100
                    percent_high = stats['high']/stats['total']*100
                else:
                    percent_low = percent_moderate = percent_high = 0
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Surface totale", f"{stats['total']:.1f} km²")
            col2.metric("Érosion faible", f"{stats['low']:.1f} km²", f"{percent_low:.1f}%")
            col3.metric("Érosion modérée", f"{stats['moderate']:.1f} km²", f"{percent_moderate:.1f}%")
            col4.metric("Érosion forte", f"{stats['high']:.1f} km²", f"{percent_high:.1f}%", delta_color="inverse")
            st.subheader("🗺️ Carte des zones d'érosion")
            m_erosion = create_erosion_map(watershed_gdf, erosion_zones)
            st_folium(m_erosion, width=800, height=500, key="erosion_map_final")
            # Recommandations
            if percent_high > 20:
                st.error("🔴 ALERTE : Fort risque d'érosion")
            elif percent_high > 10:
                st.warning("⚠️ Risque modéré à élevé")
            else:
                st.success("✅ Situation satisfaisante")
    
    # ==========================================================
    # ANALYSE LAC ITASY
    # ==========================================================
    if analyze_lake:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Analyse surface du lac...")
        st.header("🌊 Surveillance Lac Itasy")
        lake_polygon = Polygon(LAKE_ITASY_COORDS)
        lake_gdf = gpd.GeoDataFrame(geometry=[lake_polygon], crs="EPSG:4326")
        lake_geometry = ee.Geometry.Polygon(LAKE_ITASY_COORDS)
        aoi_lake = lake_geometry.bounds()
        lake_rows = []
        last_lake_image = None
        last_lake_date = None
        for i, (year, month) in enumerate(months_list):
            progress = (current_analysis - 1 + i/len(months_list)) / total_analyses
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Traitement lac {year}-{month:02d}...")
            metrics = calculate_lake_metrics(year, month, aoi_lake)
            if metrics['status'] == 'success':
                lake_rows.append({'year': year, 'month': month, 'surface_km2': round(metrics['surface_km2'],3), 'mean_mndwi': round(metrics['mean_mndwi'],3), 'water_percentage': round(metrics['water_percentage'],1)})
                # Récupérer l'image MNDWI pour la carte
                start = ee.Date.fromYMD(year, month, 1)
                end = start.advance(1, 'month')
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                              .filterDate(start, end)
                              .filterBounds(aoi_lake)
                              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                              .map(mask_s2_clouds))
                if collection.size().getInfo() > 0:
                    last_lake_image = collection.map(calculate_mndwi).median().clip(lake_geometry)
                    last_lake_date = (year, month)
        if lake_rows:
            df_lake = pd.DataFrame(lake_rows)
            fig_lake = create_lake_dashboard(df_lake)
            st.plotly_chart(fig_lake, use_container_width=True)
            st.subheader("📉 Analyse de tendance")
            slope, intercept, annual_change, percent_change, r_value, p_value, std_err = analyze_lake_surface_trend(df_lake)
            if slope is not None:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tendance mensuelle", f"{slope:.4f} km²/mois", delta_color="inverse" if slope<0 else "normal")
                col2.metric("Changement annuel", f"{annual_change:.2f} km²/an", delta_color="inverse" if annual_change<0 else "normal")
                col3.metric("Changement total", f"{percent_change:.1f}%", delta_color="inverse" if percent_change<0 else "normal")
                col4.metric("Corrélation (R²)", f"{r_value**2:.3f}")
            csv_data = df_lake.to_csv(index=False)
            st.download_button("📥 Télécharger données lac", csv_data, f"lake_itasy_{years[0]}_{years[-1]}.csv", "text/csv")
            # Carte du lac pour le dernier mois
            if last_lake_image is not None:
                st.subheader(f"🗺️ Carte du lac - {last_lake_date[0]}-{last_lake_date[1]:02d}")
                m_lake = create_lake_map(last_lake_image, watershed_gdf, last_lake_date[0], last_lake_date[1])
                st_folium(m_lake, width=800, height=500, key="lake_map")
        else:
            st.warning("Aucune donnée lac disponible")
    
    # ==========================================================
    # ANALYSE LAVAKAS
    # ==========================================================
    if analyze_lavakas:
    current_analysis += 1
    progress_bar.progress(current_analysis / total_analyses)
    status_text.text("Détection lavakas...")
    st.header("🕳️ Détection lavakas")
    
    lavaka_rows = []
    last_lavaka_score = None
    last_lavaka_date = None
    
    # Limiter le nombre de mois pour éviter les timeouts (max 24 mois)
    max_months = min(len(months_list), 24)
    months_subset = months_list[:max_months]
    
    if not months_subset:
        st.warning("Aucun mois à analyser pour les lavakas")
    else:
        for i, (year, month) in enumerate(months_subset):
            progress = (current_analysis - 1 + i/len(months_subset)) / total_analyses
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Détection lavakas {year}-{month:02d}...")
            
            lavaka_mask, lavaka_score, area_km2, num_lavakas = detect_lavakas(
                year, month, watershed_geom, aoi, scale=advanced_params.get('scale', 30)
            )
            
            lavaka_rows.append({
                'year': year, 'month': month,
                'area_km2': area_km2, 'num_lavakas': num_lavakas
            })
            
            if lavaka_score is not None:
                last_lavaka_score = lavaka_score
                last_lavaka_date = (year, month)
        
        df_lavakas = pd.DataFrame(lavaka_rows)
        
        if not df_lavakas.empty:
            st.subheader("Résultats des lavakas")
            st.dataframe(df_lavakas.round(3), use_container_width=True)
            
            # Graphique (dernières années présentes)
            years_present = sorted(df_lavakas['year'].unique())
            if len(years_present) > 0:
                fig_lavakas = plot_lavakas_timeseries(df_lavakas, years_present[-3:])
                st.plotly_chart(fig_lavakas, use_container_width=True)
            
            csv_lavakas = df_lavakas.to_csv(index=False)
            st.download_button("📥 Télécharger données lavakas", csv_lavakas,
                               f"lavakas_{years[0]}_{years[-1]}.csv", "text/csv")
            
            if last_lavaka_score is not None and last_lavaka_date is not None:
                st.subheader(f"🗺️ Carte des lavakas - {last_lavaka_date[0]}-{last_lavaka_date[1]:02d}")
                m_lavaka = create_lavaka_map(last_lavaka_score, watershed_gdf,
                                             last_lavaka_date[0], last_lavaka_date[1])
                st_folium(m_lavaka, width=800, height=500, key=f"lavaka_map_{last_lavaka_date[0]}_{last_lavaka_date[1]}")
            else:
                st.info("Aucune détection de lavakas valide. Carte non générée.")
        else:
            st.warning("Aucune donnée de lavakas n'a pu être calculée.")
    
    # ==========================================================
    # ANALYSE MÉTÉOROLOGIQUE (avec carte des précipitations)
    # ==========================================================
    if analyze_meteo:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Récupération des données météo...")
        st.header("🌤️ Analyse Météorologique")
        centroid_gdf = watershed_gdf.to_crs(epsg=4326)
        centroid_lat = centroid_gdf.geometry.centroid.y.mean()
        centroid_lon = centroid_gdf.geometry.centroid.x.mean()
        st.info(f"Coordonnées du bassin : {centroid_lat:.6f}°, {centroid_lon:.6f}°")
        if meteo_params and meteo_params.get('use_custom_range', False):
            meteo_start_year = meteo_params['start_year']
            meteo_end_year = meteo_params['end_year']
        elif meteo_params and meteo_params.get('is_last_5_years', False):
            meteo_start_year = datetime.now().year - 5
            meteo_end_year = datetime.now().year
        else:
            meteo_start_year = years[0]
            meteo_end_year = years[-1]
        start_date = date(meteo_start_year, 1, 1)
        end_date = date(meteo_end_year, 12, 31) if meteo_end_year != datetime.now().year else date.today()
        with st.spinner("Téléchargement des données météo..."):
            meteo_data = get_openmeteo_historical(centroid_lat, centroid_lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            forecast_data = get_openmeteo_forecast(centroid_lat, centroid_lon)
        if meteo_data:
            df_meteo = process_meteo_data(meteo_data)
            if df_meteo is not None:
                df_meteo_filtered = df_meteo[df_meteo['year'].between(meteo_start_year, meteo_end_year)]
                if not df_meteo_filtered.empty:
                    meteo_stats = calculate_meteo_statistics(df_meteo_filtered)
                    trends = analyze_meteo_trends(meteo_stats)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Température moyenne", f"{df_meteo_filtered['temp_mean'].mean():.1f}°C")
                    col2.metric("Précipitations totales", f"{df_meteo_filtered['precipitation'].sum():.0f} mm")
                    col3.metric("Période analysée", f"{len(df_meteo_filtered['year'].unique())} an(s)")
                    meteo_dashboard = create_meteo_dashboard(meteo_stats, trends, forecast_data)
                    if meteo_dashboard:
                        st.plotly_chart(meteo_dashboard, use_container_width=True)
                    csv_data = df_meteo_filtered.to_csv(index=False)
                    st.download_button("📥 Télécharger données météo", csv_data, f"meteo_{meteo_start_year}_{meteo_end_year}.csv", "text/csv")
                    # Carte des précipitations pour le dernier mois de la première année (exemple)
                    if months_list:
                        last_year, last_month = months_list[-1]
                        st.subheader(f"🗺️ Carte des précipitations - {last_year}-{last_month:02d}")
                        m_rain = create_rainfall_map(last_year, last_month, watershed_geom, watershed_gdf)
                        st_folium(m_rain, width=800, height=500, key="rain_map")
                else:
                    st.error("Aucune donnée pour la période")
            else:
                st.error("Erreur traitement données")
        else:
            st.error("Impossible de récupérer les données météo")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Analyse terminée")
    st.balloons()
    st.success(f"✅ Analyse terminée pour la période {years[0]}-{years[-1]}")

def show_welcome_page():
    st.info("👈 Configurez les paramètres et cliquez sur 'Lancer l'analyse'")
    st.subheader("Fonctionnalités")
    features = [
        "🌱 Analyse NDVI", "🏔️ Analyse sédimentaire", "⚠️ Zones à risque",
        "🌊 Surveillance lac", "🕳️ Détection lavakas", "🌤️ Analyse météo",
        "🗺️ Cartographie interactive", "📊 Dashboard"
    ]
    for f in features:
        st.markdown(f"- **{f}**")

if __name__ == "__main__":
    main()
