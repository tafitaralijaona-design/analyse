import ee
import os
import streamlit as st
from datetime import datetime, date
import matplotlib.pyplot as plt
from google.oauth2 import service_account   # <-- ajouter cette ligne
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
# Configuration de la page
st.set_page_config(
    page_title="Analyse Bassin Versant Lac Itasy",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- Code CSS pour masquer le badge "Built with Streamlit" et le bouton "Fullscreen" ---
hide_streamlit_footer = """
<style>
    /* Masque le pied de page complet */
    footer {
        visibility: hidden;
    }

    /* Cible et masque spécifiquement le bouton 'Fullscreen' */
    button[title="View fullscreen"] {
        display: none;
    }

    /* Une classe alternative pour le bouton Fullscreen, si le sélecteur ci-dessus ne fonctionne pas */
    .stActionButton {
        display: none;
    }
</style>
"""

# Injecte le code HTML/CSS dans l'application
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
    """Initialise Earth Engine avec le compte de service depuis les secrets."""
    try:
        # Récupérer les credentials depuis les secrets
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
# FONCTIONS UTILITAIRES
# ==========================================================
# AJOUTER CETTE FONCTION UTILITAIRE DANS LA SECTION "FONCTIONS UTILITAIRES"
def get_months_to_process(years_list, current_year, current_month):
    """
    Génère la liste des (année, mois) à traiter en fonction des années sélectionnées.
    """
    months_to_process = []
    
    for year in years_list:
        # Déterminer le dernier mois à traiter pour cette année
        if year < datetime.now().year:
            last_month = 12  # Année complète passée
        elif year == datetime.now().year:
            last_month = datetime.now().month  # Année en cours
        else:
            continue  # Année future, ignorer
        
        # Ajouter tous les mois de cette année
        for month in range(1, last_month + 1):
            months_to_process.append((year, month))
    
    return months_to_process


def create_temp_shapefile(uploaded_file) -> Optional[str]:
    """Crée un shapefile temporaire à partir d'un fichier uploadé."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file) as z:
                    z.extractall(tmpdir)
                shapefile_path = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) 
                                      if f.endswith('.shp')), None)
            else:
                shapefile_path = os.path.join(tmpdir, uploaded_file.name)
                with open(shapefile_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            
            return shapefile_path if shapefile_path and os.path.exists(shapefile_path) else None
    except Exception as e:
        st.error(f"Erreur création shapefile temporaire: {e}")
        return None
def load_watershed_shapefile(shapefile_path: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[gpd.GeoDataFrame]]:
    """Charge le bassin versant depuis un shapefile."""
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
    """Masque les nuages dans les images Sentinel-2."""
    scl = image.select('SCL')
    mask = (
        scl.neq(3)  # Nuages épais
        .And(scl.neq(8))  # Nuages cirrus
        .And(scl.neq(9))  # Ombres
        .And(scl.neq(10)) # Neige
        .And(scl.neq(11)) # Nuages
    )
    return image.updateMask(mask)

def calculate_ndvi(image: ee.Image) -> ee.Image:
    """Calcule l'NDVI."""
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

def calculate_mndwi(image: ee.Image) -> ee.Image:
    """Calcule le MNDWI pour détection d'eau."""
    return image.normalizedDifference(['B3', 'B11']).rename('MNDWI')

# ==========================================================
# FONCTIONS ANALYSE VÉGÉTATION (NDVI)
# ==========================================================
def compute_monthly_ndvi_area(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry, ndvi_threshold: float) -> Optional[float]:
    """Calcule la surface de végétation saine mensuelle."""
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(mask_s2_clouds)
            .map(calculate_ndvi)
        )
        
        size = collection.size().getInfo()
        if size == 0:
            return None
        
        ndvi_median = collection.median()
        veg_mask = ndvi_median.gt(ndvi_threshold).selfMask()
        area_img = veg_mask.multiply(ee.Image.pixelArea())
        area = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=watershed_geom,
            scale=10,
            maxPixels=1e13
        ).get('NDVI')
        
        area_val = area.getInfo()
        return area_val / 1e6 if area_val else None  # Convertir en km²
        
    except Exception as e:
        st.warning(f"Erreur calcul NDVI {year}-{month}: {e}")
        return None

def get_ndvi_image(date_str: str, watershed_geom: ee.Geometry) -> Optional[ee.Image]:
    """Récupère une image NDVI pour une date spécifique."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start = ee.Date.fromYMD(date_obj.year, date_obj.month, 1)
        end = start.advance(1, 'month')
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(watershed_geom.bounds())
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(mask_s2_clouds)
            .map(calculate_ndvi)
        )
        
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
    """Calcule la pluviométrie mensuelle CHIRPS."""
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        
        chirps = (
            ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            .filterDate(start, end)
            .filterBounds(watershed_geom.bounds())
            .sum()
            .rename('P')
        )
        
        rain = chirps.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=watershed_geom,
            scale=5000,
            maxPixels=1e13
        ).get('P')
        
        rain_val = rain.getInfo()
        return rain_val
        
    except Exception as e:
        st.warning(f"Erreur calcul pluie {year}-{month}: {e}")
        return None

def monthly_sediment_index(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry) -> Optional[float]:
    """Calcule l'indice sédimentaire mensuel."""
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        
        # Récupérer le DEM et calculer la pente
        dem = ee.Image('USGS/SRTMGL1_003').clip(watershed_geom)
        slope = ee.Terrain.slope(dem)
        LS = slope.divide(30).clamp(0, 1).rename('LS')
        
        s2 = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(mask_s2_clouds)
            .map(calculate_ndvi)
        )
        
        size = s2.size().getInfo()
        if size == 0:
            return None
        
        ndvi = s2.median().rename('NDVI')
        rainfall = monthly_rainfall(year, month, watershed_geom)
        
        if rainfall is None:
            rainfall_val = 0
        else:
            rainfall_val = rainfall
        
        sediment_index = ee.Image.constant(rainfall_val).multiply(LS).multiply(
            ee.Image(1).subtract(ndvi)
        )
        
        sed = sediment_index.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=watershed_geom,
            scale=30,
            maxPixels=1e13
        ).get('constant')
        
        sed_val = sed.getInfo()
        return sed_val
        
    except Exception as e:
        st.warning(f"Erreur calcul sédiment {year}-{month}: {e}")
        return None

# ==========================================================
# FONCTIONS MÉTÉOROLOGIQUES
# ==========================================================
def get_openmeteo_historical(latitude: float, longitude: float, start_date: str, end_date: str) -> Optional[Dict]:
    """Récupère les données historiques météorologiques d'Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "rain_sum", "snowfall_sum",
            "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration"
        ],
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur récupération données Open-Meteo: {e}")
        return None

def process_meteo_data(meteo_data: Dict) -> Optional[pd.DataFrame]:
    """Traite les données météorologiques pour créer un DataFrame."""
    if not meteo_data or "daily" not in meteo_data:
        return None
    
    daily_data = meteo_data["daily"]
    
    # Créer DataFrame directement
    df = pd.DataFrame({
        "date": pd.to_datetime(daily_data["time"]),
        "temp_max": daily_data.get("temperature_2m_max", []),
        "temp_min": daily_data.get("temperature_2m_min", []),
        "temp_mean": daily_data.get("temperature_2m_mean", []),
        "precipitation": daily_data.get("precipitation_sum", []),
        "rain": daily_data.get("rain_sum", []),
        "snowfall": daily_data.get("snowfall_sum", []),
        "wind_speed_max": daily_data.get("windspeed_10m_max", []),
        "wind_gust_max": daily_data.get("windgusts_10m_max", []),
        "wind_direction": daily_data.get("winddirection_10m_dominant", []),
        "radiation": daily_data.get("shortwave_radiation_sum", []),
        "evapotranspiration": daily_data.get("et0_fao_evapotranspiration", [])
    })
    
    # Ajouter colonnes temporelles
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    
    return df

def calculate_meteo_statistics(df: pd.DataFrame) -> Optional[Dict]:
    """Calcule les statistiques météorologiques mensuelles et annuelles."""
    if df is None or df.empty:
        return None
    
    stats = {}
    
    # Statistiques mensuelles
    monthly_stats = df.groupby(["year", "month"]).agg({
        "precipitation": "sum",
        "temp_mean": "mean",
        "temp_max": "max",
        "temp_min": "min",
        "radiation": "sum",
        "evapotranspiration": "sum"
    }).reset_index()
    
    monthly_stats["date"] = pd.to_datetime(
        monthly_stats["year"].astype(str) + "-" + 
        monthly_stats["month"].astype(str) + "-01"
    )
    
    # Statistiques annuelles
    annual_stats = df.groupby("year").agg({
        "precipitation": "sum",
        "temp_mean": "mean",
        "temp_max": "max",
        "temp_min": "min",
        "radiation": "sum",
        "evapotranspiration": "sum"
    }).reset_index()
    
    stats["monthly"] = monthly_stats
    stats["annual"] = annual_stats
    stats["daily"] = df
    
    return stats

def analyze_meteo_trends(meteo_stats: Dict) -> Optional[Dict]:
    """Analyse les tendances météorologiques à long terme."""
    if meteo_stats is None:
        return None
    
    trends = {}
    annual_df = meteo_stats["annual"]
    
    if len(annual_df) > 1:
        # Tendance des précipitations
        x = annual_df["year"]
        y_precip = annual_df["precipitation"]
        slope_precip, intercept_precip, _, _, _ = stats.linregress(x, y_precip)
        
        # Tendance de la température
        y_temp = annual_df["temp_mean"]
        slope_temp, intercept_temp, _, _, _ = stats.linregress(x, y_temp)
        
        trends["precipitation_slope"] = slope_precip  # mm/an
        trends["temperature_slope"] = slope_temp  # °C/an
        trends["precip_change_percent"] = (slope_precip * len(x)) / annual_df["precipitation"].mean() * 100
        trends["temp_change_total"] = slope_temp * len(x)
        
        # Calcul des anomalies
        mean_precip = annual_df["precipitation"].mean()
        annual_df["precip_anomaly"] = annual_df["precipitation"] - mean_precip
        
        mean_temp = annual_df["temp_mean"].mean()
        annual_df["temp_anomaly"] = annual_df["temp_mean"] - mean_temp
        
        trends["anomaly_data"] = annual_df[["year", "precip_anomaly", "temp_anomaly"]]
    
    return trends

# ==========================================================
# FONCTIONS ANALYSE LAVAKAS
# ==========================================================
def calculate_ndti(image: ee.Image) -> ee.Image:
    """Calcule le NDTI (Normalized Difference Tillage Index)."""
    return image.normalizedDifference(['B11', 'B12']).rename('NDTI')

def calculate_brightness_index(image: ee.Image) -> ee.Image:
    """Calcule l'indice de brillance pour détecter les sols exposés."""
    brightness = image.select(['B2', 'B3', 'B4', 'B8']).reduce(ee.Reducer.mean())
    return brightness.rename('Brightness')

def calculate_bsi(image):
    """Bare Soil Index"""
    return image.expression(
        '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
        {
            'SWIR': image.select('B11'),
            'RED': image.select('B4'),
            'NIR': image.select('B8'),
            'BLUE': image.select('B2')
        }
    ).rename('BSI')


def detect_lavakas(
    year: int,
    month: int,
    watershed_geom: ee.Geometry,
    aoi: ee.Geometry
):

    try:

        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')

        # ==========================
        # Sentinel-2
        # ==========================
        s2_collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(watershed_geom)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(mask_s2_clouds)
        )

        if s2_collection.size().getInfo() == 0:
            return None, None, 0.0, 0

        s2 = s2_collection.median().clip(watershed_geom)

        # ==========================
        # Eau (MNDWI)
        # ==========================
        mndwi = calculate_mndwi(s2)

        water_mask = mndwi.gt(0.25)

        land_mask = water_mask.Not()

        s2 = s2.updateMask(land_mask)

        # ==========================
        # DEM
        # ==========================
        dem = ee.Image("USGS/SRTMGL1_003").clip(watershed_geom)

        slope = ee.Terrain.slope(dem)

        # ==========================
        # Courbure
        # ==========================
        curvature = (
            dem.convolve(ee.Kernel.laplacian8())
            .rename("curvature")
        )

        # ==========================
        # Indices spectraux
        # ==========================
        ndvi = calculate_ndvi(s2)

        ndti = calculate_ndti(s2)

        bsi = calculate_bsi(s2)

        # ==========================
        # Texture GLCM
        # ==========================
        red = s2.select('B4').toUint16()

        texture = red.glcmTexture(size=3)

        contrast = texture.select('B4_contrast')

        # ==========================
        # Normalisation
        # ==========================
        slope_norm = slope.divide(90).clamp(0, 1)

        bsi_norm = bsi.clamp(-1, 1)

        ndti_norm = ndti.clamp(-1, 1)

        ndvi_norm = ndvi.clamp(-1, 1)

        contrast_norm = contrast.divide(100).clamp(0, 1)

        curvature_norm = (
            curvature.multiply(-1)
            .divide(50)
            .clamp(0, 1)
        )

        # ==========================
        # Score Lavaka
        # ==========================
        lavaka_score = (
            bsi_norm.multiply(0.30)
            .add(ndti_norm.multiply(0.20))
            .add(slope_norm.multiply(0.20))
            .add(contrast_norm.multiply(0.15))
            .add(curvature_norm.multiply(0.15))
            .subtract(ndvi_norm.multiply(0.25))
        ).rename("lavaka_score")

        # ==========================
        # Contraintes fortes
        # ==========================
        bare_soil_mask = bsi.gt(0.10)

        slope_mask = slope.gt(10)

        vegetation_mask = ndvi.lt(0.35)

        curvature_mask = curvature.lt(-2)

        candidate_mask = (
            bare_soil_mask
            .And(slope_mask)
            .And(vegetation_mask)
            .And(curvature_mask)
        )

        # ==========================
        # Seuil final
        # ==========================
        lavaka_mask = (
            lavaka_score.gt(0.35)
            .And(candidate_mask)
            .selfMask()
        )

        # ==========================
        # Suppression petits objets
        # ==========================
        connected = lavaka_mask.connectedPixelCount(
            maxSize=200,
            eightConnected=True
        )

        lavaka_mask = (
            lavaka_mask
            .updateMask(connected.gte(20))
        )

        # ==========================
        # Surface totale
        # ==========================
        area_img = (
            lavaka_mask.multiply(
                ee.Image.pixelArea()
            )
        )

        area = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=watershed_geom,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        )

        area_m2 = area.get('lavaka_score')

        area_km2 = (
            area_m2.getInfo() / 1e6
            if area_m2 else 0
        )

        # ==========================
        # Comptage des lavakas
        # ==========================
        vectors = lavaka_mask.reduceToVectors(
            geometry=watershed_geom,
            scale=10,
            geometryType='polygon',
            eightConnected=True,
            maxPixels=1e13
        )

        num_lavakas = vectors.size().getInfo()

        return (
            lavaka_mask.clip(watershed_geom),
            lavaka_score,
            area_km2,
            num_lavakas
        )

    except Exception as e:
        st.error(f"Erreur détection lavakas : {e}")
        return None, None, 0.0, 0

def analyze_lavakas_trend(lavaka_data: pd.DataFrame) -> Optional[Dict]:
    """Analyse la tendance temporelle des lavakas."""
    if lavaka_data.empty:
        return None
    
    df = lavaka_data.copy()
    
    # Tendance de surface
    if 'area_km2' in df.columns and len(df) > 1:
        x = np.arange(len(df))
        y = df['area_km2'].values
        
        # Régression linéaire
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Taux de changement annuel
        annual_change = slope * 12
        
        return {
            'slope': slope,
            'annual_change': annual_change,
            'r_squared': r_value**2,
            'p_value': p_value,
            'total_change': y[-1] - y[0] if len(y) > 0 else 0
        }
    
    return None

# ==========================================================
# FONCTIONS ANALYSE ÉROSION
# ==========================================================
def plot_erosion_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    """Crée un graphique des séries temporelles d'érosion."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    for year in years[-3:]:  # 3 dernières années seulement
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['total_erosion_km2'],
                    mode='lines+markers',
                    name=str(year),
                    line=dict(width=2),
                    marker=dict(size=8)
                )
            )
    
    fig.update_layout(
        title=dict(
            text='Évolution mensuelle des zones d\'érosion totale',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Mois',
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=MONTHS_FR
        ),
        yaxis=dict(title='Surface d\'érosion (km²)'),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def calculate_monthly_erosion(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry) -> Dict[str, float]:
    """Calcule les surfaces d'érosion pour un mois spécifique."""
    try:
        # 1. Récupérer le NDVI du mois
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        
        # Collection Sentinel-2 pour le NDVI
        s2_collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
            .map(mask_s2_clouds)
        )
        
        if s2_collection.size().getInfo() > 0:
            ndvi = (
                s2_collection
                .map(calculate_ndvi)
                .median()
                .clip(watershed_geom)
                .rename('NDVI')
            )
        else:
            # Si pas d'image ce mois, utiliser NDVI moyen
            ndvi = ee.Image.constant(0.4).clip(watershed_geom).rename('NDVI')
        
        # 2. Pente (toujours la même)
        dem = ee.Image('USGS/SRTMGL1_003').clip(watershed_geom)
        slope = ee.Terrain.slope(dem)
        
        # 3. Classification mensuelle
        high_erosion = slope.gt(15).And(ndvi.lt(0.3))
        moderate_erosion = (
            slope.gte(5).And(slope.lte(15))
            .Or(ndvi.lt(0.5).And(ndvi.gte(0.3)))
        ).And(high_erosion.Not())
        low_erosion = high_erosion.Not().And(moderate_erosion.Not())
        
        # 4. Calcul des surfaces
        def get_area_km2(mask: ee.Image) -> float:
            try:
                area_image = mask.multiply(ee.Image.pixelArea())
                area_result = area_image.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=watershed_geom,
                    scale=30,
                    maxPixels=1e13,
                    bestEffort=True
                )
                area_dict = area_result.getInfo()
                if area_dict:
                    area_m2 = list(area_dict.values())[0]
                    return area_m2 / 1e6 if area_m2 else 0.0
                return 0.0
            except:
                return 0.0
        
        # 5. Retourner les résultats
        return {
            'year': year,
            'month': month,
            'low_erosion_km2': get_area_km2(low_erosion),
            'moderate_erosion_km2': get_area_km2(moderate_erosion),
            'high_erosion_km2': get_area_km2(high_erosion),
            'total_erosion_km2': get_area_km2(high_erosion.Or(moderate_erosion))
        }
        
    except Exception as e:
        st.warning(f"Erreur calcul érosion {year}-{month}: {e}")
        return {
            'year': year,
            'month': month,
            'low_erosion_km2': None,
            'moderate_erosion_km2': None,
            'high_erosion_km2': None,
            'total_erosion_km2': None
        }

def calculate_erosion_zones(watershed_geom: ee.Geometry) -> Tuple[ee.Image, Dict[str, float]]:
    """Calcule les zones d'érosion basé sur pente et NDVI."""
    try:
        # Pente
        dem = ee.Image('USGS/SRTMGL1_003').clip(watershed_geom)
        slope = ee.Terrain.slope(dem)
        
        # NDVI récent
        end_date = ee.Date(datetime.now().strftime("%Y-%m-%d"))
        start_date = end_date.advance(-1, 'month')
        
        ndvi_collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filterBounds(watershed_geom.bounds())
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        )
        
        if ndvi_collection.size().getInfo() > 0:
            ndvi = (
                ndvi_collection
                .map(mask_s2_clouds)
                .map(calculate_ndvi)
                .median()
                .clip(watershed_geom)
                .rename('NDVI')
            )
        else:
            ndvi = ee.Image.constant(0.4).clip(watershed_geom).rename('NDVI')
        
        # Classification
        high_erosion = slope.gt(15).And(ndvi.lt(0.3))
        moderate_erosion = (
            slope.gte(5).And(slope.lte(15))
            .Or(ndvi.lt(0.5).And(ndvi.gte(0.3)))
        ).And(high_erosion.Not())
        low_erosion = high_erosion.Not().And(moderate_erosion.Not())
        
        # Image classifiée
        erosion_image = (
            low_erosion.multiply(0)
            .add(moderate_erosion.multiply(1))
            .add(high_erosion.multiply(2))
        ).rename('erosion_class').clip(watershed_geom)
        
        # Calcul surfaces
        def get_area_km2(mask: ee.Image) -> float:
            try:
                area_image = mask.multiply(ee.Image.pixelArea())
                area_result = area_image.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=watershed_geom,
                    scale=30,
                    maxPixels=1e13,
                    bestEffort=True
                )
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
        
        # Surface totale
        try:
            area_total_m2 = watershed_geom.area(maxError=1).getInfo()
            area_total = area_total_m2 / 1e6
        except:
            area_total = area_low + area_moderate + area_high or 1.0
        
        return erosion_image, {
            'low': area_low,
            'moderate': area_moderate,
            'high': area_high,
            'total': area_total
        }
        
    except Exception as e:
        st.error(f"Erreur calcul zones érosion: {e}")
        default_image = ee.Image.constant(0).clip(watershed_geom).rename('erosion_class')
        return default_image, {'low': 0.0, 'moderate': 0.0, 'high': 0.0, 'total': 0.0}

# ==========================================================
# FONCTIONS ANALYSE LAC ITASY
# ==========================================================
def calculate_lake_metrics(year: int, month: int, aoi: ee.Geometry) -> Dict[str, Any]:
    """Calcule les métriques du lac pour un mois donné."""
    try:
        current_date = datetime.now()
        if year > current_date.year or (year == current_date.year and month > current_date.month):
            return {'status': 'future_date'}
        
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(mask_s2_clouds)
        )
        
        if collection.size().getInfo() == 0:
            return {'status': 'no_images'}
        
        # Calcul MNDWI
        mndwi_collection = collection.map(calculate_mndwi)
        mndwi_image = mndwi_collection.median().clip(aoi)
        water_mask = mndwi_image.gt(0.3)
        
        # Surface
        area_img = water_mask.multiply(ee.Image.pixelArea())
        area_result = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        )
        
        area_m2 = area_result.get('MNDWI')
        area_m2_value = area_m2.getInfo() if area_m2 else None
        
        # Statistiques MNDWI
        mean_result = mndwi_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e13,
            bestEffort=True
        )
        mean_mndwi = mean_result.get('MNDWI')
        mean_mndwi_value = mean_mndwi.getInfo() if mean_mndwi else None
        
        # Résultats
        if area_m2_value is None:
            surface_km2 = 0.0
            status = 'no_water_detected'
        else:
            surface_km2 = area_m2_value / 1e6
            status = 'success'
        
        water_percentage = min(100, (surface_km2 / LAKE_REFERENCE_AREA_KM2) * 100) if LAKE_REFERENCE_AREA_KM2 > 0 else 0
        
        return {
            'surface_km2': surface_km2,
            'mean_mndwi': mean_mndwi_value if mean_mndwi_value is not None else -1.0,
            'water_percentage': water_percentage,
            'status': status
        }
        
    except Exception as e:
        return {'status': f'error: {str(e)[:80]}'}

def analyze_lake_surface_trend(surface_data: pd.DataFrame) -> Tuple:
    """Analyse la tendance de la surface du lac."""
    df = surface_data.copy()
    df = df[df['surface_km2'] > 0]  # Supprimer les valeurs invalides
    
    if len(df) < 2:
        return None, None, None, None, None, None, None
    
    # Tendance linéaire
    x = np.arange(len(df))
    y = df['surface_km2'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Changement annuel
    annual_change = slope * 12  # Conversion mensuelle à annuelle
    
    # Changement total sur la période
    total_change = (df['surface_km2'].iloc[-1] - df['surface_km2'].iloc[0])
    percent_change = (total_change / df['surface_km2'].iloc[0]) * 100 if df['surface_km2'].iloc[0] > 0 else 0
    
    return slope, intercept, annual_change, percent_change, r_value, p_value, std_err

# ==========================================================
# FONCTIONS DE VISUALISATION
# ==========================================================
def create_temperature_map(watershed_geom: ee.Geometry, year: int, watershed_gdf: gpd.GeoDataFrame, month: int = None) -> folium.Map:
    """Carte de température centrée sur le lac Itasy."""
    try:
        # Centre sur le lac
        lake_coords = LAKE_ITASY_COORDS
        lons = [c[0] for c in lake_coords]
        lats = [c[1] for c in lake_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        center = [center_lat, center_lon]
        
        m = folium.Map(location=center, zoom_start=12, control_scale=True)
        
        # Polygone du lac
        folium.Polygon(
            locations=[[lat, lon] for lon, lat in lake_coords],
            color='blue', weight=2, fill=True, fill_color='blue', fill_opacity=0.2,
            popup='Lac Itasy'
        ).add_to(m)
        
        # Bassin versant
        folium.GeoJson(
            watershed_gdf.__geo_interface__,
            name="Bassin Versant",
            style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}
        ).add_to(m)
        
        # Données ERA5
        if month is None:
            start = ee.Date.fromYMD(year, 1, 1)
            end = ee.Date.fromYMD(year, 12, 31)
            title = f"Température moyenne annuelle {year}"
        else:
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, 'month')
            title = f"Température moyenne {calendar.month_name[month]} {year}"
        
        era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
            .filterDate(start, end) \
            .select('temperature_2m') \
            .mean() \
            .clip(watershed_geom)
        
        temp_celsius = era5.subtract(273.15).rename('temp_celsius')
        viz = {'min': 10, 'max': 30, 'palette': ['blue', 'lightblue', 'cyan', 'green', 'yellow', 'orange', 'red']}
        map_id = temp_celsius.getMapId(viz)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine | ERA5-Land',
            name=title,
            overlay=True,
            control=True
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
        
    except Exception as e:
        st.error(f"Erreur carte température: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)


def create_precip_map(watershed_geom: ee.Geometry, year: int, watershed_gdf: gpd.GeoDataFrame) -> folium.Map:
    """Carte des précipitations cumulées annuelles centrée sur le lac Itasy."""
    try:
        lake_coords = LAKE_ITASY_COORDS
        lons = [c[0] for c in lake_coords]
        lats = [c[1] for c in lake_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        center = [center_lat, center_lon]
        
        m = folium.Map(location=center, zoom_start=12, control_scale=True)
        
        folium.Polygon(
            locations=[[lat, lon] for lon, lat in lake_coords],
            color='blue', weight=2, fill=True, fill_color='blue', fill_opacity=0.2,
            popup='Lac Itasy'
        ).add_to(m)
        
        folium.GeoJson(
            watershed_gdf.__geo_interface__,
            name="Bassin Versant",
            style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}
        ).add_to(m)
        
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(start, end) \
            .sum() \
            .clip(watershed_geom)
        
        viz = {'min': 0, 'max': 1500, 'palette': ['white', 'lightblue', 'blue', 'darkblue']}
        map_id = chirps.getMapId(viz)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='CHIRPS',
            name=f"Précipitations {year} (mm)",
            overlay=True,
            control=True
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
        
    except Exception as e:
        st.error(f"Erreur carte précipitations: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)
        
def create_lavakas_map(watershed_gdf: gpd.GeoDataFrame, lavaka_mask: ee.Image, lavaka_score: ee.Image = None) -> folium.Map:
    """Carte interactive des lavakas avec fond en niveaux de gris."""
    try:
        lake_coords = LAKE_ITASY_COORDS
        lons = [c[0] for c in lake_coords]
        lats = [c[1] for c in lake_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        center = [center_lat, center_lon]

        m = folium.Map(location=center, zoom_start=12, control_scale=True)

        # Lac
        folium.Polygon(
            locations=[[lat, lon] for lon, lat in lake_coords],
            color='blue', weight=2, fill=True, fill_color='blue', fill_opacity=0.2, popup='Lac Itasy'
        ).add_to(m)

        # Bassin versant
        if watershed_gdf is not None and not watershed_gdf.empty:
            geojson_data = watershed_gdf.__geo_interface__
            folium.GeoJson(
                geojson_data, name="Bassin Versant",
                style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.1}
            ).add_to(m)

        # Lavakas
        if lavaka_mask is not None:
            viz = {'min': 0, 'max': 1, 'palette': ['#FFFF00', '#FFA500', '#FF0000', '#8B0000']}
            map_id = lavaka_mask.getMapId(viz)
            folium.TileLayer(
                tiles=map_id['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name="Lavakas", overlay=True, control=True
            ).add_to(m)

        if lavaka_score is not None:
            score_viz = {'min': 0, 'max': 0.8, 'palette': ['blue', 'cyan', 'yellow', 'red']}
            score_id = lavaka_score.getMapId(score_viz)
            folium.TileLayer(
                tiles=score_id['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name="Score", overlay=True, control=True
            ).add_to(m)

        # Fonds de carte en niveaux de gris (au lieu d'OSM)
        folium.TileLayer(
            'https://stamen-tiles-{s}.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}{r}.png',
            attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
            name='Fond gris',
            overlay=False,
            control=True
        ).add_to(m)

        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Erreur carte lavakas: {e}")
        return folium.Map(location=[-19.0, 46.8], zoom_start=11)
        
def create_erosion_map(watershed_gdf: gpd.GeoDataFrame, erosion_zones: Optional[ee.Image] = None) -> folium.Map:
    """Crée une carte interactive des zones d'érosion centrée sur le Lac Itasy."""
    try:
        # Coordonnées du Lac Itasy (format [lon, lat])
        lake_coords = LAKE_ITASY_COORDS
        # Calcul du centre
        lons = [c[0] for c in lake_coords]
        lats = [c[1] for c in lake_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        center = [center_lat, center_lon]  # folium attend [lat, lon]
        
        # Carte centrée sur le lac avec zoom adapté
        m = folium.Map(location=center, zoom_start=12, control_scale=True)
        
        # Ajouter le contour du lac pour repère
        folium.Polygon(
            locations=[[lat, lon] for lon, lat in lake_coords],
            color='blue',
            weight=2,
            fill=True,
            fill_color='blue',
            fill_opacity=0.2,
            popup='Lac Itasy'
        ).add_to(m)
        
        # Bassin versant
        geojson_data = watershed_gdf.__geo_interface__
        folium.GeoJson(
            geojson_data,
            name="Bassin Versant",
            style_function=lambda x: {
                'fillColor': '#3186cc',
                'color': '#3186cc',
                'weight': 2,
                'fillOpacity': 0.1
            }
        ).add_to(m)
        
        # Zones d'érosion
        if erosion_zones is not None:
            try:
                erosion_viz = {'min': 0, 'max': 2, 'palette': ['green', 'yellow', 'red']}
                map_id_dict = erosion_zones.getMapId(erosion_viz)
                folium.TileLayer(
                    tiles=map_id_dict['tile_fetcher'].url_format,
                    attr='Google Earth Engine',
                    name="Zones d'érosion",
                    overlay=True,
                    control=True
                ).add_to(m)
            except Exception as e:
                st.warning(f"Impossible d'ajouter les zones d'érosion: {e}")
        
        # Fonds de carte
        folium.TileLayer(
            'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            name='OpenStreetMap',
            attr='© OpenStreetMap contributors',
            control=True
        ).add_to(m)
        folium.TileLayer(
            'https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
            name='CartoDB Light',
            attr='© CartoDB',
            control=True
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        # Légende
        template = """
        {% macro html(this, kwargs) %}
        <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 8px;">
            <p style="margin: 0 0 5px 0; text-align: center;"><b>Légende</b></p>
            <p style="margin: 2px;"><i style="background:green; width:15px; height:15px; display:inline-block;"></i> Faible érosion</p>
            <p style="margin: 2px;"><i style="background:yellow; width:15px; height:15px; display:inline-block;"></i> Érosion modérée</p>
            <p style="margin: 2px;"><i style="background:red; width:15px; height:15px; display:inline-block;"></i> Forte érosion</p>
        </div>
        {% endmacro %}
        """
        macro = MacroElement()
        macro._template = Template(template)
        m.get_root().add_child(macro)
        
        return m
        
    except Exception as e:
        st.error(f"Erreur création carte érosion: {e}")
        # Carte de repli centrée sur le lac
        m = folium.Map(location=[-19.0, 46.8], zoom_start=12)
        return m

def plot_ndvi_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    """Crée un graphique des séries temporelles NDVI."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    for year in years:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['surface_km2'],
                    mode='lines+markers',
                    name=str(year),
                    line=dict(width=2),
                    marker=dict(size=8)
                )
            )
    
    fig.update_layout(
        title=dict(
            text='Évolution mensuelle de la végétation saine (NDVI > seuil)',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Mois',
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=MONTHS_FR
        ),
        yaxis=dict(title='Surface (km²)'),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_sediment_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    """Crée un graphique des séries temporelles sédimentaires."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    for year in years:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['sediment_index'],
                    mode='lines+markers',
                    name=str(year),
                    line=dict(width=2),
                    marker=dict(size=8, symbol='square')
                )
            )
    
    fig.update_layout(
        title=dict(
            text='Indice mensuel de contribution sédimentaire',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Mois',
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=MONTHS_FR
        ),
        yaxis=dict(title='Indice sédimentaire'),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_lake_surface_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    """Crée un graphique des séries temporelles de la surface du lac."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Surface mensuelle du lac', 'Tendance et moyenne mobile'),
        vertical_spacing=0.15
    )
    
    # Graphique 1: Surface mensuelle
    for year in years:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['surface_km2'],
                    mode='lines+markers',
                    name=str(year),
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
    
    # Graphique 2: Moyenne mobile et tendance
    df_sorted = df.sort_values(['year', 'month'])
    df_sorted['date'] = pd.to_datetime(df_sorted['year'].astype(str) + '-' + df_sorted['month'].astype(str) + '-01')
    df_sorted['moving_avg'] = df_sorted['surface_km2'].rolling(window=6, min_periods=1).mean()
    
    # Tendance linéaire
    if len(df_sorted) > 1:
        x = np.arange(len(df_sorted))
        y = df_sorted['surface_km2'].values
        slope, intercept = np.polyfit(x, y, 1)
        trend_line = intercept + slope * x
        
        fig.add_trace(
            go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['surface_km2'],
                mode='markers',
                name='Données mensuelles',
                marker=dict(size=4, color='lightblue'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['moving_avg'],
                mode='lines',
                name='Moyenne mobile (6 mois)',
                line=dict(color='orange', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_sorted['date'],
                y=trend_line,
                mode='lines',
                name='Tendance linéaire',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=2, col=1
        )
    
    # Mise en page
    fig.update_xaxes(
        title_text='Mois',
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=MONTHS_FR,
        row=1, col=1
    )
    
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=1)
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        template='plotly_white',
        title_text="Évolution de la surface du Lac Itasy",
        title_font_size=18
    )
    
    return fig

def plot_lavakas_timeseries(df: pd.DataFrame, years: List[int]) -> go.Figure:
    """Crée un graphique des séries temporelles des lavakas."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Surface totale des lavakas', 'Nombre de lavakas', 
                       'Évolution mensuelle', 'Tendance annuelle'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Graphique 1: Surface totale
    for year in years[-3:]:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['area_km2'],
                    mode='lines+markers',
                    name=str(year),
                    line=dict(width=2)
                ),
                row=1, col=1
            )
    
    # Graphique 2: Nombre de lavakas
    for year in years[-3:]:
        sub = df[df['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['num_lavakas'],
                    mode='lines+markers',
                    name=str(year),
                    line=dict(width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Graphique 3: Évolution moyenne mensuelle
    monthly_avg = df.groupby('month').agg({
        'area_km2': 'mean',
        'num_lavakas': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=monthly_avg['month'],
            y=monthly_avg['area_km2'],
            name='Surface moyenne',
            marker_color='sienna',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['num_lavakas'],
            mode='lines+markers',
            name='Nombre moyen',
            line=dict(color='darkred', width=3),
            yaxis='y2'
        ),
        row=2, col=1
    )
    
    # Graphique 4: Tendance annuelle
    yearly_trend = df.groupby('year').agg({
        'area_km2': 'sum',
        'num_lavakas': 'sum'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=yearly_trend['year'],
            y=yearly_trend['area_km2'],
            mode='lines+markers',
            name='Surface annuelle',
            line=dict(color='brown', width=3),
            marker=dict(size=10)
        ),
        row=2, col=2
    )
    
    # Mise en page
    fig.update_xaxes(title_text='Mois', tickmode='array', tickvals=list(range(1, 13)), 
                     ticktext=MONTHS_FR, row=1, col=1)
    fig.update_xaxes(title_text='Mois', tickmode='array', tickvals=list(range(1, 13)), 
                     ticktext=MONTHS_FR, row=1, col=2)
    fig.update_xaxes(title_text='Mois', tickmode='array', tickvals=list(range(1, 13)), 
                     ticktext=MONTHS_FR, row=2, col=1)
    fig.update_xaxes(title_text='Année', row=2, col=2)
    
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=1)
    fig.update_yaxes(title_text='Nombre', row=1, col=2)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=1)
    fig.update_yaxes(title_text='Nombre', row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=2)
    
    fig.update_layout(
        height=800,
        hovermode='x unified',
        template='plotly_white',
        title_text="Analyse temporelle des lavakas",
        title_font_size=18,
        showlegend=True
    )
    
    return fig

# ==========================================================
# FONCTIONS MÉTÉOROLOGIQUES (AJOUTEZ CETTE FONCTION)
# ==========================================================
def get_openmeteo_forecast(latitude: float, longitude: float) -> Optional[Dict]:
    """Récupère les prévisions météorologiques à 7 jours d'Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "rain_sum", "snowfall_sum",
            "windspeed_10m_max", "windgusts_10m_max",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration"
        ],
        "timezone": "auto",
        "forecast_days": 7
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"⚠️ Erreur récupération prévisions Open-Meteo: {e}")
        return None

def create_lake_dashboard(lake_data: pd.DataFrame) -> go.Figure:
    """Crée un tableau de bord complet pour l'analyse du lac."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Surface mensuelle', 'Variation saisonnière', 
                       'Tendance annuelle', 'Cycle annuel'),
        vertical_spacing=0.15
    )
    
    # Surface mensuelle
    years = sorted(lake_data['year'].unique())
    for year in years[-3:]:
        sub = lake_data[lake_data['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'], 
                    y=sub['surface_km2'], 
                    mode='lines+markers', 
                    name=str(year)
                ),
                row=1, col=1
            )
    
    # Variation saisonnière
    seasonal_data = lake_data.groupby('month')['surface_km2'].agg(['mean', 'std']).reset_index()
    fig.add_trace(
        go.Bar(
            x=seasonal_data['month'],
            y=seasonal_data['mean'],
            error_y=dict(type='data', array=seasonal_data['std']),
            name='Moyenne ± écart-type',
            marker_color='skyblue'
        ),
        row=1, col=2
    )
    
    # Tendance annuelle
    yearly_data = lake_data.groupby('year')['surface_km2'].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=yearly_data['year'],
            y=yearly_data['surface_km2'],
            mode='lines+markers',
            name='Surface moyenne annuelle',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ),
        row=2, col=1
    )
    
    # Cycle annuel avec bande d'incertitude
    monthly_stats = lake_data.groupby('month')['surface_km2'].agg(['mean', 'std', 'min', 'max']).reset_index()
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['mean'],
            mode='lines',
            name='Cycle moyen annuel',
            line=dict(color='darkblue', width=3)
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(monthly_stats['month']) + list(monthly_stats['month'][::-1]),
            y=list(monthly_stats['mean'] + monthly_stats['std']) + 
               list((monthly_stats['mean'] - monthly_stats['std'])[::-1]),
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 écart-type',
            showlegend=True
        ),
        row=2, col=2
    )
    
    # Mise en page
    fig.update_xaxes(title_text='Mois', row=1, col=1)
    fig.update_xaxes(title_text='Mois', row=1, col=2)
    fig.update_xaxes(title_text='Année', row=2, col=1)
    fig.update_xaxes(title_text='Mois', row=2, col=2)
    
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=1, col=2)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=1)
    fig.update_yaxes(title_text='Surface (km²)', row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title_text="Tableau de bord - Surveillance du Lac Itasy",
        title_font_size=20
    )
    
    return fig

def create_dashboard(df_ndvi: pd.DataFrame, df_sediment: pd.DataFrame, years: List[int]) -> go.Figure:
    """Crée un tableau de bord avec plusieurs visualisations."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Surface végétale saine', 'Contribution sédimentaire',
                       'Comparaison annuelle', 'Corrélation NDVI-Sédiments'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Graphique 1: NDVI par mois
    for year in years:
        sub = df_ndvi[df_ndvi['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['surface_km2'],
                    mode='lines+markers',
                    name=f'NDVI {year}',
                    legendgroup='ndvi',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Graphique 2: Sédiments par mois
    for year in years:
        sub = df_sediment[df_sediment['year'] == year]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub['month'],
                    y=sub['sediment_index'],
                    mode='lines+markers',
                    name=f'Sédiments {year}',
                    legendgroup='sed',
                    showlegend=True
                ),
                row=1, col=2
            )
    
    # Graphique 3: Comparaison annuelle moyenne
    yearly_ndvi = df_ndvi.groupby('year')['surface_km2'].mean().reset_index()
    yearly_sed = df_sediment.groupby('year')['sediment_index'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=yearly_ndvi['year'],
            y=yearly_ndvi['surface_km2'],
            name='NDVI moyen',
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=yearly_sed['year'],
            y=yearly_sed['sediment_index'],
            name='Sédiments moyen',
            marker_color='brown',
            opacity=0.7,
            yaxis='y2'
        ),
        row=2, col=1
    )
    
    # Graphique 4: Scatter plot corrélation
    merged_df = pd.merge(df_ndvi, df_sediment, on=['year', 'month'])
    fig.add_trace(
        go.Scatter(
            x=merged_df['surface_km2'],
            y=merged_df['sediment_index'],
            mode='markers',
            name='Corrélation',
            marker=dict(
                size=8,
                color=merged_df['month'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Mois")
            ),
            text=[f"Mois {m}" for m in merged_df['month']]
        ),
        row=2, col=2
    )
    
    # Mise à jour des layout
    fig.update_xaxes(title_text="Mois", row=1, col=1)
    fig.update_xaxes(title_text="Mois", row=1, col=2)
    fig.update_xaxes(title_text="Année", row=2, col=1)
    fig.update_xaxes(title_text="Surface végétale (km²)", row=2, col=2)
    
    fig.update_yaxes(title_text="Surface (km²)", row=1, col=1)
    fig.update_yaxes(title_text="Indice sédimentaire", row=1, col=2)
    fig.update_yaxes(title_text="Surface (km²)", row=2, col=1)
    fig.update_yaxes(title_text="Indice sédimentaire", row=2, col=2, secondary_y=True)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title_text="Tableau de bord d'analyse du bassin versant",
        title_font_size=20
    )
    
    return fig

def create_multi_year_meteo_comparison(meteo_stats: Dict, years: List[int]) -> go.Figure:
    """Crée un graphique de comparaison météo entre plusieurs années."""
    if meteo_stats is None or "monthly" not in meteo_stats:
        return None
    
    monthly_df = meteo_stats["monthly"]
    
    # Créer un subplot pour chaque variable
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Température moyenne par mois',
            'Précipitations totales par mois',
            'Température max par année',
            'Précipitations totales par année',
            'Radiation solaire par mois',
            'Évapotranspiration par mois'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Température moyenne par mois (comparaison année par année)
    for year in sorted(monthly_df['year'].unique()):
        year_data = monthly_df[monthly_df['year'] == year]
        fig.add_trace(
            go.Scatter(
                x=year_data['month'],
                y=year_data['temp_mean'],
                mode='lines+markers',
                name=f'T° {year}',
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # 2. Précipitations par mois (comparaison année par année)
    for year in sorted(monthly_df['year'].unique())[-3:]:  # 3 dernières années
        year_data = monthly_df[monthly_df['year'] == year]
        fig.add_trace(
            go.Bar(
                x=year_data['month'],
                y=year_data['precipitation'],
                name=f'Précip {year}',
                opacity=0.7
            ),
            row=1, col=2
        )
    
    # 3. Température max par année
    annual_stats = meteo_stats.get("annual")
    if annual_stats is not None and not annual_stats.empty:
        fig.add_trace(
            go.Bar(
                x=annual_stats['year'],
                y=annual_stats['temp_max'],
                name='Température max annuelle',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Précipitations totales par année
        fig.add_trace(
            go.Bar(
                x=annual_stats['year'],
                y=annual_stats['precipitation'],
                name='Précipitations annuelles',
                marker_color='blue',
                opacity=0.7
            ),
            row=2, col=2
        )
    
    # 5. Radiation solaire par mois (moyenne sur toutes les années)
    monthly_avg = monthly_df.groupby('month').agg({
        'radiation': 'mean',
        'evapotranspiration': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['radiation'],
            mode='lines+markers',
            name='Radiation solaire (moyenne)',
            line=dict(color='orange', width=3),
            marker=dict(size=8, symbol='square')
        ),
        row=3, col=1
    )
    
    # 6. Évapotranspiration par mois (moyenne sur toutes les années)
    fig.add_trace(
        go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['evapotranspiration'],
            mode='lines+markers',
            name='Évapotranspiration (moyenne)',
            line=dict(color='green', width=3),
            marker=dict(size=8, symbol='circle')
        ),
        row=3, col=2
    )
    
    # Mise en page
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
    
    fig.update_layout(
        height=900,
        showlegend=True,
        template='plotly_white',
        title_text="Comparaison météorologique multi-années",
        title_font_size=18,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_meteo_dashboard(meteo_stats: Dict, trends: Dict, forecast_data: Optional[Dict] = None) -> go.Figure:
    """Crée un tableau de bord météorologique complet."""
    if meteo_stats is None:
        return None
    
    # Créer des subplots avec 3 lignes, 2 colonnes
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Précipitations mensuelles',
            'Température moyenne mensuelle',
            'Évolution annuelle des précipitations',
            'Évolution annuelle de la température',
            'Anomalies de précipitations',
            'Prévisions à 7 jours'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    monthly_df = meteo_stats["monthly"]
    annual_df = meteo_stats["annual"]
    
    # 1. Précipitations mensuelles
    fig.add_trace(
        go.Bar(
            x=monthly_df["date"],
            y=monthly_df["precipitation"],
            name="Précipitations mensuelles",
            marker_color="blue",
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 2. Température moyenne mensuelle
    fig.add_trace(
        go.Scatter(
            x=monthly_df["date"],
            y=monthly_df["temp_mean"],
            mode="lines+markers",
            name="Température moyenne",
            line=dict(color="red", width=2),
            marker=dict(size=4)
        ),
        row=1, col=2
    )
    
    # 3. Évolution annuelle des précipitations
    fig.add_trace(
        go.Bar(
            x=annual_df["year"],
            y=annual_df["precipitation"],
            name="Précipitations annuelles",
            marker_color="navy"
        ),
        row=2, col=1
    )
    
    # Ajouter la tendance des précipitations
    if trends:
        x_trend = annual_df["year"]
        y_trend = trends["precipitation_slope"] * (x_trend - x_trend.min()) + annual_df["precipitation"].mean()
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name="Tendance précipitations",
                line=dict(color="orange", width=2, dash="dash")
            ),
            row=2, col=1
        )
    
    # 4. Évolution annuelle de la température
    fig.add_trace(
        go.Scatter(
            x=annual_df["year"],
            y=annual_df["temp_mean"],
            mode="lines+markers",
            name="Température moyenne annuelle",
            line=dict(color="darkred", width=3),
            marker=dict(size=8, symbol="circle")
        ),
        row=2, col=2
    )
    
    # Ajouter la tendance de température
    if trends:
        x_temp_trend = annual_df["year"]
        y_temp_trend = trends["temperature_slope"] * (x_temp_trend - x_temp_trend.min()) + annual_df["temp_mean"].mean()
        fig.add_trace(
            go.Scatter(
                x=x_temp_trend,
                y=y_temp_trend,
                mode="lines",
                name="Tendance température",
                line=dict(color="orange", width=2, dash="dash")
            ),
            row=2, col=2
        )
    
    # 5. Anomalies de précipitations
    if trends and "anomaly_data" in trends:
        anomaly_df = trends["anomaly_data"]
        colors = ["red" if x < 0 else "blue" for x in anomaly_df["precip_anomaly"]]
        
        fig.add_trace(
            go.Bar(
                x=anomaly_df["year"],
                y=anomaly_df["precip_anomaly"],
                name="Anomalies précipitations",
                marker_color=colors,
                opacity=0.7
            ),
            row=3, col=1
        )
    
    # 6. Prévisions à 7 jours
    if forecast_data and "daily" in forecast_data:
        forecast_daily = forecast_data["daily"]
        
        # Créer un DataFrame pour les prévisions
        forecast_df = pd.DataFrame({
            "date": pd.to_datetime(forecast_daily["time"]),
            "temp_max": forecast_daily.get("temperature_2m_max", []),
            "temp_min": forecast_daily.get("temperature_2m_min", []),
            "precipitation": forecast_daily.get("precipitation_sum", []),
        })
        
        # Ajouter les températures (double axe Y)
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["temp_max"],
                mode="lines+markers",
                name="Température max (prévision)",
                line=dict(color="red", width=2),
                marker=dict(size=6)
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["temp_min"],
                mode="lines+markers",
                name="Température min (prévision)",
                line=dict(color="blue", width=2),
                marker=dict(size=6)
            ),
            row=3, col=2
        )
        
        # Ajouter les précipitations sur un axe secondaire
        fig.add_trace(
            go.Bar(
                x=forecast_df["date"],
                y=forecast_df["precipitation"],
                name="Précipitations (prévision)",
                marker_color="lightblue",
                opacity=0.6,
                yaxis="y2"
            ),
            row=3, col=2
        )
        
        # Configurer le double axe Y pour les prévisions (CORRIGÉ)
        fig.update_layout(
            yaxis5=dict(title="Température (°C)", title_font=dict(color="red")),
            yaxis6=dict(
                title="Précipitations (mm)",
                title_font=dict(color="blue"),
                overlaying="y5",
                side="right"
            )
        )
    else:
        # Message si pas de prévisions
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Prévisions non disponibles",
            showarrow=False,
            font=dict(size=16),
            row=3, col=2
        )
    
    # Mise en page
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
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        template="plotly_white",
        title_text="Tableau de bord météorologique - Bassin versant Lac Itasy",
        title_font_size=18,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# ==========================================================
# INTERFACE PRINCIPALE
# ==========================================================
def main():
    """Fonction principale de l'application."""
    st.title("🌿 Analyse du Bassin Versant du Lac Itasy")
    st.markdown("Analyse de la couverture végétale, érosion et sédimentation avec données satellitaires.")
    
    # Sidebar
    st.sidebar.header("⚙️ Paramètres d'analyse")
    
    if not initialize_ee():
        st.sidebar.warning("Authentification Earth Engine requise")
        return
    
    # ==========================================================
    # 1. DONNÉES GÉOGRAPHIQUES
    # ==========================================================
    st.sidebar.subheader("1. Données géographiques")
    use_default = st.sidebar.checkbox("Utiliser shapefile par défaut", value=True)
    
    watershed_data = None
    if use_default:
        try:
            shapefile_path = 'data/BV_SBV_Lac_Itasy_20mars.shp'
            if os.path.exists(shapefile_path):
                watershed_data = load_watershed_shapefile(shapefile_path)
                # Vérifier explicitement que tous les éléments sont non None
                if all(v is not None for v in watershed_data):
                    st.sidebar.success("✅ Shapefile par défaut chargé")
                else:
                    st.sidebar.warning("⚠️ Shapefile par défaut invalide")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur shapefile par défaut: {str(e)[:50]}...")
    
    if not watershed_data:
        uploaded_file = st.sidebar.file_uploader("📤 Uploader shapefile", type=['shp', 'zip'])
        if uploaded_file:
            with st.sidebar.spinner("Chargement du shapefile..."):
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
    
    # ==========================================================
    # 2. PÉRIODE D'ANALYSE (AMÉLIORÉE)
    # ==========================================================
    st.sidebar.subheader("2. Période d'analyse")
    current_year = datetime.now().year
    years_options = list(range(2015, current_year + 1))
    
    # Mode de sélection des années
    year_selection_mode = st.sidebar.radio(
        "Mode de sélection",
        ["Plage d'années", "Années spécifiques"],
        help="• Plage: années consécutives\n• Spécifiques: choisir années particulières",
        index=0
    )
    
    YEARS_TO_ANALYZE = []
    
    if year_selection_mode == "Plage d'années":
        # Sélection par plage continue
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_year = st.selectbox(
                "Année début", 
                years_options, 
                index=years_options.index(2020) if 2020 in years_options else 0,
                help="Première année à analyser"
            )
        
        with col2:
            end_year = st.selectbox(
                "Année fin", 
                years_options, 
                index=years_options.index(current_year - 1) if (current_year - 1) in years_options else -1,
                help="Dernière année à analyser"
            )
        
        if start_year > end_year:
            st.sidebar.error("❌ L'année début doit être inférieure ou égale à l'année fin")
            show_welcome_page()
            return
        
        YEARS_TO_ANALYZE = list(range(start_year, end_year + 1))
        
    else:
        # Sélection d'années spécifiques (multi-sélection)
        selected_years = st.sidebar.multiselect(
            "Sélectionnez les années à analyser",
            years_options,
            default=[2020, 2021, 2022, 2023] if current_year >= 2023 else years_options[-4:],
            help="Ctrl+clic pour sélectionner plusieurs années"
        )
        
        if not selected_years:
            st.sidebar.warning("⚠️ Veuillez sélectionner au moins une année")
            show_welcome_page()
            return
        
        YEARS_TO_ANALYZE = sorted(selected_years)
    
    # Affichage récapitulatif
    years_count = len(YEARS_TO_ANALYZE)
    years_text = f"{YEARS_TO_ANALYZE[0]}-{YEARS_TO_ANALYZE[-1]}" if years_count > 3 else ", ".join(map(str, YEARS_TO_ANALYZE))
    st.sidebar.info(f"📅 **{years_count} année(s) sélectionnée(s):** {years_text}")
    
    # ==========================================================
    # 3. OPTIONS D'ANALYSE PAR CATÉGORIE
    # ==========================================================
    st.sidebar.subheader("3. Options d'analyse")
    
    # 3a. ANALYSE VÉGÉTATION ET SÉDIMENTS
    with st.sidebar.expander("🌱 Végétation et sédiments", expanded=True):
        analyze_ndvi = st.checkbox("Analyse végétation (NDVI)", value=True, 
                                   help="Calcul de la surface végétale saine")
        if analyze_ndvi:
            NDVI_THRESHOLD = st.slider("Seuil NDVI", 0.0, 1.0, NDVI_THRESHOLD_DEFAULT, 0.05,
                                       help="Valeur minimale pour considérer la végétation comme 'saine'")
        else:
            NDVI_THRESHOLD = NDVI_THRESHOLD_DEFAULT
        
        analyze_sediment = st.checkbox("Analyse sédimentaire", value=True,
                                       help="Calcul des indices d'érosion et sédimentation")
    
    # 3b. ANALYSE ÉROSION (AVEC OPTION TEMPORELLE)
    with st.sidebar.expander("⚠️ Érosion", expanded=True):
        analyze_erosion_zones = st.checkbox("Analyser zones d'érosion", value=True)
        
        if analyze_erosion_zones:
            erosion_analysis_type = st.radio(
                "Type d'analyse érosion",
                ["État actuel", "Évolution temporelle"],
                help="• État actuel: carte récente\n• Évolution: analyse sur plusieurs années",
                index=0
            )
    
    # 3c. ANALYSE LAC ET LAVAKAS
    with st.sidebar.expander("🌊🌋 Hydrologie", expanded=False):
        analyze_lake = st.checkbox("Surveiller surface lac", value=True,
                                   help="Suivi de la surface du Lac Itasy")
        analyze_lavakas = st.checkbox("Détecter lavakas", value=False,
                                      help="Identification des ravines d'érosion (lavakas)")
    
    # 3d. ANALYSE MÉTÉOROLOGIQUE (AVEC OPTIONS AVANCÉES)
    with st.sidebar.expander("🌤️ Météorologie", expanded=False):
        analyze_meteo = st.checkbox("Analyse météorologique", value=False)
        
        if analyze_meteo:
            # Options pour la période météo
            meteo_range = st.radio(
                "Période des données météo",
                ["Même que l'analyse", "Personnalisée", "Dernières 5 années"],
                help="Choisissez la période pour les données météo",
                index=0
            )
            
            if meteo_range == "Personnalisée":
                col1, col2 = st.columns(2)
                with col1:
                    meteo_start_year = st.selectbox(
                        "Début météo",
                        years_options,
                        index=years_options.index(YEARS_TO_ANALYZE[0]) if YEARS_TO_ANALYZE[0] in years_options else 0
                    )
                with col2:
                    meteo_end_year = st.selectbox(
                        "Fin météo", 
                        years_options,
                        index=years_options.index(YEARS_TO_ANALYZE[-1]) if YEARS_TO_ANALYZE[-1] in years_options else -1
                    )
            elif meteo_range == "Dernières 5 années":
                meteo_start_year = current_year - 5
                meteo_end_year = current_year
                st.info(f"📅 Période: {meteo_start_year} - {meteo_end_year}")
            else:  # Même que l'analyse
                meteo_start_year = YEARS_TO_ANALYZE[0]
                meteo_end_year = YEARS_TO_ANALYZE[-1]
        else:
            # Valeurs par défaut pour éviter les erreurs de référence si la météo n'est pas activée
            meteo_range = "Même que l'analyse"
            meteo_start_year = YEARS_TO_ANALYZE[0]
            meteo_end_year = YEARS_TO_ANALYZE[-1]
    
    # ==========================================================
    # 3e. CARTES SUPPLÉMENTAIRES (NOUVEAU)
    # ==========================================================
    with st.sidebar.expander("🗺️ Cartes supplémentaires", expanded=False):
        show_lavakas_map = st.checkbox("Afficher carte des lavakas", value=True)
        show_temp_map = st.checkbox("Afficher carte température (moyenne annuelle)", value=False)
        show_precip_map = st.checkbox("Afficher carte précipitations (cumul annuel)", value=False)
        
        if show_temp_map or show_precip_map:
            map_year = st.selectbox("Année pour les cartes climatiques", years_options, index=len(years_options)-1)
        else:
            map_year = current_year
    
    # ==========================================================
    # 4. PARAMÈTRES AVANCÉS (OPTIONNEL)
    # ==========================================================
    with st.sidebar.expander("⚙️ Paramètres avancés", expanded=False):
        # Option de résolution
        analysis_resolution = st.select_slider(
            "Résolution d'analyse",
            options=["Rapide (30m)", "Standard (20m)", "Détaillée (10m)"],
            value="Standard (20m)",
            help="Résolution spatiale des calculs (plus détaillé = plus long)"
        )
        
        # Correspondance résolution -> échelle
        resolution_map = {
            "Rapide (30m)": 30,
            "Standard (20m)": 20, 
            "Détaillée (10m)": 10
        }
        analysis_scale = resolution_map[analysis_resolution]
        
        # Option pour limiter le nombre de mois analysés
        if len(YEARS_TO_ANALYZE) > 3:
            limit_months = st.checkbox(
                "Limiter aux mois clés (saison sèche/pluie)", 
                value=False,
                help="Analyser seulement les mois représentatifs pour gagner du temps"
            )
        else:
            limit_months = False
    
    # ==========================================================
    # 5. RÉCAPITULATIF ET LANCEMENT
    # ==========================================================
    st.sidebar.markdown("---")
    
    # Comptage des analyses sélectionnées
    analyses_selected = sum([
        1 if analyze_ndvi else 0,
        1 if analyze_sediment else 0,
        1 if analyze_erosion_zones else 0,
        1 if analyze_lake else 0,
        1 if analyze_lavakas else 0,
        1 if analyze_meteo else 0
    ])
    
    # Affichage du récapitulatif
    st.sidebar.subheader(f"📋 Récapitulatif ({analyses_selected} analyse(s))")
    
    recap_text = []
    if analyze_ndvi:
        recap_text.append(f"• 🌱 NDVI (seuil: {NDVI_THRESHOLD})")
    if analyze_sediment:
        recap_text.append("• 🏔️ Sédiments")
    if analyze_erosion_zones:
        type_text = "carte" if erosion_analysis_type == "État actuel" else "évolution"
        recap_text.append(f"• ⚠️ Érosion ({type_text})")
    if analyze_lake:
        recap_text.append("• 🌊 Surface lac")
    if analyze_lavakas:
        recap_text.append("• 🌋 Lavakas")
    if analyze_meteo:
        recap_text.append(f"• 🌤️ Météo ({meteo_start_year}-{meteo_end_year})")
    if show_temp_map:
        recap_text.append(f"• 🌡️ Carte température ({map_year})")
    if show_precip_map:
        recap_text.append(f"• 🌧️ Carte précipitations ({map_year})")
    
    if recap_text:
        st.sidebar.markdown("\n".join(recap_text))
    
    # Estimation du temps
    estimated_time = analyses_selected * 2  # ~2 minutes par analyse
    if len(YEARS_TO_ANALYZE) > 3:
        estimated_time *= 1.5
    
    st.sidebar.info(f"⏱️ Estimation: {estimated_time} min")
    
    # Stocker les choix des cartes dans session_state pour les récupérer dans run_analysis
    st.session_state.show_lavakas_map = show_lavakas_map
    st.session_state.show_temp_map = show_temp_map
    st.session_state.show_precip_map = show_precip_map
    st.session_state.map_year = map_year
    
    # Bouton de lancement
    if st.sidebar.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True):
        # Préparer les paramètres avancés
        advanced_params = {
            'scale': analysis_scale,
            'limit_months': limit_months,
            'resolution': analysis_resolution
        }
        
        # Préparer les paramètres météo si activée
        meteo_params = None
        if analyze_meteo:
            meteo_params = {
                'start_year': meteo_start_year,
                'end_year': meteo_end_year,
                'use_custom_range': (meteo_range == "Personnalisée"),
                'is_last_5_years': (meteo_range == "Dernières 5 années")
            }
        
        # Préparer les paramètres érosion si activée
        erosion_params = None
        if analyze_erosion_zones:
            erosion_params = {
                'analysis_type': erosion_analysis_type,
                'is_temporal': (erosion_analysis_type == "Évolution temporelle")
            }
        
        # Lancer l'analyse avec tous les paramètres
        run_analysis(
            watershed_geom=watershed_geom,
            aoi=aoi,
            watershed_gdf=watershed_gdf,
            years=YEARS_TO_ANALYZE,
            ndvi_threshold=NDVI_THRESHOLD,
            analyze_ndvi=analyze_ndvi,
            analyze_sediment=analyze_sediment,
            analyze_erosion=analyze_erosion_zones,
            analyze_lake=analyze_lake,
            analyze_lavakas=analyze_lavakas,
            analyze_meteo=analyze_meteo,
            meteo_params=meteo_params,
            erosion_params=erosion_params,
            advanced_params=advanced_params
        )
    else:
        # Afficher la page d'accueil
        show_welcome_page()

def run_analysis(watershed_geom, aoi, watershed_gdf, years, ndvi_threshold,
                 analyze_ndvi, analyze_sediment, analyze_erosion,
                 analyze_lake, analyze_lavakas, analyze_meteo,
                 meteo_params=None, erosion_params=None, advanced_params=None):
    """Exécute l'analyse complète."""

    # --- INITIALISATION DES VARIABLES POUR LE TABLEAU DE BORD ---
    df_ndvi = None
    df_sediment = None
    stats = None          # érosion état actuel
    df_lake = None
    df_lavakas = None
    df_meteo = None
    # ---

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_analyses = sum([
        1 if analyze_ndvi else 0,
        1 if analyze_sediment else 0,
        1 if analyze_erosion else 0,
        1 if analyze_lake else 0,
        1 if analyze_lavakas else 0,
        1 if analyze_meteo else 0
    ])

    if total_analyses == 0:
        st.warning("Veuillez sélectionner au moins une analyse à effectuer")
        return

    current_analysis = 0

    # ==========================================================
    # ANALYSE VÉGÉTATION (NDVI)
    # ==========================================================
    if analyze_ndvi:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Analyse végétation (NDVI)...")

        st.header("🌱 Analyse de la couverture végétale (NDVI)")

        with st.spinner("Calcul des surfaces végétales..."):
            ndvi_rows = []
            months_to_process = get_months_to_process(years, datetime.now().year, datetime.now().month)
            total_months = len(months_to_process)

            for i, (year, month) in enumerate(months_to_process, 1):
                progress = (current_analysis - 1 + (i / total_months)) / total_analyses
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Traitement NDVI {year}-{month:02d}...")

                area = compute_monthly_ndvi_area(year, month, watershed_geom, aoi, ndvi_threshold)
                ndvi_rows.append({'year': year, 'month': month, 'surface_km2': area if area is not None else None})

            df_ndvi = pd.DataFrame(ndvi_rows)

            # Affichage des résultats
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Données NDVI")
                st.dataframe(df_ndvi.round(3), use_container_width=True)
                csv = df_ndvi.to_csv(index=False)
                st.download_button("📥 Télécharger les données NDVI", csv,
                                   f"ndvi_analysis_{years[0]}_{years[-1]}.csv", "text/csv")
            with col2:
                st.subheader("Statistiques")
                if not df_ndvi.empty:
                    stats_df = df_ndvi.groupby('year').agg({'surface_km2': ['mean', 'min', 'max', 'sum']}).round(2)
                    stats_df.columns = ['Moyenne', 'Minimum', 'Maximum', 'Total']
                    st.dataframe(stats_df, use_container_width=True)

            st.subheader("📈 Évolution temporelle")
            fig_ndvi = plot_ndvi_timeseries(df_ndvi, years)
            st.plotly_chart(fig_ndvi, use_container_width=True)

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
            months_to_process = get_months_to_process(years, datetime.now().year, datetime.now().month)
            total_months = len(months_to_process)

            for i, (year, month) in enumerate(months_to_process, 1):
                progress = (current_analysis - 1 + (i / total_months)) / total_analyses
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
                st.download_button("📥 Télécharger les données sédimentaires", csv_sed,
                                   f"sediment_analysis_{years[0]}_{years[-1]}.csv", "text/csv")
            with col2:
                st.subheader("Statistiques sédimentaires")
                if not df_sediment.empty:
                    stats_sed = df_sediment.groupby('year').agg({'sediment_index': ['mean', 'min', 'max', 'std']}).round(4)
                    stats_sed.columns = ['Moyenne', 'Minimum', 'Maximum', 'Écart-type']
                    st.dataframe(stats_sed, use_container_width=True)

            st.subheader("📉 Évolution des indices sédimentaires")
            fig_sed = plot_sediment_timeseries(df_sediment, years)
            st.plotly_chart(fig_sed, use_container_width=True)

        # ==========================================================
    # ANALYSE ÉROSION (VERSION COMPLÈTE)
    # ==========================================================
    if analyze_erosion:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        
        # VÉRIFIER LES PARAMÈTRES D'ÉROSION
        is_temporal_analysis = False
        if erosion_params:
            is_temporal_analysis = erosion_params.get('is_temporal', False)
            analysis_type_name = erosion_params.get('analysis_type', 'État actuel')
        else:
            analysis_type_name = "État actuel"
            is_temporal_analysis = False
        
        # ======================================================
        # OPTION 1: ANALYSE TEMPORELLE (ÉVOLUTION)
        # ======================================================
        if is_temporal_analysis:
            status_text.text("📈 Analyse évolution temporelle de l'érosion...")
            
            st.header("📈 Évolution Temporelle des Zones d'Érosion")
            st.info("Cette analyse montre comment les surfaces érodées ont évolué au fil du temps.")
            
            # Vérifier si la fonction calculate_monthly_erosion existe
            try:
                calculate_monthly_erosion
            except NameError:
                st.error("""
                ❌ **Fonction manquante**
                
                La fonction `calculate_monthly_erosion` n'est pas définie.
                Veuillez l'ajouter dans la section des fonctions d'analyse.
                """)
                st.code("""
                # À ajouter dans vos fonctions d'analyse :
                def calculate_monthly_erosion(year: int, month: int, watershed_geom: ee.Geometry, aoi: ee.Geometry) -> Dict[str, float]:
                    \"\"\"Calcule les surfaces d'érosion pour un mois spécifique.\"\"\"
                    # Votre code ici...
                """)
                return
            
            with st.spinner("Calcul de l'évolution de l'érosion sur plusieurs années..."):
                erosion_rows = []
                
                # Utiliser la fonction utilitaire pour générer les mois à traiter
                months_to_process = get_months_to_process(years, datetime.now().year, datetime.now().month)
                total_months = len(months_to_process)
                
                # Barre de progression détaillée
                progress_text = st.empty()
                erosion_progress = st.progress(0)
                
                if total_months == 0:
                    st.warning("⚠️ Aucun mois à analyser. Vérifiez les années sélectionnées.")
                    return
                
                for i, (year, month) in enumerate(months_to_process, 1):
                    # Mise à jour de la progression
                    progress_percent = i / total_months
                    erosion_progress.progress(progress_percent)
                    progress_text.text(f"Traitement érosion {year}-{month:02d} ({i}/{total_months})...")
                    
                    # Calculer l'érosion pour ce mois
                    try:
                        erosion_data = calculate_monthly_erosion(year, month, watershed_geom, aoi)
                        erosion_rows.append(erosion_data)
                        
                        # Mise à jour de la progression globale
                        overall_progress = (current_analysis - 1 + progress_percent) / total_analyses
                        progress_bar.progress(min(overall_progress, 1.0))
                        
                    except Exception as e:
                        st.warning(f"⚠️ Erreur pour {year}-{month}: {str(e)[:100]}...")
                        # Ajouter des valeurs nulles pour maintenir la structure
                        erosion_rows.append({
                            'year': year,
                            'month': month,
                            'low_erosion_km2': None,
                            'moderate_erosion_km2': None,
                            'high_erosion_km2': None,
                            'total_erosion_km2': None
                        })
                
                # Nettoyer la progression détaillée
                progress_text.empty()
                erosion_progress.empty()
                
                # Créer le DataFrame
                df_erosion = pd.DataFrame(erosion_rows)
                
                # ======================================================
                # AFFICHAGE DES RÉSULTATS TEMPORELS
                # ======================================================
                if not df_erosion.empty:
                    # Filtrer les lignes avec données valides
                    df_valid = df_erosion.dropna(subset=['total_erosion_km2'])
                    
                    if len(df_valid) > 0:
                        # 1. RÉSUMÉ STATISTIQUE
                        st.subheader("📊 Résumé statistique")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_high = df_valid['high_erosion_km2'].mean()
                            st.metric("Érosion forte moyenne", f"{avg_high:.1f} km²")
                        
                        with col2:
                            max_high = df_valid['high_erosion_km2'].max()
                            max_date = df_valid.loc[df_valid['high_erosion_km2'].idxmax(), ['year', 'month']]
                            st.metric("Maximum érosion forte", f"{max_high:.1f} km²", 
                                     f"{int(max_date['month'])}/{int(max_date['year'])}")
                        
                        with col3:
                            total_months_analyzed = len(df_valid)
                            years_analyzed = len(df_valid['year'].unique())
                            st.metric("Période analysée", f"{years_analyzed} an(s)", 
                                     f"{total_months_analyzed} mois")
                        
                        with col4:
                            trend_text = "📈 Hausse" if df_valid['high_erosion_km2'].iloc[-1] > df_valid['high_erosion_km2'].iloc[0] else "📉 Baisse"
                            change = ((df_valid['high_erosion_km2'].iloc[-1] - df_valid['high_erosion_km2'].iloc[0]) / 
                                     df_valid['high_erosion_km2'].iloc[0] * 100) if df_valid['high_erosion_km2'].iloc[0] > 0 else 0
                            st.metric("Tendance globale", trend_text, f"{change:.1f}%")
                        
                        # 2. VISUALISATIONS
                        st.subheader("📈 Visualisations")
                        
                        # Créer la colonne date pour les graphiques
                        df_valid['date'] = pd.to_datetime(
                            df_valid['year'].astype(str) + '-' + 
                            df_valid['month'].astype(str) + '-01'
                        )
                        
                        # Onglets pour différents graphiques
                        tab1, tab2, tab3, tab4 = st.tabs(["Évolution temporelle", "Comparaison annuelle", "Saisonnalité", "Données brutes"])
                        
                        with tab1:
                            # Graphique d'évolution temporelle
                            fig_temporal = go.Figure()
                            
                            # Ajouter les trois catégories d'érosion
                            fig_temporal.add_trace(go.Scatter(
                                x=df_valid['date'], y=df_valid['high_erosion_km2'],
                                mode='lines+markers', name='Érosion forte',
                                line=dict(color='red', width=2),
                                marker=dict(size=4)
                            ))
                            
                            fig_temporal.add_trace(go.Scatter(
                                x=df_valid['date'], y=df_valid['moderate_erosion_km2'],
                                mode='lines', name='Érosion modérée',
                                line=dict(color='orange', width=1.5),
                                fill='tonexty'
                            ))
                            
                            fig_temporal.add_trace(go.Scatter(
                                x=df_valid['date'], y=df_valid['low_erosion_km2'],
                                mode='lines', name='Érosion faible',
                                line=dict(color='green', width=1),
                                fill='tonexty'
                            ))
                            
                            fig_temporal.update_layout(
                                title='Évolution des surfaces érodées',
                                xaxis_title='Date',
                                yaxis_title='Surface (km²)',
                                hovermode='x unified',
                                height=500,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_temporal, use_container_width=True)
                        
                        with tab2:
                            # Comparaison annuelle
                            yearly_comparison = df_valid.groupby('year').agg({
                                'low_erosion_km2': 'mean',
                                'moderate_erosion_km2': 'mean',
                                'high_erosion_km2': 'mean'
                            }).reset_index()
                            
                            fig_yearly = go.Figure()
                            
                            fig_yearly.add_trace(go.Bar(
                                x=yearly_comparison['year'],
                                y=yearly_comparison['high_erosion_km2'],
                                name='Érosion forte',
                                marker_color='red'
                            ))
                            
                            fig_yearly.add_trace(go.Bar(
                                x=yearly_comparison['year'],
                                y=yearly_comparison['moderate_erosion_km2'],
                                name='Érosion modérée',
                                marker_color='orange'
                            ))
                            
                            fig_yearly.add_trace(go.Bar(
                                x=yearly_comparison['year'],
                                y=yearly_comparison['low_erosion_km2'],
                                name='Érosion faible',
                                marker_color='green'
                            ))
                            
                            fig_yearly.update_layout(
                                title='Comparaison annuelle moyenne',
                                xaxis_title='Année',
                                yaxis_title='Surface moyenne (km²)',
                                barmode='stack',
                                height=500,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_yearly, use_container_width=True)
                        
                        with tab3:
                            # Analyse de saisonnalité
                            monthly_avg = df_valid.groupby('month').agg({
                                'high_erosion_km2': ['mean', 'std', 'min', 'max']
                            }).round(2)
                            monthly_avg.columns = ['Moyenne', 'Écart-type', 'Minimum', 'Maximum']
                            monthly_avg = monthly_avg.reset_index()
                            
                            fig_seasonal = go.Figure()
                            
                            fig_seasonal.add_trace(go.Scatter(
                                x=monthly_avg['month'], y=monthly_avg['Moyenne'],
                                mode='lines+markers', name='Moyenne',
                                line=dict(color='blue', width=3),
                                error_y=dict(
                                    type='data',
                                    array=monthly_avg['Écart-type'],
                                    visible=True
                                )
                            ))
                            
                            fig_seasonal.add_trace(go.Scatter(
                                x=monthly_avg['month'], y=monthly_avg['Minimum'],
                                mode='lines', name='Minimum',
                                line=dict(color='lightblue', width=1, dash='dash'),
                                showlegend=True
                            ))
                            
                            fig_seasonal.add_trace(go.Scatter(
                                x=monthly_avg['month'], y=monthly_avg['Maximum'],
                                mode='lines', name='Maximum',
                                line=dict(color='darkblue', width=1, dash='dash'),
                                showlegend=True
                            ))
                            
                            fig_seasonal.update_layout(
                                title='Saisonnalité de l\'érosion forte',
                                xaxis_title='Mois',
                                yaxis_title='Surface (km²)',
                                xaxis=dict(
                                    tickmode='array',
                                    tickvals=list(range(1, 13)),
                                    ticktext=MONTHS_FR
                                ),
                                height=500,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_seasonal, use_container_width=True)
                        
                        with tab4:
                            # Données brutes
                            st.dataframe(
                                df_valid.round(2).sort_values(['year', 'month']).style.background_gradient(
                                    subset=['high_erosion_km2', 'total_erosion_km2'],
                                    cmap='Reds'
                                ),
                                use_container_width=True,
                                height=400
                            )
                        
                        # 3. TÉLÉCHARGEMENT
                        st.subheader("💾 Export des données")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            # Données détaillées
                            csv_detailed = df_valid.to_csv(index=False, encoding='utf-8')
                            st.download_button(
                                label="📥 Données détaillées (CSV)",
                                data=csv_detailed,
                                file_name=f"erosion_detailed_{years[0]}_{years[-1]}.csv",
                                mime="text/csv",
                                help="Contient toutes les données mensuelles"
                            )
                        
                        with col_dl2:
                            # Données résumées
                            yearly_summary = df_valid.groupby('year').agg({
                                'low_erosion_km2': 'mean',
                                'moderate_erosion_km2': 'mean',
                                'high_erosion_km2': ['mean', 'max', 'min', 'std']
                            }).round(2)
                            yearly_summary.columns = ['Faible_moy', 'Modérée_moy', 'Forte_moy', 'Forte_max', 'Forte_min', 'Forte_std']
                            csv_summary = yearly_summary.to_csv(encoding='utf-8')
                            
                            st.download_button(
                                label="📊 Résumé annuel (CSV)",
                                data=csv_summary,
                                file_name=f"erosion_summary_{years[0]}_{years[-1]}.csv",
                                mime="text/csv",
                                help="Statistiques annuelles résumées"
                            )
                        
                    else:
                        st.warning("⚠️ Aucune donnée valide d'érosion trouvée pour la période sélectionnée.")
                
                else:
                    st.error("❌ Échec de la génération des données d'érosion temporelle.")
        
        # ======================================================
        # OPTION 2: ANALYSE ÉTAT ACTUEL (ORIGINALE)
        # ======================================================
        else:
            status_text.text("⚠️ Analyse zones d'érosion (état actuel)...")
            
            st.header("⚠️ Zones à Risque d'Érosion - État Actuel")
            st.info("Cette analyse montre la situation actuelle du bassin versant.")
            
            with st.spinner("Calcul des zones d'érosion actuelles..."):
                try:
                    erosion_zones, stats = calculate_erosion_zones(watershed_geom)
                    
                    # Calcul des pourcentages
                    if stats['total'] > 0:
                        percent_low = (stats['low'] / stats['total']) * 100
                        percent_moderate = (stats['moderate'] / stats['total']) * 100
                        percent_high = (stats['high'] / stats['total']) * 100
                    else:
                        percent_low = percent_moderate = percent_high = 0
                    
                    # ======================================================
                    # AFFICHAGE DES MÉTRIQUES
                    # ======================================================
                    st.subheader("📊 Métriques de l'érosion actuelle")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Surface totale",
                            value=f"{stats['total']:.1f} km²",
                            delta="Bassin versant"
                        )
                    
                    with col2:
                        st.metric(
                            label="Érosion faible",
                            value=f"{stats['low']:.1f} km²",
                            delta=f"{percent_low:.1f}%",
                            delta_color="normal"
                        )
                    
                    with col3:
                        st.metric(
                            label="Érosion modérée",
                            value=f"{stats['moderate']:.1f} km²",
                            delta=f"{percent_moderate:.1f}%",
                            delta_color="off"
                        )
                    
                    with col4:
                        st.metric(
                            label="Érosion forte",
                            value=f"{stats['high']:.1f} km²",
                            delta=f"{percent_high:.1f}%",
                            delta_color="inverse"
                        )
                    
                    # ======================================================
                    # VISUALISATION
                    # ======================================================
                    st.subheader("📈 Visualisation")
                    
                    col_viz1, col_viz2 = st.columns([2, 1])
                    
                    with col_viz1:
                        # Graphique circulaire
                        fig_pie, ax = plt.subplots(figsize=(6, 6))
                        sizes = [stats['low'], stats['moderate'], stats['high']]
                        colors = ['#4CAF50', '#FFC107', '#F44336']  # vert, jaune, rouge
                        labels = ['Faible', 'Modérée', 'Forte']
                        
                        wedges, texts, autotexts = ax.pie(
                            sizes,
                            labels=labels,
                            colors=colors,
                            autopct='%1.1f%%',
                            startangle=90,
                            explode=(0.05, 0.05, 0.1)
                        )
                        
                        # Améliorer le texte
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            autotext.set_fontsize(10)
                        
                        for text in texts:
                            text.set_fontsize(11)
                            text.set_fontweight('bold')
                        
                        ax.axis('equal')
                        ax.set_title('Répartition des zones d\'érosion', fontsize=14, fontweight='bold')
                        
                        st.pyplot(fig_pie)
                        plt.close(fig_pie)
                    
                    with col_viz2:
                        # Légende et informations
                        st.markdown("""
                        **🎯 Légende :**
                        
                        **🟢 Faible érosion**
                        - Pente < 5°
                        - Bonne couverture végétale
                        - Risque minimal
                        
                        **🟡 Érosion modérée**
                        - Pente 5-15°
                        - Couverture végétale moyenne
                        - Surveillance requise
                        
                        **🔴 Forte érosion**
                        - Pente ≥ 16°
                        - Faible couverture végétale
                        - Intervention urgente
                        """)
                    
                    # ======================================================
                    # CARTE INTERACTIVE
                    # ======================================================
                    st.subheader("🗺️ Carte des zones d'érosion")
                    
                    with st.spinner("Génération de la carte..."):
                        try:
                            m_erosion = create_erosion_map(watershed_gdf, erosion_zones)
                            st_folium(
                                m_erosion,
                                width=800,
                                height=500,
                                returned_objects=[],
                                key=f"erosion_map_{datetime.now().timestamp()}"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de la génération de la carte: {str(e)[:100]}")
                            st.info("Affichage d'une carte simplifiée...")
                            st.map(watershed_gdf)
                    
                    # ======================================================
                    # RECOMMANDATIONS
                    # ======================================================
                    st.subheader("💡 Recommandations")
                    
                    if percent_high > 20:
                        st.error("""
                        **🔴 ALERTE : FORT RISQUE D'ÉROSION**
                        
                        Plus de 20% de la surface présente une érosion forte.
                        
                        **Actions prioritaires :**
                        1. **Replanter immédiatement** les zones fortement érodées
                        2. **Construire des terrasses** anti-érosives
                        3. **Interdire le surpâturage** dans ces zones
                        4. **Surveiller après chaque pluie** importante
                        5. **Consulter un expert** en gestion des sols
                        """)
                    elif percent_high > 10:
                        st.warning("""
                        **⚠️ RISQUE MODÉRÉ À ÉLEVÉ**
                        
                        Entre 10% et 20% de la surface présente une érosion forte.
                        
                        **Actions recommandées :**
                        1. **Stabiliser** les zones à risque avec végétation
                        2. **Planter des haies** brise-vent et anti-érosion
                        3. **Pratiquer l'agriculture** de conservation
                        4. **Suivre mensuellement** l'évolution
                        5. **Préparer un plan** d'intervention
                        """)
                    else:
                        st.success("""
                        **✅ SITUATION SATISFAISANTE**
                        
                        Moins de 10% de la surface présente une érosion forte.
                        
                        **Actions de maintien :**
                        1. **Continuer** les bonnes pratiques actuelles
                        2. **Surveiller** les zones à risque modéré
                        3. **Prévenir** la déforestation
                        4. **Maintenir** la biodiversité végétale
                        5. **Éduquer** sur la conservation des sols
                        """)
                    
                    # ======================================================
                    # TÉLÉCHARGEMENT
                    # ======================================================
                    st.subheader("💾 Export des résultats")
                    
                    # Créer un DataFrame pour l'export
                    erosion_summary = pd.DataFrame({
                        'Type_d_erosion': ['Faible', 'Modérée', 'Forte', 'Total'],
                        'Surface_km2': [stats['low'], stats['moderate'], stats['high'], stats['total']],
                        'Pourcentage': [percent_low, percent_moderate, percent_high, 100]
                    })
                    
                    csv_data = erosion_summary.to_csv(index=False, encoding='utf-8')
                    
                    st.download_button(
                        label="📥 Télécharger le rapport d'érosion",
                        data=csv_data,
                        file_name=f"erosion_etat_actuel_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        help="Contient le bilan complet de l'érosion actuelle"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse de l'érosion: {str(e)[:150]}")
                    st.info("""
                    **Solutions possibles :**
                    1. Vérifiez votre connexion Internet
                    2. Réessayez dans quelques minutes
                    3. Réduisez la taille de la zone d'étude
                    4. Contactez le support technique
                    """)
    # ==========================================================
    # ANALYSE LAC ITASY
    # ==========================================================
    if analyze_lake:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Analyse surface du lac...")

        st.header("🌊 Surveillance Lac Itasy")

        lake_polygon = Polygon(LAKE_ITASY_COORDS)
        lake_geometry = ee.Geometry.Polygon(LAKE_ITASY_COORDS)
        aoi_lake = lake_geometry.bounds()

        lake_rows = []
        months_to_process = get_months_to_process(years, datetime.now().year, datetime.now().month)
        total_months = len(months_to_process)

        for i, (year, month) in enumerate(months_to_process, 1):
            progress = (current_analysis - 1 + (i / total_months)) / total_analyses
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Traitement lac {year}-{month:02d}...")

            metrics = calculate_lake_metrics(year, month, aoi_lake)
            if metrics['status'] == 'success':
                lake_rows.append({
                    'year': year, 'month': month,
                    'surface_km2': round(metrics['surface_km2'], 3),
                    'mean_mndwi': round(metrics['mean_mndwi'], 3),
                    'water_percentage': round(metrics['water_percentage'], 1)
                })

        if lake_rows:
            df_lake = pd.DataFrame(lake_rows)
            fig_lake = create_lake_dashboard(df_lake)
            st.plotly_chart(fig_lake, use_container_width=True)

            # Analyse de tendance
            st.subheader("📉 Analyse de tendance")
            slope, intercept, annual_change, percent_change, r_value, p_value, std_err = analyze_lake_surface_trend(df_lake)
            if slope is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Tendance mensuelle", f"{slope:.4f} km²/mois")
                with col2: st.metric("Changement annuel", f"{annual_change:.2f} km²/an")
                with col3: st.metric("Changement total", f"{percent_change:.1f}%")
                with col4: st.metric("Corrélation (R²)", f"{r_value**2:.3f}")

            csv_data = df_lake.to_csv(index=False)
            st.download_button("📥 Télécharger données lac", csv_data,
                               f"lake_itasy_{years[0]}_{years[-1]}.csv", "text/csv")

    # ==========================================================
    # ANALYSE LAVAKAS (avec carte)
    # ==========================================================
    if analyze_lavakas:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Détection lavakas...")

        st.header("🕳️ Détection lavakas")

        years_for_lavakas = years[-2:] if len(years) >= 2 else years
        lavaka_rows = []
        months_to_process = get_months_to_process(years_for_lavakas, datetime.now().year, datetime.now().month)
        total_months = min(len(months_to_process), 24)

        for i, (year, month) in enumerate(months_to_process[:24], 1):
            progress = (current_analysis - 1 + (i / total_months)) / total_analyses
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Détection lavakas {year}-{month:02d}...")

            lavaka_mask, lavaka_score, area_km2, num_lavakas = detect_lavakas(year, month, watershed_geom, aoi)
            lavaka_rows.append({'year': year, 'month': month, 'area_km2': area_km2, 'num_lavakas': num_lavakas})

        df_lavakas = pd.DataFrame(lavaka_rows)

        if not df_lavakas.empty:
            fig_lavakas = plot_lavakas_timeseries(df_lavakas, years_for_lavakas)
            st.plotly_chart(fig_lavakas, use_container_width=True)
        
            # --- Carte des lavakas (mois avec détection positive) ---
            st.subheader("🗺️ Carte des lavakas")
            
            # Sélectionner les mois avec une surface strictement positive
            valid_months = df_lavakas[df_lavakas['area_km2'] > 0]
            
            if not valid_months.empty:
                # Prendre le mois le plus récent parmi ceux avec détection
                best_row = valid_months.iloc[-1]  # dernier mois avec surface > 0
                map_year = int(best_row['year'])
                map_month = int(best_row['month'])
                st.caption(f"Affichage de la carte pour {map_year}-{map_month:02d} (surface détectée : {best_row['area_km2']:.2f} km²)")
                
                with st.spinner(f"Génération de la carte pour {map_year}-{map_month:02d}..."):
                    try:
                        lavaka_mask, lavaka_score, _, _ = detect_lavakas(map_year, map_month, watershed_geom, aoi)
                        if lavaka_mask is not None:
                            m_lavakas = create_lavakas_map(watershed_gdf, lavaka_mask, lavaka_score)
                            st_folium(m_lavakas, width=800, height=500, returned_objects=[])
                            
                            # Légende textuelle sous la carte
                            st.markdown("""
                            <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:10px;">
                                <b>📖 Légende de la carte :</b><br>
                                <span style="background:#0000FF; display:inline-block; width:15px; height:15px; opacity:0.2;"></span> Lac Itasy<br>
                                <span style="background:#FFFF00; display:inline-block; width:15px; height:15px;"></span> Lavaka – probabilité faible<br>
                                <span style="background:#FFA500; display:inline-block; width:15px; height:15px;"></span> Lavaka – probabilité modérée<br>
                                <span style="background:#FF0000; display:inline-block; width:15px; height:15px;"></span> Lavaka – probabilité élevée<br>
                                <span style="background:#8B0000; display:inline-block; width:15px; height:15px;"></span> Lavaka confirmé
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"⚠️ Les données Sentinel‑2 pour {map_year}-{map_month:02d} ne permettent pas de générer la carte (nuages ou absence d'image).")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération de la carte : {e}")
            else:
                st.info("ℹ️ Aucune détection de lavakas sur l'ensemble de la période. Impossible de générer une carte.")                
            csv_lavakas = df_lavakas.to_csv(index=False)
            st.download_button("📥 Télécharger données lavakas", csv_lavakas,
                               f"lavakas_{years_for_lavakas[0]}_{years_for_lavakas[-1]}.csv", "text/csv")

    # ==========================================================
    # ANALYSE MÉTÉOROLOGIQUE (avec cartes température/précipitations)
    # ==========================================================
    if analyze_meteo:
        current_analysis += 1
        progress_bar.progress(current_analysis / total_analyses)
        status_text.text("Récupération des données météo...")

        st.header("🌤️ Analyse Météorologique")

        centroid_gdf = watershed_gdf.to_crs(epsg=4326)
        centroid_lat = centroid_gdf.geometry.centroid.y.mean()
        centroid_lon = centroid_gdf.geometry.centroid.x.mean()

        if meteo_params and meteo_params.get('use_custom_range', False):
            meteo_start_year = meteo_params['start_year']
            meteo_end_year = meteo_params['end_year']
        else:
            meteo_start_year = years[0]
            meteo_end_year = years[-1]

        start_date = date(meteo_start_year, 1, 1)
        end_date = date.today() if meteo_end_year == datetime.now().year else date(meteo_end_year, 12, 31)

        with st.spinner("Téléchargement des données historiques..."):
            meteo_data = get_openmeteo_historical(centroid_lat, centroid_lon,
                                                  start_date.strftime("%Y-%m-%d"),
                                                  end_date.strftime("%Y-%m-%d"))

        with st.spinner("Prévisions à 7 jours..."):
            forecast_data = get_openmeteo_forecast(centroid_lat, centroid_lon)

        if meteo_data:
            df_meteo = process_meteo_data(meteo_data)
            if df_meteo is not None:
                df_meteo_filtered = df_meteo[df_meteo['year'].between(meteo_start_year, meteo_end_year)]
                if not df_meteo_filtered.empty:
                    meteo_stats = calculate_meteo_statistics(df_meteo_filtered)
                    trends = analyze_meteo_trends(meteo_stats)

                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("🌡️ Température moyenne", f"{df_meteo_filtered['temp_mean'].mean():.1f}°C")
                    with col2: st.metric("🌧️ Précipitations totales", f"{df_meteo_filtered['precipitation'].sum():.0f} mm")
                    with col3: st.metric("📅 Période analysée", f"{len(df_meteo_filtered['year'].unique())} an(s)")

                    meteo_dashboard = create_meteo_dashboard(meteo_stats, trends, forecast_data)
                    if meteo_dashboard:
                        st.plotly_chart(meteo_dashboard, use_container_width=True)

                    csv_data = df_meteo_filtered.to_csv(index=False, encoding='utf-8')
                    st.download_button("📥 Télécharger les données météo (CSV)", csv_data,
                                       f"donnees_meteo_lac_itasy_{meteo_start_year}_{meteo_end_year}.csv", "text/csv")

        # --- CARTES CLIMATIQUES SUPPLÉMENTAIRES (selon les options de la sidebar) ---
        # --- CARTES CLIMATIQUES SUPPLÉMENTAIRES ---
        if st.session_state.get('show_temp_map', False) and st.session_state.get('map_year'):
            st.subheader(f"🌡️ Carte de température moyenne annuelle {st.session_state.map_year}")
            with st.spinner("Génération de la carte température..."):
                m_temp = create_temperature_map(watershed_geom, st.session_state.map_year, watershed_gdf)
                st_folium(m_temp, width=800, height=500, returned_objects=[])
                st.markdown("""
                <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:10px;">
                    <b>📖 Légende :</b><br>
                    <span style="background:blue; display:inline-block; width:15px; height:15px;"></span> 10°C<br>
                    <span style="background:lightblue; display:inline-block; width:15px; height:15px;"></span> ~15°C<br>
                    <span style="background:cyan; display:inline-block; width:15px; height:15px;"></span> ~20°C<br>
                    <span style="background:green; display:inline-block; width:15px; height:15px;"></span> ~22°C<br>
                    <span style="background:yellow; display:inline-block; width:15px; height:15px;"></span> ~25°C<br>
                    <span style="background:orange; display:inline-block; width:15px; height:15px;"></span> ~27°C<br>
                    <span style="background:red; display:inline-block; width:15px; height:15px;"></span> 30°C<br>
                    <i>Température de l'air à 2m (moyenne annuelle)</i>
                </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.get('show_precip_map', False) and st.session_state.get('map_year'):
            st.subheader(f"🌧️ Carte des précipitations cumulées annuelles {st.session_state.map_year}")
            with st.spinner("Génération de la carte précipitations..."):
                m_precip = create_precip_map(watershed_geom, st.session_state.map_year, watershed_gdf)
                st_folium(m_precip, width=800, height=500, returned_objects=[])
                st.markdown("""
                <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:10px;">
                    <b>📖 Légende :</b><br>
                    <span style="background:white; display:inline-block; width:15px; height:15px;"></span> 0 mm<br>
                    <span style="background:lightblue; display:inline-block; width:15px; height:15px;"></span> ~500 mm<br>
                    <span style="background:blue; display:inline-block; width:15px; height:15px;"></span> ~1000 mm<br>
                    <span style="background:darkblue; display:inline-block; width:15px; height:15px;"></span> 1500+ mm<br>
                    <i>Précipitations cumulées annuelles (CHIRPS)</i>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================================
    # TABLEAU DE BORD SYNTHÉTIQUE
    # ==========================================================
    analyses_count = 0
    if analyze_ndvi and df_ndvi is not None and not df_ndvi.empty:
        analyses_count += 1
    if analyze_sediment and df_sediment is not None and not df_sediment.empty:
        analyses_count += 1
    if analyze_erosion and stats is not None:
        analyses_count += 1
    if analyze_lake and df_lake is not None and not df_lake.empty:
        analyses_count += 1
    if analyze_lavakas and df_lavakas is not None and not df_lavakas.empty:
        analyses_count += 1

    if analyses_count >= 2:
        create_comprehensive_dashboard(
            df_ndvi=df_ndvi,
            df_sediment=df_sediment,
            stats=stats,
            df_lake=df_lake,
            df_lavakas=df_lavakas,
            years=years,
            analyze_ndvi=analyze_ndvi,
            analyze_sediment=analyze_sediment,
            analyze_erosion=analyze_erosion,
            analyze_lake=analyze_lake,
            analyze_lavakas=analyze_lavakas
        )
    else:
        st.info("ℹ️ Tableau de bord synthétique : besoin d’au moins deux analyses avec données valides.")

    progress_bar.progress(1.0)
    status_text.text("✅ Analyse terminée")
    st.balloons()
    st.success(f"✅ Analyse terminée pour la période {years[0]}-{years[-1]}")
    
# ==========================================================
# TABLEAU DE BORD SYNTHETIQUE
# ==========================================================
def create_comprehensive_dashboard(df_ndvi, df_sediment, stats, df_lake, df_lavakas, years,
                                   analyze_ndvi, analyze_sediment, analyze_erosion,
                                   analyze_lake, analyze_lavakas):
    """Tableau de bord synthétique avec les données des analyses sélectionnées."""
    
    st.header("📊 Tableau de bord synthétique - Vue d'ensemble")
    
    # Liste pour stocker les métriques (utilisée plus tard)
    metrics_to_show = []
    
    # 1. INDICATEURS CLÉS
    st.subheader("📈 Indicateurs clés")
    cols = st.columns(5)
    
    with cols[0]:
        if analyze_ndvi and df_ndvi is not None and not df_ndvi.empty:
            avg_ndvi = df_ndvi['surface_km2'].mean()
            # Tendance annuelle (si plusieurs années)
            yearly_ndvi = df_ndvi.groupby('year')['surface_km2'].mean()
            if len(yearly_ndvi) > 1:
                trend = (yearly_ndvi.iloc[-1] - yearly_ndvi.iloc[0]) / yearly_ndvi.iloc[0] * 100
            else:
                trend = 0
            st.metric("Surface végétale moyenne", f"{avg_ndvi:.1f} km²", f"{trend:+.1f}%",
                      delta_color="normal" if trend >= 0 else "inverse")
            metrics_to_show.append(("NDVI", avg_ndvi, trend))
    
    with cols[1]:
        if analyze_sediment and df_sediment is not None and not df_sediment.empty:
            avg_sed = df_sediment['sediment_index'].mean()
            yearly_sed = df_sediment.groupby('year')['sediment_index'].mean()
            if len(yearly_sed) > 1:
                trend = (yearly_sed.iloc[-1] - yearly_sed.iloc[0]) / yearly_sed.iloc[0] * 100
            else:
                trend = 0
            st.metric("Indice sédimentaire moyen", f"{avg_sed:.3f}", f"{trend:+.1f}%",
                      delta_color="inverse" if trend > 0 else "normal")
            metrics_to_show.append(("Sédiments", avg_sed, trend))
    
    with cols[2]:
        if analyze_erosion and stats is not None:
            high_pct = (stats['high'] / stats['total'] * 100) if stats['total'] > 0 else 0
            st.metric("Érosion forte", f"{high_pct:.1f}%", f"{stats['high']:.1f} km²",
                      delta_color="inverse")
            metrics_to_show.append(("Érosion forte", high_pct, None))
    
    with cols[3]:
        if analyze_lake and df_lake is not None and not df_lake.empty:
            current = df_lake['surface_km2'].iloc[-1]
            yearly_lake = df_lake.groupby('year')['surface_km2'].mean()
            if len(yearly_lake) > 1:
                trend = (yearly_lake.iloc[-1] - yearly_lake.iloc[0]) / yearly_lake.iloc[0] * 100
            else:
                trend = 0
            status = "🔴" if current < 30 else "🟡" if current < 33 else "🟢"
            st.metric(f"Surface lac {status}", f"{current:.1f} km²", f"{trend:+.1f}%",
                      delta_color="normal" if trend >= 0 else "inverse")
            metrics_to_show.append(("Surface lac", current, trend))
    
    with cols[4]:
        if analyze_lavakas and df_lavakas is not None and not df_lavakas.empty:
            avg_lav = df_lavakas['area_km2'].mean()
            yearly_lav = df_lavakas.groupby('year')['area_km2'].mean()
            if len(yearly_lav) > 1:
                trend = (yearly_lav.iloc[-1] - yearly_lav.iloc[0]) / yearly_lav.iloc[0] * 100
            else:
                trend = 0
            st.metric("Surface lavakas moyenne", f"{avg_lav:.2f} km²", f"{trend:+.1f}%",
                      delta_color="inverse" if trend > 0 else "normal")
            metrics_to_show.append(("Lavakas", avg_lav, trend))

    # 2. VISUALISATIONS INTÉGRÉES (subplots)
    st.subheader("📊 Visualisations synthétiques")
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Évolution NDVI vs Sédiments',
            'Surface du Lac Itasy',
            'Zones d\'érosion',
            'Activité des lavakas',
            'Corrélations multi-variables',
            'Indicateurs synthétiques'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "polar"}]
        ]
    )
    
    # Graphique 1 : NDVI vs Sédiments (double axe)
    if analyze_ndvi and analyze_sediment and df_ndvi is not None and df_sediment is not None:
        if not df_ndvi.empty and not df_sediment.empty:
            merged = pd.merge(df_ndvi, df_sediment, on=['year', 'month'], suffixes=('_ndvi', '_sed'))
            merged['date'] = pd.to_datetime(merged['year'].astype(str) + '-' + merged['month'].astype(str) + '-01')
            fig.add_trace(go.Scatter(x=merged['date'], y=merged['surface_km2_ndvi'], mode='lines',
                                     name='Surface végétale (km²)', line=dict(color='green', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=merged['date'], y=merged['sediment_index'], mode='lines',
                                     name='Indice sédimentaire', line=dict(color='brown', width=2, dash='dot'),
                                     yaxis='y2'), row=1, col=1)
            fig.update_layout(
                yaxis=dict(title="Surface végétale (km²)", title_font=dict(color="green")),
                yaxis2=dict(title="Indice sédimentaire", title_font=dict(color="brown"),
                            overlaying="y", side="right")
            )
    
    # Graphique 2 : Surface du lac
    if analyze_lake and df_lake is not None and not df_lake.empty:
        df_lake['date'] = pd.to_datetime(df_lake['year'].astype(str) + '-' + df_lake['month'].astype(str) + '-01')
        fig.add_trace(go.Scatter(x=df_lake['date'], y=df_lake['surface_km2'], mode='lines+markers',
                                 name='Surface lac', line=dict(color='blue', width=3),
                                 fill='tozeroy', fillcolor='rgba(0,100,255,0.1)'), row=1, col=2)
        fig.add_hline(y=LAKE_REFERENCE_AREA_KM2, line_dash="dash", line_color="red",
                      annotation_text=f"Référence: {LAKE_REFERENCE_AREA_KM2} km²", row=1, col=2)
    
    # Graphique 3 : Zones d'érosion (barres)
    if analyze_erosion and stats is not None:
        erosion_df = pd.DataFrame({
            'Type': ['Faible', 'Modérée', 'Forte'],
            'Surface (km²)': [stats['low'], stats['moderate'], stats['high']],
            'Pourcentage': [stats['low']/stats['total']*100 if stats['total']>0 else 0,
                            stats['moderate']/stats['total']*100 if stats['total']>0 else 0,
                            stats['high']/stats['total']*100 if stats['total']>0 else 0],
            'Couleur': ['green', 'yellow', 'red']
        })
        fig.add_trace(go.Bar(x=erosion_df['Type'], y=erosion_df['Surface (km²)'],
                             marker_color=erosion_df['Couleur'],
                             text=erosion_df['Pourcentage'].apply(lambda x: f'{x:.1f}%'),
                             textposition='auto', name='Zones d\'érosion'), row=2, col=1)
    
    # Graphique 4 : Lavakas (surface + nombre)
    if analyze_lavakas and df_lavakas is not None and not df_lavakas.empty:
        df_lavakas['date'] = pd.to_datetime(df_lavakas['year'].astype(str) + '-' + df_lavakas['month'].astype(str) + '-01')
        fig.add_trace(go.Scatter(x=df_lavakas['date'], y=df_lavakas['area_km2'], mode='lines+markers',
                                 name='Surface lavakas', line=dict(color='sienna', width=2)), row=2, col=2)
        fig.add_trace(go.Bar(x=df_lavakas['date'], y=df_lavakas['num_lavakas'],
                             name='Nombre lavakas', marker_color='lightcoral', opacity=0.5,
                             yaxis='y2'), row=2, col=2)
        fig.update_layout(
            yaxis3=dict(title="Surface (km²)", title_font=dict(color="sienna")),
            yaxis4=dict(title="Nombre", title_font=dict(color="lightcoral"),
                        overlaying="y3", side="right")
        )
    
    # Graphique 5 : Corrélation NDVI vs Surface lac
    if analyze_ndvi and analyze_lake and df_ndvi is not None and df_lake is not None:
        if not df_ndvi.empty and not df_lake.empty:
            merged = pd.merge(df_ndvi, df_lake, on=['year', 'month'], suffixes=('_ndvi', '_lake'))
            if not merged.empty:
                fig.add_trace(go.Scatter(x=merged['surface_km2_ndvi'], y=merged['surface_km2_lake'],
                                         mode='markers', name='NDVI vs Lac',
                                         marker=dict(size=10, color=merged['month'], colorscale='Viridis',
                                                     showscale=True, colorbar=dict(title="Mois")),
                                         text=[f"Mois {m}" for m in merged['month']]), row=3, col=1)
                if len(merged) > 1:
                    from scipy.stats import linregress
                    slope, intercept, r, _, _ = linregress(merged['surface_km2_ndvi'], merged['surface_km2_lake'])
                    x_range = [merged['surface_km2_ndvi'].min(), merged['surface_km2_ndvi'].max()]
                    fig.add_trace(go.Scatter(x=x_range, y=[slope*x+intercept for x in x_range],
                                             mode='lines', name=f'Régression (R²={r**2:.3f})',
                                             line=dict(color='red', width=2, dash='dash')), row=3, col=1)
    
    # Graphique 6 : Indicateurs synthétiques (radar ou barres)
        # Graphique 6 : Indicateurs synthétiques (radar chart)
    if metrics_to_show:
        indicators = pd.DataFrame(metrics_to_show, columns=['Indicateur', 'Valeur', 'Tendance'])
        # Toujours un radar, même pour 1 ou 2 indicateurs
        fig.add_trace(
            go.Scatterpolar(
                r=indicators['Valeur'].values,
                theta=indicators['Indicateur'].values,
                fill='toself',
                name='Indicateurs',
                line=dict(color='blue'),
                opacity=0.6
            ),
            row=3, col=2
        )
        # Optionnel : améliorer l'affichage des angles si peu d'indicateurs
        if len(indicators) <= 2:
            fig.update_polars(angularaxis=dict(dtick=90))  # évite un angle trop serré
    
    # Mise en page finale
    fig.update_layout(height=1200, showlegend=True, template='plotly_white',
                      title_text="Tableau de bord synthétique - Bassin versant Lac Itasy",
                      title_font_size=20)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Type d'érosion", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="Surface végétale (km²)", row=3, col=1)
    fig.update_yaxes(title_text="Surface (km²)", row=1, col=1)
    fig.update_yaxes(title_text="Surface (km²)", row=1, col=2)
    fig.update_yaxes(title_text="Surface (km²)", row=2, col=1)
    fig.update_yaxes(title_text="Surface (km²)", row=2, col=2)
    fig.update_yaxes(title_text="Surface lac (km²)", row=3, col=1)
    fig.update_yaxes(title_text="Valeur", row=3, col=2)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. ANALYSE DES TENDANCES (régressions linéaires)
    st.subheader("📈 Analyse des tendances et corrélations")
    from scipy.stats import linregress
    trend_cols = st.columns(4)
    
    with trend_cols[0]:
        if analyze_ndvi and df_ndvi is not None and not df_ndvi.empty:
            yearly = df_ndvi.groupby('year')['surface_km2'].mean().reset_index()
            if len(yearly) > 1:
                slope, _, r, _, _ = linregress(yearly['year'], yearly['surface_km2'])
                st.metric("Tendance NDVI annuelle", f"{slope:.3f} km²/an", f"R²={r**2:.3f}",
                          delta_color="normal" if slope > 0 else "inverse")
    
    with trend_cols[1]:
        if analyze_lake and df_lake is not None and not df_lake.empty:
            yearly = df_lake.groupby('year')['surface_km2'].mean().reset_index()
            if len(yearly) > 1:
                slope, _, r, _, _ = linregress(yearly['year'], yearly['surface_km2'])
                st.metric("Tendance surface lac", f"{slope:.3f} km²/an", f"R²={r**2:.3f}",
                          delta_color="normal" if slope > 0 else "inverse")
    
    with trend_cols[2]:
        if analyze_sediment and df_sediment is not None and not df_sediment.empty:
            yearly = df_sediment.groupby('year')['sediment_index'].mean().reset_index()
            if len(yearly) > 1:
                slope, _, r, _, _ = linregress(yearly['year'], yearly['sediment_index'])
                st.metric("Tendance sédiments", f"{slope:.4f}/an", f"R²={r**2:.3f}",
                          delta_color="inverse" if slope > 0 else "normal")
    
    with trend_cols[3]:
        if analyze_ndvi and analyze_sediment and df_ndvi is not None and df_sediment is not None:
            if not df_ndvi.empty and not df_sediment.empty:
                merged = pd.merge(df_ndvi, df_sediment, on=['year', 'month'])
                if not merged.empty:
                    corr = merged['surface_km2'].corr(merged['sediment_index'])
                    status = "✅ Forte" if abs(corr) > 0.7 else "⚠️ Modérée" if abs(corr) > 0.3 else "ℹ️ Faible"
                    st.metric("Corrélation NDVI-Sédiments", f"{corr:.3f}", status,
                              help="Négative attendue : végétation ↘️ = sédiments ↗️")
    
    # 4. RECOMMANDATIONS SYNTHÉTIQUES
    st.subheader("💡 Recommandations synthétiques")
    recommendations = []
    
    if analyze_ndvi and df_ndvi is not None and not df_ndvi.empty:
        current = df_ndvi['surface_km2'].iloc[-1]
        avg = df_ndvi['surface_km2'].mean()
        if current < avg * 0.8:
            recommendations.append(("⚠️", "Couverture végétale en baisse", "Replanter les zones dégradées"))
        elif current > avg * 1.2:
            recommendations.append(("✅", "Bonne couverture végétale", "Maintenir les pratiques actuelles"))
    
    if analyze_erosion and stats is not None:
        if stats['high'] > stats['total'] * 0.15:
            recommendations.append(("🔴", "Risque érosion élevé", "Prioriser les mesures anti-érosives"))
        elif stats['high'] > stats['total'] * 0.05:
            recommendations.append(("🟡", "Risque érosion modéré", "Surveiller et planifier des interventions"))
    
    if analyze_lake and df_lake is not None and not df_lake.empty:
        current = df_lake['surface_km2'].iloc[-1]
        if current < LAKE_REFERENCE_AREA_KM2 * 0.9:
            recommendations.append(("💧", "Surface lac réduite", "Évaluer la consommation d'eau et l'ensablement"))
    
    if analyze_lavakas and df_lavakas is not None and not df_lavakas.empty:
        growth = df_lavakas['area_km2'].pct_change().mean() * 100
        if growth > 5:
            recommendations.append(("🕳️", "Lavakas en expansion", "Contrôler l'érosion ravinante"))
    
    if recommendations:
        for icon, title, desc in recommendations:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h4 style="margin: 0;">{icon} {title}</h4>
                <p style="margin: 5px 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ L'état général du bassin versant semble satisfaisant. Continuez les bonnes pratiques !")
    
    # 5. EXPORT DES DONNÉES SYNTHÉTIQUES
    st.subheader("💾 Export des données synthétiques")
    if st.button("📥 Générer rapport synthétique"):
        synth_data = {}
        if analyze_ndvi and df_ndvi is not None and not df_ndvi.empty:
            synth_data['NDVI_moyen_km2'] = [df_ndvi['surface_km2'].mean()]
            yearly = df_ndvi.groupby('year')['surface_km2'].mean()
            if len(yearly) > 1:
                synth_data['Tendance_NDVI_%'] = [(yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100]
        if analyze_sediment and df_sediment is not None and not df_sediment.empty:
            synth_data['Indice_sediment_moyen'] = [df_sediment['sediment_index'].mean()]
        if analyze_erosion and stats is not None:
            synth_data['Erosion_forte_km2'] = [stats['high']]
            synth_data['Erosion_forte_%'] = [(stats['high'] / stats['total'] * 100) if stats['total'] > 0 else 0]
        if analyze_lake and df_lake is not None and not df_lake.empty:
            synth_data['Surface_lac_moyenne'] = [df_lake['surface_km2'].mean()]
            synth_data['Surface_lac_actuelle'] = [df_lake['surface_km2'].iloc[-1]]
        if analyze_lavakas and df_lavakas is not None and not df_lavakas.empty:
            synth_data['Lavakas_moyen_km2'] = [df_lavakas['area_km2'].mean()]
        if synth_data:
            synth_df = pd.DataFrame(synth_data)
            st.download_button("📥 Télécharger le rapport synthétique",
                               synth_df.to_csv(index=False),
                               f"rapport_synthetique_bassin_{years[0]}_{years[-1]}.csv",
                               "text/csv")
        else:
            st.warning("Aucune donnée à exporter.")

def show_welcome_page():
    """Affiche la page d'accueil."""
    st.info("👈 Configurez les paramètres et cliquez sur 'Lancer l'analyse'")
    
    st.subheader("Fonctionnalités")
    features = [
        ("🌱 Analyse NDVI", "Suivi couverture végétale"),
        ("🏔️ Analyse sédimentaire", "Estimation érosion et sédimentation"),
        ("⚠️ Zones à risque", "Identification zones d'érosion"),
        ("🌊 Surveillance lac", "Suivi surface Lac Itasy"),
        ("🕳️ Détection lavakas", "Identification ravines d'érosion"),
        ("🌤️ Analyse météo", "Données et tendances météorologiques"),
        ("🗺️ Cartographie", "Visualisations interactives"),
        ("📊 Dashboard", "Analyses temporelles et statistiques")
    ]
    
    for icon, desc in features:
        st.markdown(f"- **{icon}**: {desc}")

# ==========================================================
# EXECUTION
# ==========================================================
if __name__ == "__main__":
    main()
