import streamlit as st
import ee
import json
import tempfile
import os
import folium
import folium.plugins as plugins
import urllib.request
from streamlit_folium import st_folium
from fpdf import FPDF
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geedim
import rasterio
from io import BytesIO

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & AUTH
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(layout="wide", page_title="Sentinel-2 NDVI/NDWI Explorer", page_icon="🌿")

try:
    json_key = None
    if 'json_data' in st.secrets:
        json_key = 'json_data'
    elif 'EE_SERVICE_ACCOUNT' in st.secrets:
        json_key = 'EE_SERVICE_ACCOUNT'

    if json_key:
        service_account_json = st.secrets[json_key]
        key_path = os.path.join(tempfile.gettempdir(), 'ee_service_account.json')
        with open(key_path, 'w') as f:
            f.write(service_account_json)
        sa_email = st.secrets.get(
            'service_account',
            json.load(open(key_path))['client_email']
        )
        credentials = ee.ServiceAccountCredentials(sa_email, key_file=key_path)
        ee.Initialize(credentials, project='spatialgeography')
    else:
        ee.Initialize(project='spatialgeography')
except Exception as e:
    st.error("🔑 **Earth Engine Authentication Failed**")
    st.write(f"**Error:** {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [data-testid="stSidebar"] { font-family: 'Inter', sans-serif; }
    h1 {
        font-weight: 800;
        background: linear-gradient(135deg, #059669, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .main { background-color: #f8fafc; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #064e3b 0%, #065f46 50%, #047857 100%); }
    [data-testid="stSidebar"] * { color: #d1fae5 !important; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .index-card {
        background: white; border-radius: 12px; padding: 16px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Sentinel-2 Vegetation & Water Index Explorer")
st.caption("NDVI • NDWI • Interactive Analysis • PDF Reports  —  Powered by Google Earth Engine")

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Analysis Settings")

# Date range
st.sidebar.subheader("📅 Date Range")
today = datetime.now().date()
default_start = today - timedelta(days=90)
date_start = st.sidebar.date_input("Start Date", value=default_start)
date_end = st.sidebar.date_input("End Date", value=today)

# Cloud filter
cloud_pct = st.sidebar.slider("Max Cloud Cover (%)", 0, 100, 20)

# Index selection
st.sidebar.subheader("📊 Indices to Compute")
calc_ndvi = st.sidebar.checkbox("NDVI (Vegetation)", value=True)
calc_ndwi = st.sidebar.checkbox("NDWI (Water)", value=True)
calc_evi = st.sidebar.checkbox("EVI (Enhanced Vegetation)", value=True)
calc_savi = st.sidebar.checkbox("SAVI (Soil-Adjusted Vegetation)", value=False)
calc_mndwi = st.sidebar.checkbox("MNDWI (Modified Water)", value=False)
calc_bsi = st.sidebar.checkbox("BSI (Bare Soil Index)", value=False)

st.sidebar.subheader("✏️ Drawing Tools")
st.sidebar.info("Draw on the map to define your study area. Stats will be computed for drawn regions.")
point_buffer_m = st.sidebar.slider("Point buffer (m)", 100, 10000, 1000)

# Report settings
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Report Settings")
report_title = st.sidebar.text_input("Report Title", "Sentinel-2 Index Analysis")
report_author = st.sidebar.text_input("Author Name", "User")


# ══════════════════════════════════════════════════════════════════════
# EARTH ENGINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def mask_s2_clouds(image):
    """Mask clouds in Sentinel-2 using QA60 band."""
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)


def get_s2_collection(geometry, start, end, cloud_max):
    """Get cloud-masked Sentinel-2 composite."""
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(str(start), str(end))
        .filterBounds(geometry)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_max))
        .map(mask_s2_clouds)
    )
    return collection


def compute_indices(image):
    """Compute all spectral indices from a Sentinel-2 image."""
    nir = image.select('B8')    # NIR
    red = image.select('B4')    # Red
    green = image.select('B3')  # Green
    blue = image.select('B2')   # Blue
    swir1 = image.select('B11') # SWIR 1
    swir2 = image.select('B12') # SWIR 2

    indices = {}

    # NDVI = (NIR - Red) / (NIR + Red)
    indices['NDVI'] = nir.subtract(red).divide(nir.add(red)).rename('NDVI')

    # NDWI = (Green - NIR) / (Green + NIR)
    indices['NDWI'] = green.subtract(nir).divide(green.add(nir)).rename('NDWI')

    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    indices['EVI'] = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).rename('EVI')

    # SAVI = 1.5 * (NIR - Red) / (NIR + Red + 0.5)
    indices['SAVI'] = nir.subtract(red).multiply(1.5).divide(
        nir.add(red).add(0.5)
    ).rename('SAVI')

    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    indices['MNDWI'] = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI')

    # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    indices['BSI'] = swir1.add(red).subtract(nir.add(blue)).divide(
        swir1.add(red).add(nir).add(blue)
    ).rename('BSI')

    return indices


@st.cache_data(ttl=600)
def compute_index_stats(_geom_info, index_image_id, index_name, start, end, cloud_max):
    """Compute statistics for a given index over a geometry."""
    geom = ee.Geometry(json.loads(_geom_info))
    collection = get_s2_collection(geom, start, end, cloud_max)
    count = collection.size().getInfo()

    if count == 0:
        return None, 0

    composite = collection.median()
    all_indices = compute_indices(composite)

    if index_name not in all_indices:
        return None, count

    index_img = all_indices[index_name]

    stats = index_img.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
            .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
            .combine(reducer2=ee.Reducer.percentile([10, 25, 50, 75, 90]), sharedInputs=True),
        geometry=geom, scale=10, maxPixels=1e9
    ).getInfo()

    result = {}
    for key, val in stats.items():
        short_key = key.replace(index_name + '_', '').replace(index_name, 'value')
        result[short_key] = round(val, 4) if val is not None else 0
    return result, count


@st.cache_data(ttl=600)
def compute_index_histogram(_geom_info, index_name, start, end, cloud_max, num_bins=50):
    """Compute histogram for an index."""
    geom = ee.Geometry(json.loads(_geom_info))
    collection = get_s2_collection(geom, start, end, cloud_max)
    if collection.size().getInfo() == 0:
        return [], []

    composite = collection.median()
    index_img = compute_indices(composite)[index_name]

    # Index-specific ranges
    ranges = {
        'NDVI': (-0.5, 1.0), 'NDWI': (-1.0, 1.0), 'EVI': (-0.5, 1.0),
        'SAVI': (-0.5, 1.0), 'MNDWI': (-1.0, 1.0), 'BSI': (-1.0, 1.0)
    }
    lo, hi = ranges.get(index_name, (-1, 1))

    hist = index_img.reduceRegion(
        reducer=ee.Reducer.fixedHistogram(lo, hi, num_bins),
        geometry=geom, scale=10, maxPixels=1e9
    ).getInfo()

    buckets = hist.get(index_name, [])
    bins = [b[0] for b in buckets]
    counts = [b[1] for b in buckets]
    return bins, counts


@st.cache_data(ttl=600)
def compute_time_series(_geom_info, index_name, start, end, cloud_max):
    """Compute monthly time series for an index."""
    geom = ee.Geometry(json.loads(_geom_info))

    def monthly_mean(year, month):
        m_start = '%04d-%02d-01' % (year, month)
        if month == 12:
            m_end = '%04d-01-01' % (year + 1)
        else:
            m_end = '%04d-%02d-01' % (year, month + 1)
        col = get_s2_collection(geom, m_start, m_end, cloud_max)
        composite = col.median()
        indices = compute_indices(composite)
        if index_name not in indices:
            return None
        val = indices[index_name].reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom, scale=10, maxPixels=1e9
        ).getInfo().get(index_name)
        return val

    # Generate monthly dates
    from datetime import date
    s = date.fromisoformat(str(start))
    e = date.fromisoformat(str(end))
    dates = []
    values = []
    current = s.replace(day=1)
    while current <= e:
        dates.append(current.strftime('%Y-%m'))
        current = (current + timedelta(days=32)).replace(day=1)

    # Batch EE calls
    for d in dates:
        y, m = int(d[:4]), int(d[5:7])
        try:
            v = monthly_mean(y, m)
            values.append(round(v, 4) if v is not None else None)
        except Exception:
            values.append(None)

    return dates, values


def get_index_map_id(geometry, index_name, start, end, cloud_max):
    """Get EE map tile URL for an index layer."""
    collection = get_s2_collection(geometry, start, end, cloud_max)
    composite = collection.median()
    index_img = compute_indices(composite)[index_name]

    palettes = {
        'NDVI': {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'NDWI': {'min': -0.5, 'max': 0.5, 'palette': ['d73027', 'fc8d59', 'fee08b', 'c6dbef', '6baed6', '2171b5']},
        'EVI':  {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'SAVI': {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'MNDWI': {'min': -0.5, 'max': 0.5, 'palette': ['d73027', 'fc8d59', 'fee08b', 'c6dbef', '6baed6', '2171b5']},
        'BSI':  {'min': -0.3, 'max': 0.3, 'palette': ['1a9850', '91cf60', 'fee08b', 'fc8d59', 'd73027', '8b0000']},
    }
    vis = palettes.get(index_name, {'min': -1, 'max': 1, 'palette': ['red', 'white', 'green']})
    map_id = index_img.getMapId(vis)
    return map_id, vis


def get_index_thumbnail(geometry, index_name, start, end, cloud_max):
    """Get a static thumbnail PNG for an index."""
    collection = get_s2_collection(geometry, start, end, cloud_max)
    composite = collection.median()
    index_img = compute_indices(composite)[index_name]

    palettes = {
        'NDVI': {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'NDWI': {'min': -0.5, 'max': 0.5, 'palette': ['d73027', 'fc8d59', 'fee08b', 'c6dbef', '6baed6', '2171b5']},
        'EVI':  {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'SAVI': {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'MNDWI': {'min': -0.5, 'max': 0.5, 'palette': ['d73027', 'fc8d59', 'fee08b', 'c6dbef', '6baed6', '2171b5']},
        'BSI':  {'min': -0.3, 'max': 0.3, 'palette': ['1a9850', '91cf60', 'fee08b', 'fc8d59', 'd73027', '8b0000']},
    }
    vis = palettes.get(index_name, {'min': -1, 'max': 1, 'palette': ['red', 'white', 'green']})

    thumb_url = index_img.getThumbURL({
        'min': vis['min'], 'max': vis['max'], 'palette': vis['palette'],
        'region': geometry, 'dimensions': 800, 'format': 'png'
    })
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    urllib.request.urlretrieve(thumb_url, tmp.name)
    return tmp.name


def create_publication_map(geometry, index_name, start, end, cloud_max, title=None):
    """Create a publication-quality map using Cartopy."""
    collection = get_s2_collection(geometry, start, end, cloud_max)
    if collection.size().getInfo() == 0:
        return None

    composite = collection.median()
    index_img = compute_indices(composite)[index_name]

    # Get data as a numpy array for plotting
    # We use getThumbURL to get a visualization that we can then geo-reference with cartopy
    # or better, use geedim to download a small chunk and plot
    region = geometry.bounds().buffer(0.01).getInfo() # slightly larger
    
    palettes = {
        'NDVI': {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'NDWI': {'min': -0.5, 'max': 0.5, 'palette': ['d73027', 'fc8d59', 'fee08b', 'c6dbef', '6baed6', '2171b5']},
        'EVI':  {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'SAVI': {'min': -0.2, 'max': 0.8, 'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '91cf60', '1a9850']},
        'MNDWI': {'min': -0.5, 'max': 0.5, 'palette': ['d73027', 'fc8d59', 'fee08b', 'c6dbef', '6baed6', '2171b5']},
        'BSI':  {'min': -0.3, 'max': 0.3, 'palette': ['1a9850', '91cf60', 'fee08b', 'fc8d59', 'd73027', '8b0000']},
    }
    vis = palettes.get(index_name, {'min': -1, 'max': 1, 'palette': ['red', 'white', 'green']})
    
    # Use geedim to get a local image for plotting
    try:
        gd_image = geedim.MaskedImage(ee.Image(index_img))
        # Download a low-res version for the plot
        tmp_tif_path = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
        gd_image.download(tmp_tif_path, region=region, scale=50, crs='EPSG:4326', overwrite=True)
            
        with rasterio.open(tmp_tif_path) as src:
            data = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Plot data
        # Use colormap corresponding to GEE palette if possible, or mapping
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('gee_p', ['#' + c for c in vis['palette']])
        
        im = ax.imshow(data, extent=extent, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vis['min'], vmax=vis['max'], origin='upper')
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, aspect=30)
        cbar.set_label(index_name)
        
        # Title
        if title:
            plt.title(title, size=14, pad=20)
        else:
            plt.title(f"Publication Quality {index_name} Map", size=14, pad=20)
            
        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Cleanup
        if os.path.exists(tmp_tif_path):
            os.remove(tmp_tif_path)
            
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error creating publication map: {e}")
        return None


def download_gee_tif(geometry, index_name, start, end, cloud_max):
    """Download GeoTIFF using geedim, handles large areas with tiling."""
    collection = get_s2_collection(geometry, start, end, cloud_max)
    if collection.size().getInfo() == 0:
        return None

    composite = collection.median()
    index_img = compute_indices(composite)[index_name]
    
    try:
        gd_image = geedim.MaskedImage(ee.Image(index_img))
        region = geometry.getInfo()
        
        tmp_path = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
        
        # geedim handles tiling automatically if needed
        gd_image.download(tmp_path, region=region, scale=10, crs='EPSG:4326', overwrite=True)
        with open(tmp_path, 'rb') as f:
            data = f.read()
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return data
    except Exception as e:
        st.error(f"Error downloading GeoTIFF: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# CHART CREATION
# ══════════════════════════════════════════════════════════════════════

INDEX_COLORS = {
    'NDVI': '#1a9850', 'NDWI': '#2171b5', 'EVI': '#4daf4a',
    'SAVI': '#ff7f00', 'MNDWI': '#377eb8', 'BSI': '#a65628',
}

INDEX_DESCRIPTIONS = {
    'NDVI': 'Normalized Difference Vegetation Index - measures green vegetation density and health. Values: -1 to 1 (>0.3 = healthy vegetation).',
    'NDWI': 'Normalized Difference Water Index - detects surface water and moisture. Values: -1 to 1 (>0 = water presence).',
    'EVI': 'Enhanced Vegetation Index - improved vegetation signal in high biomass areas with atmospheric correction. Values: -1 to 1.',
    'SAVI': 'Soil-Adjusted Vegetation Index - reduces soil brightness influence on vegetation detection. Values: -1 to 1.',
    'MNDWI': 'Modified NDWI - uses SWIR band for better urban water detection. Values: -1 to 1 (>0 = water).',
    'BSI': 'Bare Soil Index - highlights bare soil and non-vegetated areas. Values: -1 to 1 (>0 = bare soil).',
}


def create_index_histogram(bins, counts, index_name):
    color = INDEX_COLORS.get(index_name, '#333')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bins, y=counts, marker_color=color,
        marker_line_color='rgba(0,0,0,0.2)', marker_line_width=0.5, opacity=0.85,
        hovertemplate='%s: %%{x:.3f}<br>Pixels: %%{y:,.0f}<extra></extra>' % index_name
    ))
    fig.update_layout(
        title=dict(text='%s Distribution' % index_name, font=dict(size=15, family='Inter')),
        xaxis_title=index_name, yaxis_title='Pixel Count',
        template='plotly_white', height=350,
        margin=dict(l=50, r=20, t=45, b=45), font=dict(family='Inter')
    )
    return fig


def create_time_series_chart(dates, values, index_name):
    color = INDEX_COLORS.get(index_name, '#333')
    # Filter out None values
    valid_dates = [d for d, v in zip(dates, values) if v is not None]
    valid_values = [v for v in values if v is not None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid_dates, y=valid_values, mode='lines+markers',
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color, line=dict(width=1, color='white')),
        fill='tozeroy', fillcolor=color.replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in color else None,
        hovertemplate='%s: %%{y:.4f}<br>Date: %%{x}<extra></extra>' % index_name
    ))
    fig.update_layout(
        title=dict(text='%s Time Series (Monthly Mean)' % index_name, font=dict(size=15, family='Inter')),
        xaxis_title='Month', yaxis_title=index_name,
        template='plotly_white', height=350,
        margin=dict(l=50, r=20, t=45, b=45), font=dict(family='Inter')
    )
    return fig


def create_comparison_bar(all_stats):
    """Create a grouped bar comparing all selected indices."""
    fig = go.Figure()
    names = list(all_stats.keys())
    means = [all_stats[n].get('mean', 0) for n in names]
    colors = [INDEX_COLORS.get(n, '#333') for n in names]

    fig.add_trace(go.Bar(
        x=names, y=means, marker_color=colors,
        text=['%.4f' % m for m in means], textposition='auto',
        hovertemplate='%{x}: %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Index Comparison (Mean Values)', font=dict(size=15, family='Inter')),
        yaxis_title='Mean Value', template='plotly_white', height=350,
        margin=dict(l=50, r=20, t=45, b=45), font=dict(family='Inter')
    )
    return fig


def create_land_cover_pie(ndvi_bins, ndvi_counts):
    """Classify NDVI into land cover categories."""
    categories = ['Water/Cloud', 'Bare Soil/Rock', 'Sparse Vegetation', 'Moderate Vegetation', 'Dense Vegetation']
    cat_colors = ['#2171b5', '#d2b48c', '#ffffb2', '#78c679', '#006837']
    thresholds = [-0.5, 0.0, 0.15, 0.3, 0.5, 1.0]

    cat_counts = [0.0] * 5
    for b, c in zip(ndvi_bins, ndvi_counts):
        for i in range(5):
            if thresholds[i] <= b < thresholds[i + 1]:
                cat_counts[i] += c
                break

    total = sum(cat_counts)
    if total == 0:
        return None
    pcts = [round(c / total * 100, 1) for c in cat_counts]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=categories, values=pcts, marker_colors=cat_colors,
        hole=0.4, textinfo='percent+label', textfont_size=11,
        hovertemplate='%{label}<br>%{percent}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Land Cover Classification (from NDVI)', font=dict(size=15, family='Inter')),
        template='plotly_white', height=400,
        margin=dict(l=20, r=20, t=50, b=20), font=dict(family='Inter'),
        showlegend=True
    )
    return fig, categories, pcts


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS DISPLAY FUNCTION
# ══════════════════════════════════════════════════════════════════════

def display_index_analysis(geom_info, area_label, selected_indices):
    """Display full analysis for a given geometry."""

    # Compute stats for all selected indices
    all_stats = {}
    image_count = 0
    for idx_name in selected_indices:
        with st.spinner("Computing %s..." % idx_name):
            stats, cnt = compute_index_stats(
                geom_info, idx_name, idx_name,
                str(date_start), str(date_end), cloud_pct
            )
            if stats:
                all_stats[idx_name] = stats
            image_count = max(image_count, cnt)

    if not all_stats:
        st.warning("No Sentinel-2 images found for this area and date range. Try expanding the date range or increasing cloud cover tolerance.")
        return all_stats

    st.info("📡 **%d Sentinel-2 images** found for the selected period" % image_count)

    # Stats cards for each index
    for idx_name, stats in all_stats.items():
        st.markdown("#### %s — %s" % (idx_name, INDEX_DESCRIPTIONS.get(idx_name, '').split('.')[0]))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean", "%.4f" % stats.get('mean', 0))
        c2.metric("Min", "%.4f" % stats.get('min', 0))
        c3.metric("Max", "%.4f" % stats.get('max', 0))
        c4.metric("Std Dev", "%.4f" % stats.get('stdDev', 0))
        c5.metric("Median", "%.4f" % stats.get('p50', 0))

    # Create tabs for detailed analysis
    tab_names = ["📊 Histograms", "📈 Time Series", "🔄 Index Comparison"]
    if 'NDVI' in all_stats:
        tab_names.append("🗺️ Land Cover")
    tab_names.extend(["🗾 Publication Maps", "📥 Download Data"])

    tabs = st.tabs(tab_names)

    # Tab 1: Histograms
    with tabs[0]:
        for idx_name in all_stats:
            with st.spinner("Computing %s histogram..." % idx_name):
                bins, counts = compute_index_histogram(
                    geom_info, idx_name,
                    str(date_start), str(date_end), cloud_pct
                )
                if bins:
                    fig = create_index_histogram(bins, counts, idx_name)
                    st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Time Series
    with tabs[1]:
        st.caption("⏳ Time series computation may take a moment — it queries EE for each month.")
        for idx_name in all_stats:
            if st.button("Load %s Time Series" % idx_name, key="ts_%s_%s" % (idx_name, area_label)):
                with st.spinner("Computing monthly %s..." % idx_name):
                    dates, values = compute_time_series(
                        geom_info, idx_name,
                        str(date_start), str(date_end), cloud_pct
                    )
                    if dates:
                        fig = create_time_series_chart(dates, values, idx_name)
                        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Comparison
    with tabs[2]:
        if len(all_stats) >= 2:
            fig = create_comparison_bar(all_stats)
            st.plotly_chart(fig, use_container_width=True)

            # Percentile comparison table
            table = []
            for name, s in all_stats.items():
                table.append({
                    'Index': name,
                    'Mean': s.get('mean', 0), 'Min': s.get('min', 0),
                    'Max': s.get('max', 0), 'Std Dev': s.get('stdDev', 0),
                    'P10': s.get('p10', 0), 'P25': s.get('p25', 0),
                    'Median': s.get('p50', 0), 'P75': s.get('p75', 0),
                    'P90': s.get('p90', 0),
                })
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.info("Select at least 2 indices in the sidebar to see comparisons.")

    # Tab 4: Land cover (if NDVI)
    if 'NDVI' in all_stats and len(tabs) > 3:
        with tabs[3]:
            with st.spinner("Computing land cover classification..."):
                bins, counts = compute_index_histogram(
                    geom_info, 'NDVI',
                    str(date_start), str(date_end), cloud_pct
                )
                if bins:
                    result = create_land_cover_pie(bins, counts)
                    if result:
                        fig, cats, pcts = result
                        st.plotly_chart(fig, use_container_width=True)
                        lc_data = [{'Category': c, 'Percentage': '%.1f%%' % p} for c, p in zip(cats, pcts)]
                        st.dataframe(lc_data, use_container_width=True, hide_index=True)

    # New Tab: Publication Maps
    pub_tab_idx = 4 if 'NDVI' in all_stats else 3
    with tabs[pub_tab_idx]:
        st.markdown("### 🗾 Publication-Quality Cartographic Maps")
        st.info("These maps are generated with Cartopy and include Latitude/Longitude gridlines and scale bars for publication use.")
        
        for idx_name in all_stats:
            if st.button(f"Generate High-Quality {idx_name} Map", key=f"pub_map_{idx_name}_{area_label}"):
                with st.spinner(f"Generating {idx_name} map with Cartopy..."):
                    geom = ee.Geometry(json.loads(geom_info))
                    pub_map_bytes = create_publication_map(
                        geom, idx_name, str(date_start), str(date_end), cloud_pct,
                        title=f"Sentinel-2 {idx_name} Index Map - {area_label}"
                    )
                    if pub_map_bytes:
                        st.image(pub_map_bytes, caption=f"Publication Ready {idx_name} Map", use_container_width=True)
                        st.download_button(
                            label=f"📥 Download {idx_name} Map (PNG)",
                            data=pub_map_bytes,
                            file_name=f"{area_label}_{idx_name}_publication_map.png",
                            mime="image/png"
                        )

    # New Tab: Download Data
    dl_tab_idx = pub_tab_idx + 1
    with tabs[dl_tab_idx]:
        st.markdown("### 📥 Download Analysis Data")
        st.info("Download the full resolution (10m) GeoTIFF for your study area. Large areas are automatically tiled and merged.")
        
        for idx_name in all_stats:
            if st.button(f"Download {idx_name} GeoTIFF", key=f"dl_tif_{idx_name}_{area_label}"):
                with st.spinner(f"Preparing GeoTIFF for {idx_name}..."):
                    geom = ee.Geometry(json.loads(geom_info))
                    tif_data = download_gee_tif(geom, idx_name, str(date_start), str(date_end), cloud_pct)
                    if tif_data:
                        st.success(f"✅ {idx_name} GeoTIFF ready!")
                        st.download_button(
                            label=f"💾 Save {idx_name} TIF",
                            data=tif_data,
                            file_name=f"{area_label}_{idx_name}.tif",
                            mime="image/tiff"
                        )

    return all_stats


# ══════════════════════════════════════════════════════════════════════
# MAP & MAIN UI
# ══════════════════════════════════════════════════════════════════════

# Default location
DEFAULT_LAT, DEFAULT_LON = 20.30, 85.82
DEFAULT_ZOOM = 10

# Build selected indices list
selected_indices = []
if calc_ndvi: selected_indices.append('NDVI')
if calc_ndwi: selected_indices.append('NDWI')
if calc_evi: selected_indices.append('EVI')
if calc_savi: selected_indices.append('SAVI')
if calc_mndwi: selected_indices.append('MNDWI')
if calc_bsi: selected_indices.append('BSI')

if not selected_indices:
    st.warning("Please select at least one index from the sidebar.")
    st.stop()

# Create map
m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=DEFAULT_ZOOM)

# Try to add index layers for default view area
try:
    default_roi = ee.Geometry.Point(DEFAULT_LON, DEFAULT_LAT).buffer(5000)
    for idx_name in selected_indices[:2]:  # Only show first 2 as tile layers to avoid slowness
        map_id, vis = get_index_map_id(default_roi, idx_name, str(date_start), str(date_end), cloud_pct)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine - %s' % idx_name,
            name='%s Layer' % idx_name,
            overlay=True, control=True
        ).add_to(m)
except Exception:
    pass  # No imagery available for default area

# Drawing tools
draw = plugins.Draw(
    export=True, position='topleft',
    draw_options={
        'polyline': False,
        'polygon': {'allowIntersection': False, 'shapeOptions': {'color': '#10b981', 'weight': 3, 'fillOpacity': 0.15}},
        'rectangle': {'shapeOptions': {'color': '#8b5cf6', 'weight': 3, 'fillOpacity': 0.15}},
        'circle': {'shapeOptions': {'color': '#f59e0b', 'weight': 3, 'fillOpacity': 0.15}},
        'marker': True, 'circlemarker': False,
    },
    edit_options={'edit': True, 'remove': True}
)
draw.add_to(m)
folium.LayerControl().add_to(m)

st.markdown("### 🗺️ Interactive Map — Draw Your Study Area")
st.markdown(
    "✏️ Use the **drawing toolbar** (left side of map) to draw polygons, rectangles, circles, "
    "or place markers. Sentinel-2 index analysis runs automatically for drawn areas."
)
map_data = st_folium(m, width=None, height=550, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# DRAWN FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def geojson_to_ee_geometry(feature):
    geom = feature.get('geometry', feature)
    geom_type = geom.get('type', '')
    coords = geom.get('coordinates', [])
    if geom_type == 'Point':
        return ee.Geometry.Point(coords).buffer(point_buffer_m)
    elif geom_type == 'Polygon':
        return ee.Geometry.Polygon(coords)
    return None


def get_feature_label(feature, index):
    geom = feature.get('geometry', feature)
    geom_type = geom.get('type', '')
    coords = geom.get('coordinates', [])
    if geom_type == 'Point':
        return "Point %d (%.4f°N, %.4f°E)" % (index, coords[1], coords[0])
    elif geom_type == 'Polygon':
        n = len(coords[0]) - 1 if coords else 0
        return "Polygon %d (%d vertices)" % (index, n)
    return "Shape %d" % index


all_drawings = map_data.get('all_drawings', []) if map_data else []
if 'drawn_features' not in st.session_state:
    st.session_state.drawn_features = []
if 'drawn_analysis' not in st.session_state:
    st.session_state.drawn_analysis = []

if all_drawings and len(all_drawings) > 0:
    st.session_state.drawn_features = all_drawings

st.markdown("---")

if len(st.session_state.drawn_features) > 0:
    st.markdown("### 🔬 Study Area Analysis")
    st.success("**%d study area(s)** defined — Analyzing with Sentinel-2 imagery..." % len(st.session_state.drawn_features))

    analysis_results = []
    for idx, feature in enumerate(st.session_state.drawn_features, 1):
        ee_geom = geojson_to_ee_geometry(feature)
        if ee_geom:
            try:
                geom_info = json.dumps(ee_geom.getInfo())
                label = get_feature_label(feature, idx)
                st.markdown("## 📍 %s" % label)
                stats = display_index_analysis(geom_info, label, selected_indices)
                analysis_results.append({
                    'label': label,
                    'stats': stats,
                    'geom_info': geom_info,
                })
            except Exception as e:
                st.error("Error analyzing feature %d: %s" % (idx, str(e)))

    st.session_state.drawn_analysis = analysis_results

    # Multi-site comparison
    if len(analysis_results) > 1 and len(selected_indices) > 0:
        st.markdown("---")
        st.markdown("### 📋 Multi-Site Comparison")
        for idx_name in selected_indices:
            compare_data = []
            for ar in analysis_results:
                if idx_name in ar['stats']:
                    compare_data.append({
                        'Site': ar['label'],
                        'Mean': ar['stats'][idx_name].get('mean', 0),
                        'Min': ar['stats'][idx_name].get('min', 0),
                        'Max': ar['stats'][idx_name].get('max', 0),
                        'Std Dev': ar['stats'][idx_name].get('stdDev', 0),
                    })
            if compare_data:
                st.markdown("#### %s Comparison" % idx_name)
                st.dataframe(compare_data, use_container_width=True, hide_index=True)
else:
    st.markdown("### ✏️ Draw on the Map to Begin Analysis")
    st.markdown(
        '<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1px solid #86efac;'
        'border-radius:12px;padding:20px;margin:10px 0;">'
        '<h4 style="margin:0 0 8px;color:#166534;">🌿 How to use this app</h4>'
        '<ol style="color:#15803d;margin:8px 0;">'
        '<li>Select your <strong>date range</strong> and <strong>indices</strong> in the sidebar</li>'
        '<li>Use the <strong>drawing toolbar</strong> on the map to define study areas</li>'
        '<li>View <strong>histograms, time series, comparisons, and land cover</strong> analysis</li>'
        '<li>Generate a <strong>professional PDF report</strong> with all results</li>'
        '</ol></div>',
        unsafe_allow_html=True
    )

# ── Manual Coordinate Input ──────────────────────────────────────────
st.markdown("---")
st.markdown("### 📍 Analyze by Coordinates")
mc1, mc2, mc3 = st.columns([2, 2, 1])
with mc1:
    manual_lat = st.number_input("Latitude", value=20.30, format="%.4f", step=0.01)
with mc2:
    manual_lon = st.number_input("Longitude", value=85.82, format="%.4f", step=0.01)
with mc3:
    manual_buf = st.number_input("Buffer (m)", value=1000, min_value=100, max_value=50000, step=100)

if st.button("🔍 Analyze This Location", type="primary"):
    man_geom = ee.Geometry.Point(manual_lon, manual_lat).buffer(manual_buf)
    man_info = json.dumps(man_geom.getInfo())
    st.markdown("## 📍 Analysis for (%.4f°N, %.4f°E) — %dm buffer" % (manual_lat, manual_lon, manual_buf))
    display_index_analysis(man_info, "Manual (%.3f, %.3f)" % (manual_lat, manual_lon), selected_indices)


# ══════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════

def pdf_safe(text):
    """Sanitize text for FPDF latin-1 encoding."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '*', '\u2026': '...',
        '\u00b0': 'deg', '\u2019': "'", '\u00d7': 'x',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode('latin-1', errors='replace').decode('latin-1')

class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1: return
        self.set_fill_color(5, 150, 105)
        self.rect(0, 0, 210, 14, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", 'B', 9)
        self.set_y(2)
        self.cell(0, 10, "Sentinel-2 Index Analysis Report", align='C')
        self.set_text_color(0, 0, 0)
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Google Earth Engine | Sentinel-2 SR | Page %s" % self.page_no(), align='C')


def sec_head(pdf, num, title):
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(5, 150, 105)
    pdf.cell(0, 10, txt="%s. %s" % (num, title), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(5, 150, 105)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)


def generate_pdf_report(title, author, date_s, date_e, cloud, indices_list, analysis_list, thumbnails=None):
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    sec = [0]
    def ns():
        sec[0] += 1
        return sec[0]

    # COVER PAGE
    pdf.add_page()
    pdf.set_fill_color(5, 150, 105)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 24)
    pdf.set_y(8)
    pdf.cell(0, 14, txt="SENTINEL-2 INDEX ANALYSIS", ln=True, align='C')
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 8, txt="NDVI / NDWI / Spectral Indices Report", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    pdf.set_fill_color(240, 240, 240)
    pdf.set_draw_color(200, 200, 200)
    for lbl, val in [
        ("Report Title:", title), ("Author:", author),
        ("Date Generated:", datetime.now().strftime('%B %d, %Y %H:%M')),
        ("Analysis Period:", "%s to %s" % (date_s, date_e)),
        ("Cloud Filter:", "<%s%%" % cloud),
        ("Indices Computed:", ", ".join(indices_list)),
        ("GEE Project:", "spatialgeography"),
        ("Satellite:", "Sentinel-2 SR Harmonized (10m)"),
    ]:
        pdf.set_x(15)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(80, 8, pdf_safe("  %s" % lbl), border=1, fill=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(100, 8, pdf_safe("  %s" % val), border=1, fill=True, ln=True)
    pdf.ln(4)

    # Thumbnails on cover
    if thumbnails:
        for idx_name, path in thumbnails.items():
            if path and os.path.exists(path):
                pdf.set_font("Arial", 'B', 11)
                pdf.set_text_color(5, 150, 105)
                pdf.cell(0, 8, txt="%s Map" % idx_name, ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.image(path, x=20, w=170)
                pdf.ln(3)
                pdf.set_font("Arial", 'I', 8)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5, txt="Figure: %s visualization for study area" % idx_name, ln=True, align='C')
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)

    # DATA SOURCE & METHODOLOGY
    pdf.add_page()
    sec_head(pdf, ns(), "DATA SOURCE")
    pdf.set_font("Arial", size=10)
    for k, v in [
        ("Satellite", "Sentinel-2 (ESA Copernicus)"),
        ("Product", "Level-2A Surface Reflectance (Harmonized)"),
        ("Collection ID", "COPERNICUS/S2_SR_HARMONIZED"),
        ("Spatial Resolution", "10m (Bands B2-B4, B8), 20m (B11-B12)"),
        ("Temporal Resolution", "5-day revisit"),
        ("Cloud Masking", "QA60 band (opaque clouds + cirrus)"),
    ]:
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(60, 7, "  %s" % k, border=1, fill=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(120, 7, "  %s" % v, border=1, ln=True)
    pdf.ln(4)

    sec_head(pdf, ns(), "INDEX FORMULAS")
    pdf.set_font("Arial", size=10)
    formulas = [
        ("NDVI", "(B8 - B4) / (B8 + B4)", "Vegetation density"),
        ("NDWI", "(B3 - B8) / (B3 + B8)", "Water detection"),
        ("EVI", "2.5*(B8-B4)/(B8+6*B4-7.5*B2+1)", "Enhanced vegetation"),
        ("SAVI", "1.5*(B8-B4)/(B8+B4+0.5)", "Soil-adjusted vegetation"),
        ("MNDWI", "(B3 - B11) / (B3 + B11)", "Urban water detection"),
        ("BSI", "((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))", "Bare soil detection"),
    ]
    for name, formula, desc in formulas:
        if name in indices_list:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(20, 7, "  %s" % name, border=1, fill=True)
            pdf.set_font("Arial", size=9)
            pdf.cell(90, 7, "  %s" % formula, border=1)
            pdf.cell(70, 7, "  %s" % desc, border=1, ln=True)
    pdf.ln(4)

    # RESULTS FOR EACH STUDY AREA
    if analysis_list:
        for ar in analysis_list:
            pdf.add_page()
            sec_head(pdf, ns(), pdf_safe("RESULTS: %s" % ar['label']))

            for idx_name, stats in ar['stats'].items():
                pdf.set_font("Arial", 'B', 11)
                pdf.set_text_color(5, 150, 105)
                pdf.cell(0, 8, txt=idx_name, ln=True)
                pdf.set_text_color(0, 0, 0)

                pdf.set_fill_color(245, 245, 245)
                pdf.set_draw_color(200, 200, 200)
                for sk, sv in [
                    ("Mean", stats.get('mean', 0)), ("Min", stats.get('min', 0)),
                    ("Max", stats.get('max', 0)), ("Std Dev", stats.get('stdDev', 0)),
                    ("P10", stats.get('p10', 0)), ("P25", stats.get('p25', 0)),
                    ("Median", stats.get('p50', 0)), ("P75", stats.get('p75', 0)),
                    ("P90", stats.get('p90', 0)),
                ]:
                    pdf.set_font("Arial", 'B', 9)
                    pdf.cell(40, 6, "  %s" % sk, border=1, fill=True)
                    pdf.set_font("Arial", size=9)
                    pdf.cell(40, 6, "  %.4f" % sv, border=1, ln=True)
                pdf.ln(3)

    # INTERPRETATION
    pdf.add_page()
    sec_head(pdf, ns(), "INTERPRETATION GUIDE")
    pdf.set_font("Arial", size=10)
    for idx_name, desc in INDEX_DESCRIPTIONS.items():
        if idx_name in indices_list:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 7, txt=idx_name, ln=True)
            pdf.set_font("Arial", size=9)
            pdf.multi_cell(0, 5, txt=pdf_safe(desc))
            pdf.ln(2)

    # DISCLAIMER
    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4,
        txt=pdf_safe("Disclaimer: Auto-generated using Sentinel-2 SR data via Google Earth Engine. "
            "Cloud masking may not remove all cloud artifacts. Results should be validated "
            "with field data for critical applications. Generated: %s." % datetime.now().strftime('%Y-%m-%d %H:%M'))
    )

    return bytes(pdf.output())


# SIDEBAR PDF BUTTON
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Generate Report")
inc_thumbnails = st.sidebar.checkbox("Include basic thumbnails", value=False)
inc_pub_maps = st.sidebar.checkbox("Include Publication-Quality maps", value=True)

if st.sidebar.button("Generate PDF Report", type="primary"):
    with st.sidebar:
        analysis_data = st.session_state.get('drawn_analysis', [])
        thumbs = {}

        if analysis_data:
            geom = ee.Geometry(json.loads(analysis_data[0]['geom_info']))
            
            if inc_thumbnails:
                with st.spinner("Generating basic thumbnails..."):
                    for idx_name in selected_indices[:2]:
                        try:
                            thumbs[idx_name] = get_index_thumbnail(
                                geom, idx_name, str(date_start), str(date_end), cloud_pct
                            )
                        except Exception:
                            pass
            
            if inc_pub_maps:
                with st.spinner("Generating publication quality maps..."):
                    for idx_name in selected_indices[:2]:
                        try:
                            map_bytes = create_publication_map(
                                geom, idx_name, str(date_start), str(date_end), cloud_pct
                            )
                            if map_bytes:
                                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                tmp.write(map_bytes)
                                tmp.close()
                                thumbs[f"{idx_name} (Publication Quality)"] = tmp.name
                        except Exception:
                            pass

        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_pdf_report(
                report_title, report_author,
                str(date_start), str(date_end), cloud_pct,
                selected_indices, analysis_data, thumbs
            )
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name="%s.pdf" % report_title.replace(' ', '_'),
                mime="application/pdf"
            )
            st.success("PDF ready!")

        for p in thumbs.values():
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Gemini AI Assistant")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get a free key at ai.google.dev")
st.sidebar.markdown(
    "<small style='color:#a7f3d0'>Sentinel-2 &bull; Earth Engine &bull; Folium &bull; Plotly &bull; Gemini AI</small>",
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════════════
# GEMINI AI ASSISTANT — WITH LIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def build_analysis_context():
    """Build a text summary of all current analysis data for Gemini."""
    ctx = []
    ctx.append("=== SENTINEL-2 INDEX ANALYSIS SESSION ===")
    ctx.append("Date Range: %s to %s" % (date_start, date_end))
    ctx.append("Cloud Filter: <%d%%" % cloud_pct)
    ctx.append("Selected Indices: %s" % ", ".join(selected_indices))

    analysis_data = st.session_state.get('drawn_analysis', [])
    if analysis_data:
        ctx.append("\n=== ANALYSIS RESULTS ===")
        for i, ar in enumerate(analysis_data, 1):
            ctx.append("\n--- Study Area %d: %s ---" % (i, ar['label']))
            for idx_name, stats in ar['stats'].items():
                ctx.append("%s: %s" % (idx_name, str(stats)))
    else:
        ctx.append("\nNo study areas drawn yet.")

    ctx.append("\n=== INDEX REFERENCE ===")
    for name, desc in INDEX_DESCRIPTIONS.items():
        ctx.append("%s: %s" % (name, desc))

    return "\n".join(ctx)


def execute_ai_analysis(lat, lon, buffer_m, indices_to_compute):
    """Run actual EE analysis for a location."""
    geom = ee.Geometry.Point(lon, lat).buffer(buffer_m)
    geom_info = json.dumps(geom.getInfo())

    results = {}
    for idx_name in indices_to_compute:
        stats, count = compute_index_stats(
            geom_info, idx_name, idx_name,
            str(date_start), str(date_end), cloud_pct
        )
        if stats:
            results[idx_name] = stats

    return results, geom_info, count


def parse_coordinates(text):
    """Try to extract lat, lon, buffer from user text."""
    import re
    # patterns: "20.5, 85.8" or "lat 20.5 lon 85.8" or "20.5N 85.8E"
    coords = re.findall(r'[-+]?\d+\.?\d*', text)
    lat, lon, buf = None, None, 1000

    # Look for lat/lon pairs
    nums = [float(x) for x in coords]
    if len(nums) >= 2:
        # Heuristic: if a number is between -90..90, likely lat; 
        # if between -180..180, likely lon
        candidates = []
        for n in nums:
            if -90 <= n <= 90:
                candidates.append(('lat', n))
            elif -180 <= n <= 180:
                candidates.append(('lon', n))
            elif 100 <= n <= 50000:
                buf = int(n)

        if len(candidates) >= 2:
            lat = candidates[0][1]
            lon = candidates[1][1]
        elif len(nums) >= 2:
            lat = nums[0]
            lon = nums[1]

        if len(nums) >= 3 and buf == 1000:
            third = nums[2]
            if 100 <= third <= 50000:
                buf = int(third)

    return lat, lon, buf


def detect_action(text):
    """Detect if user wants a specific action performed."""
    text_lower = text.lower()

    # Compute/analyze at location
    if any(kw in text_lower for kw in ['compute', 'analyze', 'calculate', 'check', 'scan', 'run analysis']):
        lat, lon, buf = parse_coordinates(text)
        if lat is not None and lon is not None:
            indices = []
            for idx in ['NDVI', 'NDWI', 'EVI', 'SAVI', 'MNDWI', 'BSI']:
                if idx.lower() in text_lower:
                    indices.append(idx)
            if not indices:
                indices = selected_indices  # Use sidebar selection
            return {'action': 'compute', 'lat': lat, 'lon': lon,
                    'buffer': buf, 'indices': indices}

    # Histogram request
    if any(kw in text_lower for kw in ['histogram', 'distribution', 'frequency']):
        for idx in ['NDVI', 'NDWI', 'EVI', 'SAVI', 'MNDWI', 'BSI']:
            if idx.lower() in text_lower:
                return {'action': 'histogram', 'index': idx}
        return {'action': 'histogram', 'index': selected_indices[0]}

    # Time series request
    if any(kw in text_lower for kw in ['time series', 'trend', 'temporal', 'monthly']):
        for idx in ['NDVI', 'NDWI', 'EVI', 'SAVI', 'MNDWI', 'BSI']:
            if idx.lower() in text_lower:
                return {'action': 'timeseries', 'index': idx}
        return {'action': 'timeseries', 'index': selected_indices[0]}

    # Comparison
    if any(kw in text_lower for kw in ['compare', 'comparison', 'bar chart', 'side by side']):
        return {'action': 'compare'}

    # Land cover
    if any(kw in text_lower for kw in ['land cover', 'land use', 'classify', 'classification']):
        return {'action': 'landcover'}

    return None


st.markdown("---")
st.markdown("### 🤖 Gemini AI Analysis Assistant")

if gemini_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        st.markdown(
            '<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1px solid #93c5fd;'
            'border-radius:12px;padding:16px;margin:8px 0;">'
            '<p style="margin:0;color:#1e40af;">🤖 <strong>Gemini AI</strong> is connected and can <strong>perform live analysis!</strong> '
            'Try: "Compute NDVI for 20.46, 85.88" or "Show NDVI histogram" or "Analyze 20.30, 85.82 with 2000m buffer"</p></div>',
            unsafe_allow_html=True
        )

        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                # Re-render charts from history
                if 'chart' in msg:
                    st.plotly_chart(msg['chart'], use_container_width=True)
                if 'metrics' in msg:
                    cols = st.columns(len(msg['metrics']))
                    for i, (k, v) in enumerate(msg['metrics'].items()):
                        cols[i].metric(k, v)

        user_prompt = st.chat_input(
            "Try: 'Compute NDVI for 20.46, 85.88' or 'Show histogram' or 'Analyze vegetation at 20.3, 85.8'"
        )

        if user_prompt:
            st.session_state.chat_messages.append({'role': 'user', 'content': user_prompt})
            with st.chat_message('user'):
                st.markdown(user_prompt)

            # Detect if user wants a live action
            action = detect_action(user_prompt)

            with st.chat_message('assistant'):
                # === ACTION: COMPUTE STATS AT LOCATION ===
                if action and action['action'] == 'compute':
                    lat, lon, buf = action['lat'], action['lon'], action['buffer']
                    indices = action['indices']
                    st.markdown("**🔄 Running live analysis at (%.4f, %.4f) with %dm buffer...**" % (lat, lon, buf))
                    st.markdown("Indices: %s | Period: %s to %s" % (", ".join(indices), date_start, date_end))

                    with st.spinner("Querying Google Earth Engine..."):
                        try:
                            results, geom_info, img_count = execute_ai_analysis(
                                lat, lon, buf, indices
                            )

                            if results:
                                st.success("✅ Found **%d Sentinel-2 images** — Analysis complete!" % img_count)

                                msg_parts = ["**📍 Analysis Results for (%.4f, %.4f) — %dm buffer**\n" % (lat, lon, buf)]
                                msg_parts.append("**%d images** | Period: %s to %s\n" % (img_count, date_start, date_end))

                                for idx_name, stats in results.items():
                                    st.markdown("**%s Results:**" % idx_name)
                                    metric_cols = st.columns(5)
                                    metric_cols[0].metric("Mean", "%.4f" % stats.get('mean', 0))
                                    metric_cols[1].metric("Min", "%.4f" % stats.get('min', 0))
                                    metric_cols[2].metric("Max", "%.4f" % stats.get('max', 0))
                                    metric_cols[3].metric("Std Dev", "%.4f" % stats.get('stdDev', 0))
                                    metric_cols[4].metric("Median", "%.4f" % stats.get('p50', 0))

                                    msg_parts.append("\n**%s:** Mean=%.4f, Min=%.4f, Max=%.4f, StdDev=%.4f" % (
                                        idx_name, stats.get('mean', 0), stats.get('min', 0),
                                        stats.get('max', 0), stats.get('stdDev', 0)
                                    ))

                                # Generate comparison chart if multiple indices
                                if len(results) >= 2:
                                    fig = create_comparison_bar(results)
                                    st.plotly_chart(fig, use_container_width=True)

                                # Ask Gemini to interpret the results
                                interp_prompt = (
                                    "You are a remote sensing expert. The user just computed these indices at "
                                    "(%.4f, %.4f) with %dm buffer:\n%s\n\n"
                                    "Provide a brief (3-4 sentences) expert interpretation. "
                                    "What does this tell us about the land cover and environment?" % (
                                        lat, lon, buf, str(results)
                                    )
                                )
                                interp_resp = model.generate_content(interp_prompt)
                                st.markdown("---")
                                st.markdown("**🧠 AI Interpretation:**")
                                st.markdown(interp_resp.text)
                                msg_parts.append("\n\n**AI Interpretation:**\n" + interp_resp.text)

                                st.session_state.chat_messages.append({
                                    'role': 'assistant',
                                    'content': "\n".join(msg_parts)
                                })
                            else:
                                msg = "⚠️ No Sentinel-2 images found for this location. Try a different date range or increase cloud cover tolerance."
                                st.warning(msg)
                                st.session_state.chat_messages.append({'role': 'assistant', 'content': msg})

                        except Exception as e:
                            msg = "❌ Analysis error: %s" % str(e)
                            st.error(msg)
                            st.session_state.chat_messages.append({'role': 'assistant', 'content': msg})

                # === ACTION: HISTOGRAM ===
                elif action and action['action'] == 'histogram':
                    idx_name = action['index']
                    analysis_data = st.session_state.get('drawn_analysis', [])

                    if analysis_data:
                        geom_info = analysis_data[0]['geom_info']
                        st.markdown("**📊 Computing %s histogram for your study area...**" % idx_name)

                        with st.spinner("Computing histogram..."):
                            try:
                                bins, counts = compute_index_histogram(
                                    geom_info, idx_name,
                                    str(date_start), str(date_end), cloud_pct
                                )
                                if bins:
                                    fig = create_index_histogram(bins, counts, idx_name)
                                    st.plotly_chart(fig, use_container_width=True)

                                    # AI interpretation
                                    peak_bin = bins[counts.index(max(counts))]
                                    interp = model.generate_content(
                                        "Remote sensing expert: The %s histogram for this area shows "
                                        "peak values around %.3f. Total pixels: %d. "
                                        "Give a 2-sentence interpretation." % (
                                            idx_name, peak_bin, sum(counts))
                                    )
                                    st.markdown(interp.text)
                                    st.session_state.chat_messages.append({
                                        'role': 'assistant',
                                        'content': "Generated %s histogram. %s" % (idx_name, interp.text)
                                    })
                            except Exception as e:
                                st.error("Histogram error: %s" % str(e))
                    else:
                        st.info("Draw a study area on the map first, then ask for a histogram.")
                        st.session_state.chat_messages.append({
                            'role': 'assistant',
                            'content': "Please draw a study area on the map first, then I can generate histograms."
                        })

                # === ACTION: TIME SERIES ===
                elif action and action['action'] == 'timeseries':
                    idx_name = action['index']
                    analysis_data = st.session_state.get('drawn_analysis', [])

                    if analysis_data:
                        geom_info = analysis_data[0]['geom_info']
                        st.markdown("**📈 Computing %s time series...**" % idx_name)

                        with st.spinner("Computing monthly %s (this may take a moment)..." % idx_name):
                            try:
                                dates, values = compute_time_series(
                                    geom_info, idx_name,
                                    str(date_start), str(date_end), cloud_pct
                                )
                                if dates:
                                    fig = create_time_series_chart(dates, values, idx_name)
                                    st.plotly_chart(fig, use_container_width=True)

                                    valid_vals = [v for v in values if v is not None]
                                    if valid_vals:
                                        trend = "increasing" if valid_vals[-1] > valid_vals[0] else "decreasing"
                                        interp = model.generate_content(
                                            "The %s time series from %s to %s shows values: %s. "
                                            "Overall trend: %s. Range: %.4f to %.4f. "
                                            "Give a 3-sentence expert interpretation about vegetation/water dynamics." % (
                                                idx_name, date_start, date_end,
                                                str(valid_vals[:6]), trend,
                                                min(valid_vals), max(valid_vals)
                                            )
                                        )
                                        st.markdown(interp.text)
                                        st.session_state.chat_messages.append({
                                            'role': 'assistant',
                                            'content': "Generated %s time series. %s" % (idx_name, interp.text)
                                        })
                            except Exception as e:
                                st.error("Time series error: %s" % str(e))
                    else:
                        st.info("Draw a study area on the map first.")
                        st.session_state.chat_messages.append({
                            'role': 'assistant',
                            'content': "Please draw a study area on the map first."
                        })

                # === ACTION: LAND COVER ===
                elif action and action['action'] == 'landcover':
                    analysis_data = st.session_state.get('drawn_analysis', [])
                    if analysis_data:
                        geom_info = analysis_data[0]['geom_info']
                        st.markdown("**🗺️ Computing land cover classification...**")

                        with st.spinner("Classifying land cover from NDVI..."):
                            try:
                                bins, counts = compute_index_histogram(
                                    geom_info, 'NDVI',
                                    str(date_start), str(date_end), cloud_pct
                                )
                                if bins:
                                    result = create_land_cover_pie(bins, counts)
                                    if result:
                                        fig, cats, pcts = result
                                        st.plotly_chart(fig, use_container_width=True)

                                        lc_summary = ", ".join(["%s: %.1f%%" % (c, p) for c, p in zip(cats, pcts)])
                                        interp = model.generate_content(
                                            "Land cover classification from NDVI for this area: %s. "
                                            "Provide a 3-sentence expert interpretation about land use." % lc_summary
                                        )
                                        st.markdown(interp.text)
                                        st.session_state.chat_messages.append({
                                            'role': 'assistant',
                                            'content': "Land Cover: %s\n\n%s" % (lc_summary, interp.text)
                                        })
                            except Exception as e:
                                st.error("Land cover error: %s" % str(e))
                    else:
                        st.info("Draw a study area on the map first.")
                        st.session_state.chat_messages.append({
                            'role': 'assistant', 'content': "Please draw a study area first."
                        })

                # === ACTION: COMPARE ===
                elif action and action['action'] == 'compare':
                    analysis_data = st.session_state.get('drawn_analysis', [])
                    if analysis_data and analysis_data[0]['stats']:
                        stats = analysis_data[0]['stats']
                        if len(stats) >= 2:
                            fig = create_comparison_bar(stats)
                            st.plotly_chart(fig, use_container_width=True)

                            interp = model.generate_content(
                                "Compare these index values for a study area: %s. "
                                "What does the combination of these indices reveal?" % str(stats)
                            )
                            st.markdown(interp.text)
                            st.session_state.chat_messages.append({
                                'role': 'assistant', 'content': interp.text
                            })
                        else:
                            st.info("Select at least 2 indices in the sidebar.")
                    else:
                        st.info("Draw a study area and compute indices first.")
                        st.session_state.chat_messages.append({
                            'role': 'assistant', 'content': "Please draw a study area first."
                        })

                # === NO ACTION — PURE AI CHAT ===
                else:
                    analysis_context = build_analysis_context()
                    system_prompt = (
                        "You are a remote sensing expert in a Sentinel-2 analysis app. "
                        "You can interpret data AND tell users to use these commands:\n"
                        "- 'Compute NDVI for <lat>, <lon>' to run live analysis\n"
                        "- 'Show NDVI histogram' to generate distribution chart\n"
                        "- 'Show NDVI time series' for monthly trends\n"
                        "- 'Land cover classification' for NDVI-based land use\n"
                        "- 'Compare indices' for bar chart comparison\n\n"
                        "Current data:\n%s\n\n"
                        "Answer the user's question. If they want analysis, suggest the right command.\n"
                    ) % analysis_context

                    chat_history = []
                    for msg in st.session_state.chat_messages[:-1]:
                        chat_history.append({
                            'role': msg['role'] if msg['role'] == 'user' else 'model',
                            'parts': [msg['content']]
                        })

                    with st.spinner("Thinking..."):
                        try:
                            chat = model.start_chat(history=chat_history)
                            response = chat.send_message(system_prompt + "\nUser: " + user_prompt)
                            st.markdown(response.text)
                            st.session_state.chat_messages.append({
                                'role': 'assistant', 'content': response.text
                            })
                        except Exception as e:
                            st.error("Error: %s" % str(e))

        # Quick action buttons
        st.markdown("#### ⚡ Quick Actions")
        analysis_data = st.session_state.get('drawn_analysis', [])
        has_data = len(analysis_data) > 0

        qr1, qr2, qr3, qr4 = st.columns(4)
        with qr1:
            if st.button("📊 NDVI Histogram", disabled=not has_data, key="qa_hist"):
                st.session_state.chat_messages.append({'role': 'user', 'content': 'Show NDVI histogram'})
                st.rerun()
        with qr2:
            if st.button("📈 NDVI Trend", disabled=not has_data, key="qa_trend"):
                st.session_state.chat_messages.append({'role': 'user', 'content': 'Show NDVI time series'})
                st.rerun()
        with qr3:
            if st.button("🗺️ Land Cover", disabled=not has_data, key="qa_lc"):
                st.session_state.chat_messages.append({'role': 'user', 'content': 'land cover classification'})
                st.rerun()
        with qr4:
            if st.button("🔄 Compare All", disabled=not has_data, key="qa_comp"):
                st.session_state.chat_messages.append({'role': 'user', 'content': 'compare indices'})
                st.rerun()

        qr5, qr6, qr7, qr8 = st.columns(4)
        with qr5:
            if st.button("🌱 Interpret Results", disabled=not has_data, key="qa_interp"):
                st.session_state.chat_messages.append({'role': 'user', 'content': 'Interpret all my spectral index results comprehensively'})
                st.rerun()
        with qr6:
            if st.button("💧 Water Analysis", disabled=not has_data, key="qa_water"):
                st.session_state.chat_messages.append({'role': 'user', 'content': 'Analyze water indices NDWI and MNDWI for flood risk'})
                st.rerun()
        with qr7:
            if st.button("📋 Recommendations", disabled=not has_data, key="qa_recs"):
                st.session_state.chat_messages.append({'role': 'user', 'content': 'Give 5 environmental management recommendations based on my analysis'})
                st.rerun()
        with qr8:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()

    except ImportError:
        st.error("Install `google-generativeai` — add it to requirements.txt.")
    except Exception as e:
        st.error("Gemini error: %s" % str(e))
else:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#fefce8,#fef9c3);border:1px solid #fde047;'
        'border-radius:12px;padding:20px;margin:10px 0;">'
        '<h4 style="margin:0 0 8px;color:#854d0e;">🤖 AI Assistant (Optional)</h4>'
        '<p style="margin:0;color:#a16207;">Enter your <strong>Gemini API key</strong> in the sidebar. '
        'The AI can <strong>perform live analysis</strong> via chat:</p>'
        '<ul style="color:#a16207;margin:8px 0;">'
        '<li><strong>"Compute NDVI for 20.46, 85.88"</strong> - runs real EE analysis</li>'
        '<li><strong>"Show NDVI histogram"</strong> - generates distribution chart</li>'
        '<li><strong>"Show NDVI time series"</strong> - monthly trend analysis</li>'
        '<li><strong>"Land cover classification"</strong> - NDVI-based land use</li>'
        '<li><strong>"Compare indices"</strong> - bar chart comparison</li>'
        '</ul>'
        '<p style="margin:4px 0 0;color:#a16207;">Free API key: '
        '<a href="https://ai.google.dev" target="_blank" style="color:#854d0e;font-weight:bold;">ai.google.dev</a></p>'
        '</div>',
        unsafe_allow_html=True
    )

