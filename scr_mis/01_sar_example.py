import ee
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

# Google Earth Engine 
GEE_PROJECT = "GET-YOUR-GEE-PROJECT-ID"      # Your GEE project ID (https://developers.google.com/earth-engine/guides/quickstart_python)
ee.Initialize(project=GEE_PROJECT)

# Coordinates of Los Angeles Port
lat, lon = 33.75, -118.25
BUFFER_RADIUS_M = 3000

point = ee.Geometry.Point([lon, lat])
aoi = point.buffer(BUFFER_RADIUS_M).bounds()

# First image in August 2024
collection = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate('2024-08-01', '2024-08-31')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .sort('system:time_start')
)

first_image = ee.Image(collection.first())
image_date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print(f"Image date: {image_date}")

# Visualization parameters (this is an example, the analysis does not use visualized images)
vis_params_vv = {
    'min': -20.00, 
    'max': 10.00, 
    'bands': ['VV'], 
    'palette': ['000000', 'ffffff']
}

vis_params_vh = {
    'min': -10.00, 
    'max': 0.00, 
    'bands': ['VH'], 
    'palette': ['000000', 'ffffff']
}

def get_image_url(image, vis_params, region):
    url = image.getThumbURL({
        'region': region,
        'dimensions': 1024,
        'format': 'png',
        **vis_params
    })
    return url

vv_url = get_image_url(first_image, vis_params_vv, aoi)
vh_url = get_image_url(first_image, vis_params_vh, aoi)

vv_response = requests.get(vv_url)
vh_response = requests.get(vh_url)

vv_img = Image.open(BytesIO(vv_response.content))
vh_img = Image.open(BytesIO(vh_response.content))


# Save
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(vv_img, cmap='gray')
ax.axis('off')
plt.savefig('watching_trade/output_figures/sar_vv_example.pdf', bbox_inches='tight', pad_inches=0)


fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(vh_img, cmap='gray')
ax.axis('off')
plt.savefig('watching_trade/output_figures/sar_vh_example.pdf', bbox_inches='tight', pad_inches=0)

print(f"\nSaved: sar_vv_example.pdf, sar_vh_example.pdf")