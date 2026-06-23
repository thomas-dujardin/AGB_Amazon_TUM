# =======================================================
# Batch-start annual 320x320 Sentinel input tiles
# and 32x32 ESA CCI AGB target tiles (100mx100m, or 1 Hectare).
#
# Corrected for:
#   - keeping SD uncertainty band
#   - avoiding incomplete edge tiles
#
# Exports:
#
#   img_tile_i.tif:
#       band 1 = VV
#       band 2 = VH
#       band 3 = B02
#       band 4 = B03
#       band 5 = B04
#       band 6 = B08
#       band 7 = B11
#       band 8 = B12
#
#   esa_tile_i.tif:
#       band 1 = AGB
#       band 2 = SD
#       band 3 = valid_mask
#
# Notes:
#   - AGB and SD are annual.
#   - valid_mask = 1 where AGB > 0.
#   - invalid pixels get AGB = -9999 and SD = -9999.
#   - no .clip(amazon) is used before export;
#     region=tile defines each square tile.
# =======================================================

import time
import ee

# -------------------------------------------------------
# Authenticate / initialize
# -------------------------------------------------------

ee.Authenticate()

ee.Initialize(project="x")

# -------------------------------------------------------
# Global settings
# -------------------------------------------------------

YEAR = 2021
START_DATE = f"{YEAR}-01-01"
END_DATE = f"{YEAR + 1}-01-01"

EXPORT_CRS = "EPSG:3857"
PATCH_SIZE_M = 3200

INPUT_DIM = 320
TARGET_DIM = 32

DRIVE_FOLDER = "biomass_tiles_320_annual_agb_sd_corrected"

START_INDEX = 0
MAX_TILES = 1

SLEEP_BETWEEN_TASKS_SECONDS = 0.5

VALID_MASK_MODE = "AGB_GT_0"

print("YEAR:", YEAR)
print("START_DATE:", START_DATE)
print("END_DATE:", END_DATE)
print("DRIVE_FOLDER:", DRIVE_FOLDER)
print("VALID_MASK_MODE:", VALID_MASK_MODE)

# -------------------------------------------------------
# Amazon AOI (SMALL)
# -------------------------------------------------------

amazon = ee.Geometry.Polygon([
    [
        [-60.427, -2.798],
        [-60.427, -2.372],
        [-59.764, -2.372],
        [-59.764, -2.798],
        [-60.427, -2.798],
    ]
])

# -------------------------------------------------------
# Sentinel-1: VV, VH
# -------------------------------------------------------

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(amazon)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .select(["VV", "VH"])
    .median()
    .toFloat()
)

# -------------------------------------------------------
# Sentinel-2 masking
# -------------------------------------------------------

def mask_s2(image):
    scl = image.select("SCL")

    valid = (
        scl.neq(0)          # no data
        .And(scl.neq(1))    # saturated / defective
        .And(scl.neq(3))    # cloud shadow
        .And(scl.neq(8))    # medium probability cloud
        .And(scl.neq(9))    # high probability cloud
        .And(scl.neq(10))   # cirrus
        .And(scl.neq(11))   # snow / ice
    )

    return (
        image
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
        .multiply(0.0001)
        .updateMask(valid)
        .copyProperties(image, ["system:time_start"])
    )

# -------------------------------------------------------
# Sentinel-2: B2, B3, B4, B8, B11, B12
# -------------------------------------------------------

s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(amazon)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    .map(mask_s2)
    .median()
    .select(
        ["B2", "B3", "B4", "B8", "B11", "B12"],
        ["B02", "B03", "B04", "B08", "B11", "B12"],
    )
    .toFloat()
)

# -------------------------------------------------------
# Corrected input stack
# -------------------------------------------------------
#
# Important:
# Do NOT do .clip(amazon) here.
#
# The export call already uses region=tile.
# If you clip to amazon, edge tiles are cut/masked.
# -------------------------------------------------------

input_stack = (
    ee.Image.cat([
        s1.select("VV"),
        s1.select("VH"),
        s2.select("B02"),
        s2.select("B03"),
        s2.select("B04"),
        s2.select("B08"),
        s2.select("B11"),
        s2.select("B12"),
    ])
    .toFloat()
    .unmask(0, False)
    .updateMask(ee.Image.constant(1))
)

# -------------------------------------------------------
# ESA CCI AGB targets
# -------------------------------------------------------
#
# Asset:
#   projects/sat-io/open-datasets/ESA/ESA_CCI_AGB
#
# Expected raw bands:
#   AGB
#   SD
#
# SD is the uncertainty / standard deviation band.
# -------------------------------------------------------

agb_collection = ee.ImageCollection(
    "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB"
)

agb_img = (
    agb_collection
    .filterDate(START_DATE, END_DATE)
    .first()
    .toFloat()
)

raw_band_names = agb_img.bandNames().getInfo()

# -------------------------------------------------------
# Select AGB and SD robustly
# -------------------------------------------------------
#
# If the band names are exactly "AGB" and "SD", use them.
# If the UI shows generic band names, fall back to band index:
#   index 0 = AGB
#   index 1 = SD
# -------------------------------------------------------

if "AGB" in raw_band_names:
    agb_raw = agb_img.select("AGB").rename("AGB").toFloat()
else:
    agb_raw = agb_img.select([0]).rename("AGB").toFloat()

if "SD" in raw_band_names:
    sd_raw = agb_img.select("SD").rename("SD").toFloat()
else:
    if len(raw_band_names) >= 2:
        sd_raw = agb_img.select([1]).rename("SD").toFloat()


# -------------------------------------------------------
# Valid mask
# -------------------------------------------------------
#
# Default:
#   valid = AGB > 0
#
# Alternative:
#   valid = AGB > 0 and SD > 0
# -------------------------------------------------------

if VALID_MASK_MODE == "AGB_GT_0":
    valid_mask = agb_raw.gt(0)
elif VALID_MASK_MODE == "AGB_AND_SD_GT_0":
    valid_mask = agb_raw.gt(0).And(sd_raw.gt(0))

valid_mask = valid_mask.rename("valid_mask").toFloat()

# -------------------------------------------------------
# Fill invalid pixels
# -------------------------------------------------------
#
# For ML:
#   AGB = -9999 where invalid
#   SD  = -9999 where invalid
#   valid_mask = 0 where invalid
#
# Do NOT clip to amazon.
# region=tile defines the exported crop.
# -------------------------------------------------------

agb_filled = (
    agb_raw
    .where(valid_mask.eq(0), -9999)
    .rename("AGB")
    .toFloat()
)

sd_filled = (
    sd_raw
    .where(valid_mask.eq(0), -9999)
    .rename("SD")
    .toFloat()
)

target_stack = (
    agb_filled
    .addBands(sd_filled)
    .addBands(valid_mask.rename("valid_mask").toFloat())
    .toFloat()
    .unmask(-9999, False)
    .updateMask(ee.Image.constant(1))
)

# -------------------------------------------------------
# Grid: 3200m x 3200m tiles
# -------------------------------------------------------

grid_projection = ee.Projection(EXPORT_CRS).atScale(PATCH_SIZE_M)
grid = amazon.coveringGrid(grid_projection)

grid_size = grid.size().getInfo()

tiles = grid.toList(grid_size)

end_index = min(START_INDEX + MAX_TILES, grid_size)

print(f"Exporting tiles {START_INDEX} to {end_index - 1}")

# -------------------------------------------------------
# Batch-start export tasks
# -------------------------------------------------------

started_tasks = []

for i in range(START_INDEX, end_index):
    tile = ee.Feature(tiles.get(i)).geometry()

    img_description = f"img_tile_{i}"
    esa_description = f"esa_tile_{i}"

    img_task = ee.batch.Export.image.toDrive(
        image=input_stack,
        description=img_description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=img_description,
        region=tile,
        crs=EXPORT_CRS,
        dimensions=f"{INPUT_DIM}x{INPUT_DIM}",
        fileFormat="GeoTIFF",
        maxPixels=1e8,
    )

    esa_task = ee.batch.Export.image.toDrive(
        image=target_stack,
        description=esa_description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=esa_description,
        region=tile,
        crs=EXPORT_CRS,
        dimensions=f"{TARGET_DIM}x{TARGET_DIM}",
        fileFormat="GeoTIFF",
        maxPixels=1e8,
    )

    img_task.start()
    print(f"Started: {img_description}")
    started_tasks.append(img_task)
    time.sleep(SLEEP_BETWEEN_TASKS_SECONDS)

    esa_task.start()
    print(f"Started: {esa_description}")
    started_tasks.append(esa_task)
    time.sleep(SLEEP_BETWEEN_TASKS_SECONDS)

print()
print(f"Submitted {len(started_tasks)} tasks.")

for task in started_tasks:
    status = task.status()
    print(status.get("description"), status.get("state"))