# tc-dav

Tropical cyclone intensity and structure models based on Deviation Angle Variance (DAV).

This package provides tools for analysing tropical cyclone wind structure using geostationary brightness temperature images.

---

## Installation

```bash
python -m pip install tc-dav
```

> **Note:**  
> This package is installed as `tc-dav` but imported as `dav`.

```bash
pip install tc-dav
```

```python
import dav
```

---

## Troubleshooting

This package uses GPU acceleration via numba. A CUDA-capable NVIDIA GPU with drivers installed is required.

Run in command prompt:

```bash
nvidia-smi
```

You should see:

- A GPU name (e.g. NVIDIA RTX 4500)
- A Driver Version (e.g. 573.73)
- A CUDA Version (e.g. 12.8)

If this command fails or shows errors:

- Your NVIDIA drivers may not be installed correctly
- Your GPU may not be detected

If CUDA is missing:

- GPU acceleration will not work

Note: values do not need to match exactly.

---

## Example Loading Images
```python
import numpy as np
from dav.utils import IbtracsReader, MergirReader

ibtracs_fn = "C:/Path/to/Ibtracs.nc"  # https://doi.org/10.25921/82ty-9e16
mergir_dir = "C:/Path/to/MERGIR"      # https://doi.org/10.5067/P4HZB9N27EKU

tc_name = "Chris"
tc_year = 2024

print("Setting up IBTrACS")
ibtracs = IbtracsReader(ibtracs_fn)

# The following approach at collecting the files is not recommended in
# practice. This will take a long time to execute for large datasets.
mergir = MergirReader(f"{mergir_dir}/*.nc4", size=10)

print("Reading images")
tc = ibtracs.read(tc_name, tc_year)
images = [mergir.read_tc_index(tc, i) for i in range(len(tc['iso_time']))]
images = np.array(images, dtype=np.float32)

# This will produce a NumPy array of Brightness temperature values:
# images.shape -> (time, height, width)

```

## Example Generating DAV maps
```python
from dav.generate import dav

dav_radius_km = 300  # radius of DAV calculation in km
image_resolution = 8  # km per pixel

radius_pixels = dav_radius_km / image_resolution

dav_maps = dav(images, radius_pixels)

# This will produce a NumPy array of DAV maps:
# dav_maps.shape -> (time, height, width)
```

## Example predicting TC intensity
```python
from dav.generate import centre_dav
from dav.intensity import model as intensity_model

dav_radius_km = 300  # radius of DAV calculation in km
image_resolution = 8  # km per pixel

radius_pixels = dav_radius_km / image_resolution
starting_basin = tc['basin'][0]

cdav_values = centre_dav(images, radius_pixels) # images retrieved from earlier example
predicted_intensity = intensity_model.predict(cdav_values, starting_basin)

# This will produce a NumPy array of predicted wind intensity in knots:
# dav_maps.shape -> (time)
```

## Example predicting TC wind radii
```python
from dav.utils import get_dav_profile, get_tc_age
from dav.wind_radii import model as wind_radii_model

image_resolution = 8

profile = get_dav_profile(dav_maps, # dav_maps from earlier example
                          resolution=image_resolution,
                          radius=75)
tc_age = get_tc_age(tc['usa_wind'], samples_per_hour=1/3) # For three-hourly samples.

# sst can be retrieved from elsewhere, during experiments we used ERA5 data
data = {"profile": profile,
        "age": tc_age,
        "sst": np.array([29.73, 29.73, 29.26, 29.11, np.nan]),
        "wind": tc['usa_wind']}

tc_basin = tc['basin'][0]
quadrant = "symmetric"
radius = "r34"

predictions = wind_radii_model.predict(data,
                                       basin=tc_basin,
                                       quadrant=quadrant,
                                       radius=radius)

# This will produce 34-kt wind radii estimates symmetrically around the TC.
# predictions.shape -> (time,)
```
