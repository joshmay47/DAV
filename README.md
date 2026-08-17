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

tc\_name = "Chris"
tc\_year = 2024

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
from dav.intensity import model

dav_radius_km = 300  # radius of DAV calculation in km
image_resolution = 8  # km per pixel

cdav_values = centre_dav(images, radius_pixels) # images retrieved from earlier example
helene_predicted_intensity = model.predict(cdav_values, tc['basin'][0]) # Starting basin retrieved from earlier example

# This will produce a NumPy array of predicted wind intensity in knots:
# dav_maps.shape -> (time)
```
