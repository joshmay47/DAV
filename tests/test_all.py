"""
Tests used to check if package is working correctly.

Dependencies:
    numpy
    tc-dav

Author: Joshua May (josh.w.may@gmail.com)
"""
from pathlib import Path
import pytest

# Loading TC information and images modules
from dav.utils import IbtracsReader, MergirReader
import numpy as np

# Loading TC intensity estimation modules
from dav.generate import centre_dav
from dav.intensity import model

# Loading DAV map generation modules
from dav.generate import dav


@pytest.fixture(scope="session")
def test_data():
    """This will produce a NumPy array of Brightness temperature values

    images.shape -> (time, height, width)
    """
    ibtracs_fn = Path(__file__).parent / "example_ibtracs.nc"
    mergir_dir = Path(__file__).parent / "data"

    ibtracs = IbtracsReader(ibtracs_fn,
                            interpolating=False,  # Three-hourly for testing
                            columns=("iso_time",
                                     "lat",
                                     "lon",
                                     "basin",
                                     "usa_wind"))

    # The following approach at collecting the files is not recommended in
    # practice. This will take a long time to execute for large datasets.
    mergir = MergirReader(f"{mergir_dir}/*.nc4", size=10)

    tc = ibtracs.read("Chris", 2024)
    images = [mergir.read_tc_index(tc, i) for i in range(len(tc['iso_time']))]

    return np.array(images, dtype=np.float32), tc


def test_dav_maps(test_data):
    """This will produce a NumPy array of DAV maps

    dav_maps.shape -> (time, height, width)
    """
    images, _ = test_data

    dav_radius_km = 300  # radius of DAV calculation in km
    image_resolution = 8  # km per pixel

    radius_pixels = dav_radius_km/image_resolution
    maps = dav(images, radius_pixels)

    assert maps.shape == images.shape


def test_intensity_prediction(test_data):
    """This will produce a NumPy array of predicted wind intensity in knots

    dav_maps.shape -> (time)
    """
    images, tc = test_data

    dav_radius_km = 300  # radius of DAV calculation in km
    image_resolution = 8  # km per pixel
    start_basin = tc['basin'][0]

    radius_pixels = dav_radius_km/image_resolution
    cdav = centre_dav(images, radius_pixels)

    pred = model.predict(cdav, start_basin, min_periods=0)

    assert pred is not None
    assert len(pred) == len(images)
    assert np.all(np.isfinite(pred))
