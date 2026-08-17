"""
Tests used to check if package is working correctly.


Test data:
    MERGIR data is distributed under the Creative Commons
    Attribution 4.0 International (CC BY 4.0) licence.

    IBTrACS data is produced by NOAA/NCEI and is a U.S.
    federal government work, and is public domain in the
    United States.

Dependencies:
    numpy
    tc-dav

Author: Joshua May (josh.w.may@gmail.com)
"""
from pathlib import Path
import json
import pytest
import numpy as np

# Loading TC information and images modules
from dav.utils import IbtracsReader, MergirReader

# Loading DAV map generation modules
from dav.generate import dav

# Loading TC intensity estimation modules
from dav.generate import centre_dav
from dav.intensity import model as intensity_model

# Loading wind radii utils and estimation modules
from dav.utils import get_dav_profile, get_tc_age
from dav.wind_radii import model as wind_radii_model


@pytest.fixture(scope="session")
def example_tc_data():
    """This will produce a NumPy array of Brightness temperature values

    images.shape -> (time, height, width)
    """
    ibtracs_fn = Path(__file__).parent / "example_ibtracs.nc"
    mergir_dir = Path(__file__).parent / "mergir_data"

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


@pytest.fixture(scope="session")
def dav_maps(example_tc_data):
    images, _ = example_tc_data

    dav_radius_km = 300
    image_resolution = 8

    radius_pixels = dav_radius_km / image_resolution

    return dav(images, radius_pixels)


def test_tc_data(example_tc_data):
    images, tc = example_tc_data
    lengths = list(map(len, (tc['iso_time'],
                             tc['lat'],
                             tc['lon'],
                             tc['basin'],
                             tc['usa_wind'])))

    assert len(set(lengths)) == 1
    assert len(images) == len(tc['iso_time'])
    assert images.ndim == 3
    assert images.dtype == np.float32


def test_dav_maps(example_tc_data, dav_maps):
    """This will produce a NumPy array of DAV maps

    dav_maps.shape -> (time, height, width)
    """
    images, _ = example_tc_data

    assert dav_maps.shape == images.shape
    assert np.all(np.isfinite(dav_maps))


def test_intensity_prediction(example_tc_data):
    """This will produce a NumPy array of predicted wind intensity in knots

    dav_maps.shape -> (time)
    """
    images, tc = example_tc_data

    dav_radius_km = 300  # radius of DAV calculation in km
    image_resolution = 8  # km per pixel
    start_basin = tc['basin'][0]

    radius_pixels = dav_radius_km/image_resolution
    cdav = centre_dav(images, radius_pixels)

    pred = intensity_model.predict(cdav, start_basin, min_periods=0)

    assert pred is not None
    assert len(pred) == len(images)
    assert np.all(np.isfinite(pred))


def test_wind_radii_estimation(example_tc_data, dav_maps):
    """This will produce a NumPy array of predicted wind radii in km

    wind_radii.shape -> (time)
    """
    _, tc = example_tc_data

    model_json = Path(__file__).parent / "MODEL_NA_symmetric_r34.json"
    with open(model_json, 'r') as file:
        model_json = json.load(file)

    profile = get_dav_profile(dav_maps, 75)
    # sst can be retrieved from elsewhere, during experiments we used ERA5 data
    data = {"profile": profile,
            "age": get_tc_age(tc['usa_wind'], samples_per_hour=1/3),
            "sst": np.array([29.73, 29.73, 29.26, 29.11, np.nan]),
            "wind": tc['usa_wind']}

    predictions = wind_radii_model.predict_from_json(model_json, data)

    assert len(predictions) == 5
    assert np.all(np.isfinite(predictions[:4]))
    assert not np.isfinite(predictions[-1])
