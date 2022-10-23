import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from morphometrics_engine import (
    _measurements,
    available_measurements,
    register_measurement,
)
from morphometrics_engine.types import LabelImage, LabelMeasurementTable


def test_register_measurement():
    """Test registering a single measurement function"""

    measurement_name = "measure_area"

    @register_measurement(name=measurement_name, uses_intensity_image=False)
    def measure_area(label_image: LabelImage) -> LabelMeasurementTable:
        region_props = regionprops_table(
            label_image, properties=("label", "area")
        )

        return pd.DataFrame(region_props).set_index("label")

    assert measurement_name in _measurements
    measurement_entry = _measurements[measurement_name]
    assert measurement_entry["type"] == "single"
    assert measurement_entry["intensity_image"] is False
    assert measurement_entry["callable"] is measure_area

    # there should be just the measurement we added in available_measurements
    all_measurement_names = available_measurements()
    np.testing.assert_array_equal([measurement_name], all_measurement_names)
