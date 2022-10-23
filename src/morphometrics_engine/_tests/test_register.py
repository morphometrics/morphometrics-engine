import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from morphometrics_engine import (
    _measurements,
    available_measurements,
    measure_all_with_defaults,
    measure_selected,
    register_measurement,
    register_measurement_set,
)
from morphometrics_engine.types import LabelImage, LabelMeasurementTable


def simple_3d_label_image() -> LabelImage:
    label_im = np.zeros((10, 10, 10), dtype=int)
    label_im[5:10, 5:10, 5:10] = 1
    label_im[5:10, 0:5, 0:5] = 2
    label_im[0:5, 0:10, 0:10] = 3

    return label_im


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

    label_image = simple_3d_label_image()
    label_table = measure_all_with_defaults(label_image)

    assert len(label_table) == 3
    assert "area" in label_table.columns


def test_register_measurement_set():
    measurement_name = "region_props"

    @register_measurement_set(
        name=measurement_name,
        uses_intensity_image=False,
        choices=["area", "centroid"],
    )
    def measure_region_props(
        label_image: LabelImage, area: bool = True, centroid: bool = True
    ) -> LabelMeasurementTable:
        base_measurement = regionprops_table(
            label_image, properties=("label",)
        )
        base_table = pd.DataFrame(base_measurement).set_index("label")

        if area is True:
            area_measurements = regionprops_table(
                label_image, properties=("label", "area")
            )
            area_table = pd.DataFrame(area_measurements).set_index("label")
            base_table = pd.concat([base_table, area_table], axis=1)

        if centroid is True:
            centroid_measurements = regionprops_table(
                label_image, properties=("label", "centroid")
            )
            centroid_table = pd.DataFrame(centroid_measurements).set_index(
                "label"
            )
            base_table = pd.concat([base_table, centroid_table], axis=1)

        return base_table

    assert measurement_name in _measurements
    measurement_entry = _measurements[measurement_name]
    assert measurement_entry["type"] == "set"
    assert measurement_entry["intensity_image"] is False
    assert measurement_entry["callable"] is measure_region_props
    np.testing.assert_array_equal(
        measurement_entry["choices"], ["area", "centroid"]
    )

    # there should be just the measurement we added in available_measurements
    all_measurement_names = available_measurements()
    np.testing.assert_array_equal([measurement_name], all_measurement_names)

    # measure all
    label_image = simple_3d_label_image()
    label_table = measure_all_with_defaults(label_image)
    assert len(label_table) == 3
    assert "area" in label_table.columns
    assert "centroid-0" in label_table.columns
    assert "centroid-1" in label_table.columns
    assert "centroid-2" in label_table.columns

    # measure selected
    measurement_selection = [
        {
            "name": measurement_name,
            "choices": {"area": True, "centroid": False},
        }
    ]
    selected_measurements = measure_selected(
        label_image, measurement_selection=measurement_selection
    )
    assert len(selected_measurements) == 3
    assert "area" in selected_measurements.columns
    assert "centroid-0" not in selected_measurements.columns
    assert "centroid-1" not in selected_measurements.columns
    assert "centroid-2" not in selected_measurements.columns
