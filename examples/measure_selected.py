"""This example shows how to register measurements
and make measurements with a subset
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from morphometrics_engine import (
    measure_selected,
    register_measurement,
    register_measurement_set,
)
from morphometrics_engine.types import (
    IntensityImage,
    LabelImage,
    LabelMeasurementTable,
)


@register_measurement(name="measure_area", uses_intensity_image=False)
def measure_area(label_image: LabelImage) -> LabelMeasurementTable:
    region_props = regionprops_table(label_image, properties=("label", "area"))

    return pd.DataFrame(region_props).set_index("label")


@register_measurement_set(
    name="region_props",
    uses_intensity_image=True,
    choices=["intensity", "centroid"],
)
def measure_region_props(
    label_image: LabelImage,
    intensity_image: IntensityImage,
    intensity: bool = True,
    centroid: bool = True,
) -> LabelMeasurementTable:
    base_measurement = regionprops_table(label_image, properties=("label",))
    base_table = pd.DataFrame(base_measurement).set_index("label")

    if intensity is True:
        area_measurements = regionprops_table(
            label_image,
            intensity_image=intensity_image,
            properties=("label", "intensity_mean"),
        )
        area_table = pd.DataFrame(area_measurements).set_index("label")
        base_table = pd.concat([base_table, area_table], axis=1)

    if centroid is True:
        centroid_measurements = regionprops_table(
            label_image, properties=("label", "centroid")
        )
        centroid_table = pd.DataFrame(centroid_measurements).set_index("label")
        base_table = pd.concat([base_table, centroid_table], axis=1)

    return base_table


if __name__ == "__main__":
    # make a simple label image
    label_im = np.zeros((10, 10, 10), dtype=int)
    label_im[5:10, 5:10, 5:10] = 1
    label_im[5:10, 0:5, 0:5] = 2
    label_im[0:5, 0:10, 0:10] = 3

    # make the intensity image
    intensity_image = np.random.random((10, 10, 10))

    measurement_selection = [
        "measure_area",  # single measurements can be specified with string
        {
            "name": "region_props",  # sets are specified with a dictionary
            "choices": {  # choices can be specified with a dictionary
                "intensity": True,  # the key is the name of the choice
                "centroid": False,  # the value is True/False
            },
        },
    ]
    print(
        measure_selected(
            label_image=label_im,
            intensity_image=intensity_image,
            measurement_selection=measurement_selection,
        )
    )
