import napari
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from morphometrics_engine import (
    measure_all_with_defaults,
    register_measurement,
)
from morphometrics_engine.types import LabelImage, LabelMeasurementTable


@register_measurement(name="measure_area", uses_intensity_image=False)
def measure_area(label_image: LabelImage) -> LabelMeasurementTable:
    region_props = regionprops_table(label_image, properties=("label", "area"))

    return pd.DataFrame(region_props).set_index("label")


if __name__ == "__main__":
    # make a simple label image
    label_im = np.zeros((10, 10, 10), dtype=int)
    label_im[5:10, 5:10, 5:10] = 1
    label_im[5:10, 0:5, 0:5] = 2
    label_im[0:5, 0:10, 0:10] = 3

    print(measure_area(label_im))

    print(measure_all_with_defaults(label_im))

    viewer = napari.Viewer()
    viewer.add_labels(label_im)

    viewer.window.add_plugin_dock_widget(
        plugin_name="morphometrics-engine",
        widget_name="Measure region properties",
    )

    napari.run()
