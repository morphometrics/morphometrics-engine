import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from morphometrics_engine import register_measurement, register_measurement_set
from morphometrics_engine._widget import QtMeasurementWidget
from morphometrics_engine.types import (
    IntensityImage,
    LabelImage,
    LabelMeasurementTable,
)


def test_qt_measurement_widget(make_napari_viewer):
    @register_measurement(name="measure_area", uses_intensity_image=False)
    def measure_area(label_image: LabelImage) -> LabelMeasurementTable:
        region_props = regionprops_table(
            label_image, properties=("label", "area")
        )

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
        base_measurement = regionprops_table(
            label_image, properties=("label",)
        )
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
            centroid_table = pd.DataFrame(centroid_measurements).set_index(
                "label"
            )
            base_table = pd.concat([base_table, centroid_table], axis=1)

        return base_table

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(np.random.random((10, 10, 10)))

    # create our widget, passing in the viewer
    measurement_widget = QtMeasurementWidget(viewer)

    label_im = np.zeros((10, 10, 10), dtype=int)
    label_im[5:10, 5:10, 5:10] = 1
    label_im[5:10, 0:5, 0:5] = 2
    label_im[0:5, 0:10, 0:10] = 3

    labels_layer = viewer.add_labels(label_im)

    # set the label image
    measurement_widget._select_layers(
        intensity_image=image_layer, label_image=labels_layer
    )

    # select the area measurement
    measurement_widget.measurement_widgets[0]._check_box.setChecked(True)

    # select all of the choices in regionprops
    regionprops_widget = measurement_widget.measurement_widgets[1]
    for widget in regionprops_widget._choice_widgets:
        widget._check_box.setChecked(True)

    # simulate running the measurement
    measurement_widget._run()

    # check that the measurements were added
    feature_table = labels_layer.features
    assert "area" in feature_table.columns
    assert "intensity_mean" in feature_table.columns
    assert "centroid-0" in feature_table.columns
    assert "centroid-1" in feature_table.columns
    assert "centroid-2" in feature_table.columns
