import numpy as np

from morphometrics_engine._widget import QtMeasurementWidget


def test_qt_measurement_widget(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    _ = QtMeasurementWidget(viewer)
