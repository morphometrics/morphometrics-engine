from typing import Callable, List, Optional

from toolz import curry

_measurements = dict()


try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


@curry
def register_measurement(
    func: Callable,
    name: Optional[str] = None,
    uses_intensity_image: bool = True,
) -> Callable:
    _measurements[name] = {
        "type": "single",
        "callable": func,
        "choices": None,
        "intensity_image": uses_intensity_image,
    }
    return func


@curry
def register_measurement_set(
    func: Callable,
    choices: List[str],
    name: Optional[str] = None,
    uses_intensity_image: bool = True,
) -> Callable:
    _measurements[name] = {
        "type": "set",
        "callable": func,
        "choices": choices,
        "intensity_image": uses_intensity_image,
    }
    return func


def available_measurements() -> List[str]:
    """Get the names of all available measurements."""
    return [k for k in _measurements]
