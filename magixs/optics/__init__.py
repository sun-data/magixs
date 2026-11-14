"""Models of the optical system."""

from ._telescope_mirrors import AbstractTelescopeMirror, TelescopeMirror
from ._slots import AbstractSlot, Slot
from ._spectrometer_mirrors import AbstractSpectrometerMirror, SpectrometerMirror
from ._gratings import AbstractGrating, Grating
from ._sensors import Sensor
from ._instruments import AbstractInstrument, Instrument

__all__ = [
    "AbstractTelescopeMirror",
    "TelescopeMirror",
    "AbstractSlot",
    "Slot",
    "AbstractSpectrometerMirror",
    "SpectrometerMirror",
    "AbstractGrating",
    "Grating",
    "Sensor",
    "AbstractInstrument",
    "Instrument",
]
