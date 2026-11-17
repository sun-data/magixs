import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import magixs


@pytest.mark.parametrize(
    argnames="slot",
    argvalues=[
        magixs.optics.Slot(
            half_width=na.Cartesian2dVectorArray(3.84, 10.6) * u.mm / 2,
        ),
    ],
)
class TestSlot:

    def test_surface(self, slot: magixs.optics.Slot):
        result = slot.surface
        assert isinstance(result, optika.surfaces.Surface)
        assert isinstance(result.aperture, optika.apertures.RectangularAperture)
        assert np.all(result.aperture.half_width == slot.half_width)
        assert result.is_field_stop
