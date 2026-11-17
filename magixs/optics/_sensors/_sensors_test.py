import pytest
import astropy.units as u
import optika
import magixs


@pytest.mark.parametrize(
    argnames="sensor",
    argvalues=[
        magixs.optics.Sensor(
            wavelength_center=17 * u.AA,
        ),
    ],
)
class TestSensor:

    def test_surface(self, sensor: magixs.optics.Sensor):
        result = sensor.surface
        assert isinstance(result, optika.sensors.ImagingSensor)
        assert result.width_pixel == sensor.width_pixel
        assert result.num_pixel.x == 2048
        # the long axis of the active area is the dispersion direction
        assert result.num_pixel.x > result.num_pixel.y
