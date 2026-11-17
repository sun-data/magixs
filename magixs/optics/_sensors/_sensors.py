import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import msfc_ccd

__all__ = [
    "Sensor",
]


@dataclasses.dataclass(eq=False, repr=False)
class Sensor(
    msfc_ccd.TeledyneCCD230,
):
    """
    The CCD sensor which records the dispersed image of the slot.

    The sensor is perpendicular to the central ray of the wavelength which
    lands on its center, and the grating disperses along its :math:`x` axis.
    """

    wavelength_center: u.Quantity | na.AbstractScalar = 0 * u.AA
    """
    The wavelength of the on-axis ray which lands on the center of the
    sensor.
    """

    @property
    def surface(self) -> optika.sensors.ImagingSensor:
        """
        Represent this object as an :mod:`optika` surface, centered on the
        origin and perpendicular to the :math:`z` axis.
        """
        return optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=self.width_pixel,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=self.num_pixel_active,
            read_noise=self.readout_noise,
            material=self.material,
            aperture_mechanical=optika.apertures.RectangularAperture(
                half_width=self.width_package / 2,
            ),
        )
