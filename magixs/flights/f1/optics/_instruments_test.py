import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import magixs


def _trace(
    instrument: magixs.optics.Instrument,
    wavelength: u.Quantity | na.AbstractScalar,
    field: na.AbstractCartesian2dVectorArray,
) -> optika.rays.RayVectorArray:
    """
    Trace a grid of rays which fills the illuminated sector of the telescope
    and return the rays at every surface, in global coordinates.

    Parameters
    ----------
    instrument
        The instrument to trace rays through.
    wavelength
        The wavelengths of the rays.
    field
        The angles of the rays on the sky.
    """
    aperture = instrument.telescope_mirror.primary.aperture
    radius = na.linspace(
        start=aperture.radius_inner,
        stop=aperture.radius_outer,
        axis="radius",
        num=7,
    )[dict(radius=slice(1, ~0))]
    azimuth = na.linspace(
        start=aperture.angle_start,
        stop=aperture.angle_stop,
        axis="azimuth",
        num=9,
    )[dict(azimuth=slice(1, ~0))]
    pupil = na.Cartesian2dVectorArray(
        x=radius * np.cos(azimuth),
        y=radius * np.sin(azimuth),
    )
    system = instrument.system
    rays = system.raytrace(
        wavelength=wavelength,
        field=field,
        pupil=pupil,
        normalized_field=False,
        normalized_pupil=False,
    )
    return rays.outputs


@pytest.mark.parametrize(
    argnames="instrument",
    argvalues=[
        magixs.flights.f1.optics.design(),
    ],
)
class TestDesign:

    def test_length(self, instrument: magixs.optics.Instrument):
        """
        Compare the layout to Figure 2 of Champey et al. (2022).
        """
        z_spectrometer = instrument.z_slot + instrument.distance_spectrometer
        distance_sensor = instrument.position_sensor.z - z_spectrometer
        assert np.isclose(instrument.length, 2513 * u.mm, atol=2 * u.mm)
        assert np.isclose(distance_sensor, 645 * u.mm, atol=2 * u.mm)

    def test_spectrometer_mirrors(self, instrument: magixs.optics.Instrument):
        """
        The two spectrometer mirrors face each other across a small gap.
        """
        mirror = instrument.spectrometer_mirror
        gap = 2 * (instrument.distance_spectrometer - mirror.distance_max)
        assert gap > 0 * u.mm
        assert gap < mirror.length

    def test_image(self, instrument: magixs.optics.Instrument):
        """
        Every ray from the center of the field must reach the sensor,
        every wavelength must come to a focus,
        and the central wavelength must land on the center of the sensor.
        """
        system = instrument.system
        wavelength_center = instrument.sensor.wavelength_center
        wavelength = na.stack(
            arrays=[6 * u.AA, wavelength_center, 24 * u.AA],
            axis="wavelength",
        )
        rays = _trace(
            instrument=instrument,
            wavelength=wavelength,
            field=na.Cartesian2dVectorArray(0, 0) * u.arcmin,
        )
        assert np.all(rays.unvignetted)

        # past the object plane, which is the intersection plane of the
        # telescope, the rays never travel backward between surfaces
        z = rays.position.z[{system.axis_surface: slice(1, None)}]
        assert np.all(np.diff(z, axis=system.axis_surface) > 0 * u.mm)

        # the beam fits on the ruled area of the grating
        grating = system.surfaces[~0]
        index_grating = {system.axis_surface: ~1}
        position_grating = rays.position[index_grating]
        position_grating = grating.transformation.inverse(position_grating)
        assert np.all(position_grating.x > instrument.grating.position_start)
        assert np.all(position_grating.x < instrument.grating.position_stop)

        rays = rays[{system.axis_surface: ~0}]
        position = system.sensor.transformation.inverse(rays.position)
        axis = ("radius", "azimuth")
        width_pixel = instrument.sensor.width_pixel

        # the spots are smaller than a pixel
        assert np.all(position.x.std(axis) < width_pixel)
        assert np.all(position.y.std(axis) < width_pixel)

        # the central wavelength lands on the center of the sensor
        center = position.mean(axis)[dict(wavelength=1)]
        assert np.abs(center.x) < width_pixel
        assert np.abs(center.y) < width_pixel

        # the whole passband fits on the sensor
        width_sensor = system.sensor.num_pixel.x * width_pixel
        assert np.all(np.abs(position.x) < width_sensor / 2)

        # and longer wavelengths land at larger x
        x = position.x.mean(axis)
        assert np.all(np.diff(x, axis="wavelength") > 0 * u.mm)

    def test_plate_scale(self, instrument: magixs.optics.Instrument):
        """
        Compare the plate scale across the dispersion to the 2.8 arcseconds
        per pixel of Champey et al. (2022).
        """
        system = instrument.system
        angle = 5 * u.arcmin
        field = na.Cartesian2dVectorArray(
            x=0 * u.arcmin,
            y=na.linspace(-angle, angle, axis="field", num=2),
        )
        rays = _trace(
            instrument=instrument,
            wavelength=15 * u.AA,
            field=field,
        )
        rays = rays[{system.axis_surface: ~0}]
        position = system.sensor.transformation.inverse(rays.position)
        y = position.y.mean(("radius", "azimuth"))
        separation = np.abs(np.diff(y, axis="field")) / instrument.sensor.width_pixel
        result = (2 * angle / separation).to(u.arcsec)
        assert np.allclose(result, 2.8 * u.arcsec, rtol=0.02)

    def test_dispersion(self, instrument: magixs.optics.Instrument):
        """
        Compare the dispersion to the 10 to 15 milliangstroms per pixel
        measured by Athiray et al. (2021).
        """
        system = instrument.system
        delta = 0.1 * u.AA
        wavelength = 15 * u.AA + na.linspace(-delta, delta, axis="wavelength", num=2)
        rays = _trace(
            instrument=instrument,
            wavelength=wavelength,
            field=na.Cartesian2dVectorArray(0, 0) * u.arcmin,
        )
        rays = rays[{system.axis_surface: ~0}]
        position = system.sensor.transformation.inverse(rays.position)
        x = position.x.mean(("radius", "azimuth"))
        separation = np.diff(x, axis="wavelength") / instrument.sensor.width_pixel
        result = (2 * delta / separation).to(u.mAA)
        assert np.all(result > 10 * u.mAA)
        assert np.all(result < 15 * u.mAA)

    def test_system(self, instrument: magixs.optics.Instrument):
        """
        The default grid, in normalized coordinates, must reach the sensor.
        """
        rays = instrument.system.rayfunction_default.outputs
        assert np.any(rays.unvignetted)
        center = dict(field_x=2, field_y=2)
        assert np.all(np.any(rays.unvignetted[center], axis=("pupil_x", "pupil_y")))
