import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import magixs


def _z(
    surface: optika.surfaces.Surface,
    radius: u.Quantity | na.AbstractScalar,
) -> u.Quantity | na.AbstractScalar:
    """
    The axial position of a surface at a given radius, in global coordinates.

    Parameters
    ----------
    surface
        The surface to evaluate.
    radius
        The radius at which to evaluate the surface.
    """
    point = na.Cartesian3dVectorArray(radius, 0 * u.mm, 0 * u.mm)
    point.z = surface.sag(point)
    return surface.transformation(point).z


@pytest.mark.parametrize(
    argnames="telescope",
    argvalues=[
        magixs.optics.Telescope(
            focal_length=2 * u.m,
            radius=72 * u.mm,
            length_primary=200 * u.mm,
            length_secondary=200 * u.mm,
        ),
        magixs.optics.Telescope(
            focal_length=1.09 * u.m,
            radius=73.06 * u.mm,
            length_primary=120 * u.mm,
            length_secondary=80 * u.mm,
        ),
        magixs.optics.Telescope(
            focal_length=na.linspace(2, 3, axis="f", num=2) * u.m,
            radius=50 * u.mm,
            length_primary=100 * u.mm,
            length_secondary=100 * u.mm,
        ),
    ],
)
class TestTelescope:

    def test_grazing_angle(self, telescope: magixs.optics.Telescope):
        result = telescope.grazing_angle
        assert np.all(result > 0 * u.deg)
        assert np.all(result < 1 * u.deg)
        # the two reflections of 2 alpha each turn the ray onto the focus
        assert np.allclose(
            np.tan(4 * result),
            telescope.radius / telescope.focal_length,
        )

    def test_surfaces(self, telescope: magixs.optics.Telescope):
        result = telescope.surfaces
        assert len(result) == 2
        primary, secondary = result
        assert isinstance(primary.sag, optika.sags.ParabolicSag)
        assert isinstance(secondary.sag, optika.sags.ConicSag)
        assert np.all(secondary.sag.conic < -1)
        for surface in result:
            assert isinstance(surface.material, optika.materials.Mirror)
            assert isinstance(surface.aperture, optika.apertures.AnnularAperture)
        # the mirrors meet at the radius of the telescope
        assert np.all(primary.aperture.radius_inner == telescope.radius)
        assert np.all(secondary.aperture.radius_outer == telescope.radius)
        assert np.all(primary.aperture.radius_outer > telescope.radius)
        assert np.all(secondary.aperture.radius_inner < telescope.radius)
        assert np.all(secondary.aperture.radius_inner > 0 * u.mm)

    def test_intersection(self, telescope: magixs.optics.Telescope):
        """
        Both mirrors pass through the radius of the telescope at the
        intersection plane, and their apertures end one mirror length away.
        """
        primary, secondary = telescope.surfaces
        r = telescope.radius
        assert np.allclose(_z(primary, r), 0 * u.mm, atol=1 * u.nm)
        assert np.allclose(_z(secondary, r), 0 * u.mm, atol=1 * u.nm)
        assert np.allclose(
            _z(primary, primary.aperture.radius_outer),
            -telescope.length_primary,
            atol=1 * u.nm,
        )
        assert np.allclose(
            _z(secondary, secondary.aperture.radius_inner),
            telescope.length_secondary,
            atol=1 * u.nm,
        )

    def test_slopes(self, telescope: magixs.optics.Telescope):
        """
        At the intersection plane the paraboloid is inclined to the axis by
        the grazing angle and the hyperboloid by three times it, so both graze
        the ray at the same angle.
        """
        primary, secondary = telescope.surfaces
        r = telescope.radius
        dr = 1 * u.um
        alpha = telescope.grazing_angle
        for surface, factor in [(primary, 1), (secondary, 3)]:
            dz = _z(surface, r + dr) - _z(surface, r - dr)
            slope = -2 * dr / dz
            assert np.allclose(slope, np.tan(factor * alpha), rtol=1e-5)

    def test_focus(self, telescope: magixs.optics.Telescope):
        """
        An on-axis annulus of rays must converge to a point at the focus.
        """
        aperture = telescope.primary.aperture
        radius = (aperture.radius_inner + aperture.radius_outer) / 2
        azimuth = na.linspace(0, 360, axis="azimuth", num=8, endpoint=False) * u.deg
        pupil = na.Cartesian2dVectorArray(
            x=radius * np.cos(azimuth),
            y=radius * np.sin(azimuth),
        )
        sensor = optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=10 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=na.Cartesian2dVectorArray(256, 256),
            transformation=na.transformations.Cartesian3dTranslation(
                z=telescope.focal_length,
            ),
        )
        grid = optika.vectors.ObjectVectorArray(
            wavelength=15 * u.AA,
            field=na.Cartesian2dVectorArray(0, 0) * u.deg,
            pupil=pupil,
        )
        # the rays start from a plane upstream of the whole primary
        sky = optika.surfaces.Surface(
            name="sky",
            transformation=na.transformations.Cartesian3dTranslation(
                z=-2 * telescope.length_primary,
            ),
        )
        system = optika.systems.SequentialSystem(
            object=sky,
            surfaces=telescope.surfaces,
            sensor=sensor,
            grid_input=grid,
        )
        rays = system.raytrace(
            wavelength=grid.wavelength,
            field=grid.field,
            pupil=grid.pupil,
            normalized_field=False,
            normalized_pupil=False,
        )
        final = rays.outputs[{system.axis_surface: ~0}]
        assert np.all(final.unvignetted)
        # the sensor is at the focus, so every ray lands on the axis
        assert np.all(final.position.xy.length < 1 * u.um)
        # and the rays never travel backward between surfaces
        z = rays.outputs.position.z
        assert np.all(np.diff(z, axis=system.axis_surface) > 0 * u.mm)
