import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import magixs


@pytest.mark.parametrize(
    argnames="mirror",
    argvalues=[
        magixs.optics.SpectrometerMirror(
            radius_min=39.21 * u.mm,
            radius_max=41.91 * u.mm,
            length=80 * u.mm,
        ),
        magixs.optics.SpectrometerMirror(
            radius_min=na.linspace(30, 35, axis="r", num=2) * u.mm,
            radius_max=40 * u.mm,
            length=100 * u.mm,
        ),
    ],
)
class TestSpectrometerMirror:

    def test_parameter(self, mirror: magixs.optics.SpectrometerMirror):
        assert np.all(mirror.parameter > 0 * u.mm)

    def test_distance(self, mirror: magixs.optics.SpectrometerMirror):
        result = mirror.distance_max - mirror.distance_min
        assert np.allclose(result, mirror.length)
        assert np.all(mirror.distance_min > 0 * u.mm)

    def test_angle_center(self, mirror: magixs.optics.SpectrometerMirror):
        result = mirror.angle_center
        angle_min = np.arctan2(mirror.radius_max, mirror.distance_max)
        angle_max = np.arctan2(mirror.radius_min, mirror.distance_min)
        assert np.all(result > angle_min)
        assert np.all(result < angle_max)

    def test_surface(self, mirror: magixs.optics.SpectrometerMirror):
        """
        The ends of the mirror must lie on the surface, at their distances
        from the focus, which is the origin.
        """
        result = mirror.surface
        assert isinstance(result.sag, optika.sags.ParabolicSag)
        assert isinstance(result.material, optika.materials.Mirror)
        for radius, distance in [
            (mirror.radius_min, mirror.distance_min),
            (mirror.radius_max, mirror.distance_max),
        ]:
            point = na.Cartesian3dVectorArray(radius, 0 * u.mm, 0 * u.mm)
            point.z = result.sag(point)
            z = result.transformation(point).z
            assert np.allclose(z, distance, atol=1 * u.nm)
