import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import magixs


@pytest.mark.parametrize(
    argnames="grating",
    argvalues=[
        magixs.optics.Grating(
            coefficients_spacing={
                0: 476 * u.nm,
                1: -2 * u.nm / u.mm,
            },
            diffraction_order=1,
            position_start=-32 * u.mm,
            position_stop=41 * u.mm,
            width=25 * u.mm,
            angle_grazing=2 * u.deg,
            distance=174 * u.mm,
        ),
    ],
)
class TestGrating:

    def test_spacing(self, grating: magixs.optics.Grating):
        assert grating.spacing == grating.coefficients_spacing[0]

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            15 * u.AA,
            na.linspace(6, 24, axis="wavelength", num=4) * u.AA,
        ],
    )
    def test_angle_diffracted(
        self,
        grating: magixs.optics.Grating,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = grating.angle_diffracted(wavelength)
        # the first order leaves the grating more steeply than it arrived
        assert np.all(result > grating.angle_grazing)
        # and satisfies the grating equation
        difference = np.cos(grating.angle_grazing) - np.cos(result)
        assert np.allclose(difference, wavelength / grating.spacing)

    def test_surface(self, grating: magixs.optics.Grating):
        result = grating.surface
        assert isinstance(result, optika.surfaces.Surface)
        assert isinstance(result.material, optika.materials.Mirror)
        # the ruled area spans the start and stop positions
        wire = result.aperture.wire()
        assert np.isclose(wire.x.min(), grating.position_start)
        assert np.isclose(wire.x.max(), grating.position_stop)
        assert np.isclose(wire.y.max() - wire.y.min(), grating.width)
        # the spacing at the origin is the constant coefficient
        position = na.Cartesian3dVectorArray() * u.mm
        normal = result.sag.normal(position)
        spacing = result.rulings.spacing_(position, normal).length
        assert np.isclose(spacing, grating.spacing)
