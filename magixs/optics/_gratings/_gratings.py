import abc
import dataclasses
import astropy.units as u
import numpy as np
import named_arrays as na
import optika

__all__ = [
    "AbstractGrating",
    "Grating",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractGrating(
    optika.mixins.Printable,
):
    """
    An interface describing the planar varied-line-space reflection grating.

    The grating is described in a frame where its surface is the
    :math:`x`-:math:`y` plane and its rulings are parallel to the
    :math:`y` axis.
    Light travels toward :math:`+x`, and the origin is the point where the
    central ray of the beam lands.
    """

    @property
    @abc.abstractmethod
    def coefficients_spacing(self) -> dict[int, u.Quantity | na.AbstractScalar]:
        """
        The coefficients of the polynomial which gives the ruling spacing as
        a function of :math:`x`, keyed by the power of :math:`x`.
        """

    @property
    @abc.abstractmethod
    def diffraction_order(self) -> int | na.AbstractScalar:
        """
        The diffraction order which reaches the detector.
        """

    @property
    @abc.abstractmethod
    def position_start(self) -> u.Quantity | na.AbstractScalar:
        """
        The :math:`x` coordinate of the upstream edge of the ruled area.
        """

    @property
    @abc.abstractmethod
    def position_stop(self) -> u.Quantity | na.AbstractScalar:
        """
        The :math:`x` coordinate of the downstream edge of the ruled area.
        """

    @property
    @abc.abstractmethod
    def width(self) -> u.Quantity | na.AbstractScalar:
        """
        The width of the ruled area along the rulings.
        """

    @property
    @abc.abstractmethod
    def angle_grazing(self) -> u.Quantity | na.AbstractScalar:
        """
        The angle between the surface of the grating and the central ray of
        the incident beam.
        """

    @property
    @abc.abstractmethod
    def distance(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial distance from the midpoint between the two spectrometer
        mirrors to the origin of the grating.
        """

    @property
    @abc.abstractmethod
    def material(self) -> None | optika.materials.AbstractMaterial:
        """
        The material of the grating.
        """

    @property
    def _material(self) -> optika.materials.AbstractMaterial:
        material = self.material
        if material is None:
            material = optika.materials.Mirror()
        return material

    @property
    def spacing(self) -> u.Quantity | na.AbstractScalar:
        """
        The ruling spacing at the origin, where the central ray lands.
        """
        return self.coefficients_spacing[0]

    def angle_diffracted(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        r"""
        The angle between the surface of the grating and the central ray
        after it is diffracted.

        For a ray in the plane perpendicular to the rulings, which meets the
        grating at a grazing angle :math:`\alpha` and leaves it at a grazing
        angle :math:`\gamma`, the grating equation is
        :math:`\cos \alpha - \cos \gamma = m \lambda / d`.

        Parameters
        ----------
        wavelength
            The wavelength of the ray.
        """
        m = self.diffraction_order
        cos = np.cos(self.angle_grazing) - m * wavelength / self.spacing
        return np.arccos(cos.to(u.dimensionless_unscaled)).to(u.deg)

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        Represent this object as an :mod:`optika` surface.
        """
        x_start = self.position_start
        x_stop = self.position_stop
        return optika.surfaces.Surface(
            name="grating",
            material=self._material,
            aperture=optika.apertures.RectangularAperture(
                half_width=na.Cartesian2dVectorArray(
                    x=(x_stop - x_start) / 2,
                    y=self.width / 2,
                ),
                transformation=na.transformations.Cartesian3dTranslation(
                    x=(x_start + x_stop) / 2,
                ),
            ),
            rulings=optika.rulings.Rulings(
                spacing=optika.rulings.Polynomial1dRulingSpacing(
                    coefficients=self.coefficients_spacing,
                    normal=na.Cartesian3dVectorArray(1, 0, 0),
                ),
                diffraction_order=self.diffraction_order,
            ),
        )


@dataclasses.dataclass(eq=False, repr=False)
class Grating(
    AbstractGrating,
):
    """
    The planar varied-line-space reflection grating.

    The grating sits in the converging beam downstream of the second
    spectrometer mirror and is struck at grazing incidence.
    A plane grating with evenly-spaced rulings would blur a converging
    beam, since every ray meets it at a different angle.
    The spacing of the rulings varies along the grating to cancel this,
    so that each wavelength comes to a focus on the detector.

    The grating is described in a frame where its surface is the
    :math:`x`-:math:`y` plane and its rulings are parallel to the
    :math:`y` axis.
    Light travels toward :math:`+x`, and the origin is the point where the
    central ray of the beam lands.
    """

    coefficients_spacing: dict[int, u.Quantity | na.AbstractScalar] = (
        dataclasses.MISSING
    )
    """
    The coefficients of the polynomial which gives the ruling spacing as a
    function of :math:`x`, keyed by the power of :math:`x`.
    """

    diffraction_order: int | na.AbstractScalar = 1
    """
    The diffraction order which reaches the detector.
    """

    position_start: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The :math:`x` coordinate of the upstream edge of the ruled area.
    """

    position_stop: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The :math:`x` coordinate of the downstream edge of the ruled area.
    """

    width: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The width of the ruled area along the rulings.
    """

    angle_grazing: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle between the surface of the grating and the central ray of the
    incident beam.
    """

    distance: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The axial distance from the midpoint between the two spectrometer
    mirrors to the origin of the grating.
    """

    material: None | optika.materials.AbstractMaterial = None
    """
    The material of the grating.
    If :obj:`None`, the default, a perfect :class:`optika.materials.Mirror`.
    """
