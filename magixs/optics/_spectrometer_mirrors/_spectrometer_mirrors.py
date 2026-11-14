import abc
import dataclasses
import astropy.units as u
import numpy as np
import named_arrays as na
import optika

__all__ = [
    "AbstractSpectrometerMirror",
    "SpectrometerMirror",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSpectrometerMirror(
    optika.mixins.Printable,
):
    """
    An interface describing one of the two identical spectrometer mirrors:
    a segment of a paraboloid struck at grazing incidence.

    The mirror is described in a frame where its focus is the origin and
    its axis is the :math:`z` axis.
    The segment lies at positive :math:`z` and widens away from the focus.
    """

    @property
    @abc.abstractmethod
    def radius_min(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the mirror at its narrow end, the one nearer the focus.
        """

    @property
    @abc.abstractmethod
    def radius_max(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the mirror at its wide end, the one farther from the
        focus.
        """

    @property
    @abc.abstractmethod
    def length(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial length of the mirror.
        """

    @property
    @abc.abstractmethod
    def material(self) -> None | optika.materials.AbstractMaterial:
        """
        The material of the mirror.
        """

    @property
    def _material(self) -> optika.materials.AbstractMaterial:
        material = self.material
        if material is None:
            material = optika.materials.Mirror()
        return material

    @property
    def parameter(self) -> u.Quantity | na.AbstractScalar:
        r"""
        The distance :math:`p` from the vertex of the paraboloid to its focus.

        The paraboloid is :math:`r^2 = 4 p (z + p)`, so the radii of the two
        ends of a segment of length :math:`L` satisfy
        :math:`r_\text{max}^2 - r_\text{min}^2 = 4 p L`.
        """
        r_min = self.radius_min
        r_max = self.radius_max
        return (np.square(r_max) - np.square(r_min)) / (4 * self.length)

    def _distance(
        self,
        radius: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        The axial distance from the focus to the point on the paraboloid
        with the given radius.

        Parameters
        ----------
        radius
            The radius of the point on the paraboloid.
        """
        p = self.parameter
        return np.square(radius) / (4 * p) - p

    @property
    def distance_min(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial distance from the focus to the narrow end of the mirror.
        """
        return self._distance(self.radius_min)

    @property
    def distance_max(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial distance from the focus to the wide end of the mirror.
        """
        return self._distance(self.radius_max)

    @property
    def radius_center(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius halfway between the two ends of the mirror.
        """
        return (self.radius_min + self.radius_max) / 2

    @property
    def angle_center(self) -> u.Quantity | na.AbstractScalar:
        """
        The angle between the axis and the ray which joins the focus to the
        point on the mirror at :attr:`radius_center`.
        """
        r = self.radius_center
        return np.arctan2(r, self._distance(r))

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        Represent this object as an :mod:`optika` surface.

        The surface opens toward :math:`+z`, so it collimates the light
        which diverges from its focus at the origin.
        """
        p = self.parameter
        return optika.surfaces.Surface(
            name="spectrometer mirror",
            sag=optika.sags.ParabolicSag(focal_length=p),
            material=self._material,
            aperture=optika.apertures.AnnularAperture(
                radius_inner=self.radius_min,
                radius_outer=self.radius_max,
            ),
            transformation=na.transformations.Cartesian3dTranslation(z=-p),
        )


@dataclasses.dataclass(eq=False, repr=False)
class SpectrometerMirror(
    AbstractSpectrometerMirror,
):
    """
    One of the two identical spectrometer mirrors: a segment of a paraboloid
    struck at grazing incidence.

    The first mirror is confocal with the telescope and collimates the
    light diverging from the slot.
    The second, which faces the first, focuses the collimated beam toward
    the detector.
    Together they relay the slot at unit magnification.

    The mirror is fixed by the radii of its two ends and by its length.
    It is described in a frame where its focus is the origin and its axis
    is the :math:`z` axis, with the segment at positive :math:`z`.
    """

    radius_min: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the mirror at its narrow end, the one nearer the focus.
    """

    radius_max: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the mirror at its wide end, the one farther from the
    focus.
    """

    length: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The axial length of the mirror.
    """

    material: None | optika.materials.AbstractMaterial = None
    """
    The material of the mirror.
    If :obj:`None`, the default, a perfect :class:`optika.materials.Mirror`.
    """
