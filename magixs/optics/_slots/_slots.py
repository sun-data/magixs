import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractSlot",
    "Slot",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSlot(
    optika.mixins.Printable,
):
    """
    An interface describing the slot at the focus of the telescope.
    """

    @property
    @abc.abstractmethod
    def half_width(self) -> na.AbstractCartesian2dVectorArray:
        """
        The half-width of the slot along the dispersion direction, :math:`x`,
        and along the cross-dispersion direction, :math:`y`.
        """

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        Represent this object as an :mod:`optika` surface, centered on the
        origin and perpendicular to the :math:`z` axis.
        """
        return optika.surfaces.Surface(
            name="slot",
            aperture=optika.apertures.RectangularAperture(
                half_width=self.half_width,
            ),
            is_field_stop=True,
        )


@dataclasses.dataclass(eq=False, repr=False)
class Slot(
    AbstractSlot,
):
    """
    The slot at the focus of the telescope.

    The slot is the field stop of the instrument.
    It is much wider than the image of a spectral line,
    so the instrument forms an overlapping image of the slot in every
    spectral line instead of a spectrum of a single line of sight.
    """

    half_width: na.AbstractCartesian2dVectorArray = 0 * u.mm
    """
    The half-width of the slot along the dispersion direction, :math:`x`,
    and along the cross-dispersion direction, :math:`y`.
    """
