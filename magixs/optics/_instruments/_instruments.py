from typing import Any
import abc
import dataclasses
import functools
import astropy.units as u
import numpy as np
import named_arrays as na
import optika
from .._telescope_mirrors import AbstractTelescopeMirror
from .._slots import AbstractSlot
from .._spectrometer_mirrors import AbstractSpectrometerMirror
from .._gratings import AbstractGrating
from .._sensors import Sensor

__all__ = [
    "AbstractInstrument",
    "Instrument",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInstrument(
    optika.mixins.Printable,
):
    """
    An interface describing the entire optical system.

    The optic axis is the :math:`z` axis, and the origin is the center of
    the intersection plane of the telescope.
    The grating folds the beam in the :math:`x`-:math:`z` plane,
    so the illuminated sector of the telescope is centered on the
    :math:`x` axis and the spectrum is dispersed along :math:`x`.
    """

    @property
    @abc.abstractmethod
    def telescope_mirror(self) -> AbstractTelescopeMirror:
        """
        A model of the mirror of the Wolter-I telescope.
        """

    @property
    @abc.abstractmethod
    def slot(self) -> AbstractSlot:
        """
        A model of the slot at the focus of the telescope.
        """

    @property
    @abc.abstractmethod
    def spectrometer_mirror(self) -> AbstractSpectrometerMirror:
        """
        A model of the two spectrometer mirrors, which are identical.
        """

    @property
    @abc.abstractmethod
    def grating(self) -> AbstractGrating:
        """
        A model of the diffraction grating.
        """

    @property
    @abc.abstractmethod
    def sensor(self) -> Sensor:
        """
        A model of the CCD sensor.
        """

    @property
    @abc.abstractmethod
    def distance_spectrometer(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial distance from the slot to the midpoint between the two
        spectrometer mirrors.
        This is also the axial distance from that midpoint to the focus of
        the second spectrometer mirror.
        """

    @property
    @abc.abstractmethod
    def wavelength(self) -> None | u.Quantity | na.AbstractScalar:
        """
        A default grid of wavelengths to trace through the system.
        """

    @property
    @abc.abstractmethod
    def field(self) -> None | na.AbstractCartesian2dVectorArray:
        """
        A default grid of field positions to trace through the system.
        """

    @property
    @abc.abstractmethod
    def pupil(self) -> None | na.AbstractCartesian2dVectorArray:
        """
        A default grid of pupil positions to trace through the system.
        """

    @property
    @abc.abstractmethod
    def kwargs_plot(self) -> None | dict[str, Any]:
        """
        Extra keyword arguments used to plot the optical system.
        """

    @property
    def z_slot(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial position of the slot, which is at the focus of the
        telescope.
        """
        return self.telescope_mirror.focal_length

    @property
    def z_focus_spectrometer(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial position of the focus of the second spectrometer mirror,
        where the beam would converge if the grating were removed.
        """
        return self.z_slot + 2 * self.distance_spectrometer

    @property
    def position_grating(self) -> na.Cartesian3dVectorArray:
        """
        The position of the origin of the grating.

        The origin of the grating is on the central ray of the beam: the
        one which leaves the second spectrometer mirror halfway between its
        ends and heads for its focus.
        The beam crosses the axis at the slot, so past the slot it is on the
        opposite side of the axis from the illuminated sector of the
        telescope.
        """
        z = self.z_slot + self.distance_spectrometer + self.grating.distance
        angle = self.spectrometer_mirror.angle_center
        x = -(self.z_focus_spectrometer - z) * np.tan(angle)
        return na.Cartesian3dVectorArray(x=x, y=0 * x, z=z)

    @property
    def distance_grating_focus(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the origin of the grating to the focus of the
        second spectrometer mirror, measured along the central ray.
        """
        z = self.position_grating.z
        angle = self.spectrometer_mirror.angle_center
        return (self.z_focus_spectrometer - z) / np.cos(angle)

    @property
    def angle_grating(self) -> u.Quantity | na.AbstractScalar:
        """
        The angle between the axis and the surface of the grating.

        The central ray climbs toward the axis and the grating lies beneath
        it, so the grating is steeper than the ray by the grazing angle.
        """
        angle = self.spectrometer_mirror.angle_center
        return angle + self.grating.angle_grazing

    def angle_diffracted(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        The angle between the axis and the central ray after it is
        diffracted by the grating.

        Parameters
        ----------
        wavelength
            The wavelength of the ray.
        """
        return self.angle_grating + self.grating.angle_diffracted(wavelength)

    @property
    def position_sensor(self) -> na.Cartesian3dVectorArray:
        """
        The position of the center of the sensor.

        The sensor is as far from the grating as the focus of the second
        spectrometer mirror is, measured along the central ray of the
        wavelength which lands on its center.
        """
        angle = self.angle_diffracted(self.sensor.wavelength_center)
        direction = na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0 * np.sin(angle),
            z=np.cos(angle),
        )
        return self.position_grating + self.distance_grating_focus * direction

    @property
    def length(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial distance from the entrance of the telescope to the center
        of the sensor.
        """
        return self.telescope_mirror.length_primary + self.position_sensor.z

    @property
    def surface_spectrometer_entrance(self) -> optika.surfaces.Surface:
        """
        A transparent plane just upstream of the first spectrometer mirror.

        A ray leaving the slot crosses the paraboloid of the first
        spectrometer mirror twice: once next to its vertex, which is just
        behind the slot, and once on the mirror itself.
        The raytrace takes the nearer crossing, so the rays are first
        carried to this plane, from which the mirror is the nearer one.
        """
        distance = 0.9 * self.spectrometer_mirror.distance_min
        return optika.surfaces.Surface(
            name="spectrometer entrance",
            transformation=na.transformations.Cartesian3dTranslation(
                z=self.z_slot + distance,
            ),
        )

    @property
    def surface_spectrometer_mirror_1(self) -> optika.surfaces.Surface:
        """
        The first spectrometer mirror, which is confocal with the telescope
        and collimates the light diverging from the slot.
        """
        surface = self.spectrometer_mirror.surface
        return dataclasses.replace(
            surface,
            name="spectrometer mirror 1",
            transformation=na.transformations.TransformationList(
                [
                    surface.transformation,
                    na.transformations.Cartesian3dTranslation(z=self.z_slot),
                ]
            ),
        )

    @property
    def surface_spectrometer_mirror_2(self) -> optika.surfaces.Surface:
        """
        The second spectrometer mirror, which faces the first and focuses
        the collimated beam.
        """
        surface = self.spectrometer_mirror.surface
        return dataclasses.replace(
            surface,
            name="spectrometer mirror 2",
            transformation=na.transformations.TransformationList(
                [
                    surface.transformation,
                    na.transformations.Cartesian3dRotationY(180 * u.deg),
                    na.transformations.Cartesian3dTranslation(
                        z=self.z_focus_spectrometer,
                    ),
                ]
            ),
        )

    @property
    def surface_grating(self) -> optika.surfaces.Surface:
        """
        The grating, tilted to graze the central ray and moved onto it.
        """
        position = self.position_grating
        return dataclasses.replace(
            self.grating.surface,
            transformation=na.transformations.TransformationList(
                [
                    na.transformations.Cartesian3dRotationY(
                        angle=self.angle_grating - 90 * u.deg,
                    ),
                    na.transformations.Cartesian3dTranslation(
                        x=position.x,
                        y=position.y,
                        z=position.z,
                    ),
                ]
            ),
        )

    @property
    def surface_sensor(self) -> optika.sensors.ImagingSensor:
        """
        The sensor, perpendicular to the central ray of the wavelength which
        lands on its center.
        """
        position = self.position_sensor
        return dataclasses.replace(
            self.sensor.surface,
            transformation=na.transformations.TransformationList(
                [
                    na.transformations.Cartesian3dRotationY(
                        angle=self.angle_diffracted(self.sensor.wavelength_center),
                    ),
                    na.transformations.Cartesian3dTranslation(
                        x=position.x,
                        y=position.y,
                        z=position.z,
                    ),
                ]
            ),
        )

    @functools.cached_property
    def system(self) -> optika.systems.SequentialSystem:
        """
        Convert this model into an instance of
        :class:`optika.systems.SequentialSystem`.

        The paraboloid of the telescope is the pupil stop and the slot is
        the field stop.

        This is a cached property that is only computed once.
        """
        slot = dataclasses.replace(
            self.slot.surface,
            transformation=na.transformations.Cartesian3dTranslation(
                z=self.z_slot,
            ),
        )
        surfaces = [
            *self.telescope_mirror.surfaces,
            slot,
            self.surface_spectrometer_entrance,
            self.surface_spectrometer_mirror_1,
            self.surface_spectrometer_mirror_2,
            self.surface_grating,
        ]
        return optika.systems.SequentialSystem(
            surfaces=surfaces,
            sensor=self.surface_sensor,
            grid_input=optika.vectors.ObjectVectorArray(
                wavelength=self.wavelength,
                field=self.field,
                pupil=self.pupil,
            ),
            kwargs_plot=self.kwargs_plot,
        )


@dataclasses.dataclass(eq=False, repr=False)
class Instrument(
    AbstractInstrument,
):
    """
    A model of the entire optical system.

    Sunlight is focused by a Wolter-I telescope onto a slot.
    Past the slot, a pair of identical grazing-incidence paraboloids relays
    the slot at unit magnification: the first collimates the diverging beam
    and the second focuses it.
    A planar varied-line-space grating in the converging beam then
    disperses the light onto a CCD sensor, which records an image of the
    slot in every spectral line.

    The optic axis is the :math:`z` axis, and the origin is the center of
    the intersection plane of the telescope.
    The grating folds the beam in the :math:`x`-:math:`z` plane,
    so the illuminated sector of the telescope is centered on the
    :math:`x` axis and the spectrum is dispersed along :math:`x`.
    """

    telescope_mirror: AbstractTelescopeMirror = dataclasses.MISSING
    """
    A model of the mirror of the Wolter-I telescope.
    """

    slot: AbstractSlot = dataclasses.MISSING
    """
    A model of the slot at the focus of the telescope.
    """

    spectrometer_mirror: AbstractSpectrometerMirror = dataclasses.MISSING
    """
    A model of the two spectrometer mirrors, which are identical.
    """

    grating: AbstractGrating = dataclasses.MISSING
    """
    A model of the diffraction grating.
    """

    sensor: Sensor = dataclasses.MISSING
    """
    A model of the CCD sensor.
    """

    distance_spectrometer: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The axial distance from the slot to the midpoint between the two
    spectrometer mirrors.
    This is also the axial distance from that midpoint to the focus of the
    second spectrometer mirror.
    """

    wavelength: None | u.Quantity | na.AbstractScalar = None
    """
    A default grid of wavelengths to trace through the system.
    """

    field: None | na.AbstractCartesian2dVectorArray = None
    """
    A default grid of field positions to trace through the system.
    """

    pupil: None | na.AbstractCartesian2dVectorArray = None
    """
    A default grid of pupil positions to trace through the system.
    """

    kwargs_plot: None | dict[str, Any] = None
    """
    Extra keyword arguments used to plot the optical system.
    """
