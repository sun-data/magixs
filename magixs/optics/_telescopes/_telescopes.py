import abc
import dataclasses
import astropy.units as u
import numpy as np
import named_arrays as na
import optika

__all__ = [
    "AbstractTelescope",
    "Telescope",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTelescope(
    optika.mixins.Printable,
):
    """
    An interface describing a Wolter Type-I telescope :cite:p:`Wolter1952`:
    a paraboloid followed by a confocal hyperboloid, both struck at grazing
    incidence.

    The telescope is described about its intersection plane,
    :math:`z = 0`, where the two mirrors meet, following
    :cite:t:`VanSpeybroeck1972`.
    The paraboloid lies upstream of this plane, the hyperboloid downstream,
    and the focus is at :math:`z = f`.
    """

    @property
    @abc.abstractmethod
    def focal_length(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the intersection plane to the focus.
        """

    @property
    @abc.abstractmethod
    def radius(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the mirrors at the intersection plane.
        """

    @property
    @abc.abstractmethod
    def length_primary(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial length of the paraboloid, which ends at the intersection
        plane.
        """

    @property
    @abc.abstractmethod
    def length_secondary(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial length of the hyperboloid, which starts at the
        intersection plane.
        """

    @property
    @abc.abstractmethod
    def material(self) -> None | optika.materials.AbstractMaterial:
        """
        The material of both mirrors.
        """

    @property
    def _material(self) -> optika.materials.AbstractMaterial:
        material = self.material
        if material is None:
            material = optika.materials.Mirror()
        return material

    @property
    def grazing_angle(self) -> u.Quantity | na.AbstractScalar:
        r"""
        The grazing angle :math:`\alpha` of a ray which meets the mirrors at
        the intersection plane.

        The ray enters parallel to the axis and each mirror turns it by
        :math:`2 \alpha`, so it reaches the focus at :math:`4 \alpha` to the
        axis and :math:`\tan 4 \alpha = r / f`.
        The paraboloid and the hyperboloid graze the ray at the same angle.
        """
        return np.arctan(self.radius / self.focal_length) / 4

    @property
    def _parameter_primary(self) -> u.Quantity | na.AbstractScalar:
        r"""
        The distance from the vertex of the paraboloid to its focus.

        The paraboloid is inclined to the axis by :math:`\alpha` at the
        intersection plane, which fixes this distance.
        """
        return self.radius * np.tan(self.grazing_angle) / 2

    @property
    def _z_vertex_primary(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial position of the vertex of the paraboloid.
        """
        return self.radius / (2 * np.tan(self.grazing_angle))

    @property
    def z_focus_primary(self) -> u.Quantity | na.AbstractScalar:
        r"""
        The axial position of the focus of the paraboloid, which is also the
        far focus of the hyperboloid.

        A ray reflected at the intersection plane travels at
        :math:`2 \alpha` to the axis toward this focus, so it sits at
        :math:`z = r / \tan 2 \alpha`.
        """
        return self._z_vertex_primary - self._parameter_primary

    @property
    def _center_secondary(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial position of the center of the hyperboloid, halfway between
        its foci.
        """
        return (self.z_focus_primary + self.focal_length) / 2

    @property
    def _semimajor_secondary(self) -> u.Quantity | na.AbstractScalar:
        """
        The semi-major axis of the hyperboloid.

        A hyperbola is the locus of points whose distances to its two foci
        differ by twice the semi-major axis, and the hyperboloid passes
        through the mirrors' radius at the intersection plane.
        """
        r = self.radius
        distance_far = np.sqrt(np.square(r) + np.square(self.z_focus_primary))
        distance_near = np.sqrt(np.square(r) + np.square(self.focal_length))
        return (distance_far - distance_near) / 2

    @property
    def _eccentricity_secondary(self) -> u.Quantity | na.AbstractScalar:
        """
        The eccentricity of the hyperboloid.
        """
        distance_focus = (self.z_focus_primary - self.focal_length) / 2
        return distance_focus / self._semimajor_secondary

    def _radius_primary(
        self,
        z: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the paraboloid at a given axial position.

        Parameters
        ----------
        z
            The axial position on the paraboloid.
        """
        z_vertex = self._z_vertex_primary
        return np.sqrt(4 * self._parameter_primary * (z_vertex - z))

    def _radius_secondary(
        self,
        z: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the hyperboloid at a given axial position.

        Parameters
        ----------
        z
            The axial position on the hyperboloid.
        """
        a = self._semimajor_secondary
        e = self._eccentricity_secondary
        b = a * np.sqrt(np.square(e) - 1)
        distance = (self._center_secondary - z) / a
        return b * np.sqrt(np.square(distance) - 1)

    @property
    def primary(self) -> optika.surfaces.Surface:
        """
        The paraboloid.

        It opens toward :math:`-z` so that rays travelling toward :math:`+z`
        reflect inward.
        Its aperture spans its whole length, from :math:`z = -L_p` out to the
        intersection plane, where it meets the hyperboloid at the radius of
        the telescope.
        """
        return optika.surfaces.Surface(
            name="paraboloid",
            sag=optika.sags.ParabolicSag(focal_length=-self._parameter_primary),
            material=self._material,
            aperture=optika.apertures.AnnularAperture(
                radius_inner=self.radius,
                radius_outer=self._radius_primary(-self.length_primary),
            ),
            transformation=na.transformations.Cartesian3dTranslation(
                z=self._z_vertex_primary,
            ),
        )

    @property
    def secondary(self) -> optika.surfaces.Surface:
        """
        The hyperboloid.

        It is confocal with the paraboloid: its far focus is the focus of the
        paraboloid and its near focus is the focus of the telescope, so a
        beam converging on the former is re-imaged to a point at the latter.
        A conic is fixed by its two foci and one point it passes through,
        which here is the radius of the telescope at the intersection plane.
        Its aperture spans its whole length, from the intersection plane to
        :math:`z = L_s`.
        """
        a = self._semimajor_secondary
        e = self._eccentricity_secondary
        return optika.surfaces.Surface(
            name="hyperboloid",
            sag=optika.sags.ConicSag(
                radius=-a * (np.square(e) - 1),
                conic=-np.square(e),
            ),
            material=self._material,
            aperture=optika.apertures.AnnularAperture(
                radius_inner=self._radius_secondary(self.length_secondary),
                radius_outer=self.radius,
            ),
            transformation=na.transformations.Cartesian3dTranslation(
                z=self._center_secondary - a,
            ),
        )

    @property
    def surfaces(self) -> list[optika.surfaces.Surface]:
        """
        The primary and the secondary, in the order light meets them.
        """
        return [self.primary, self.secondary]


@dataclasses.dataclass(eq=False, repr=False)
class Telescope(
    AbstractTelescope,
):
    """
    A Wolter Type-I telescope :cite:p:`Wolter1952`: a paraboloid followed by
    a confocal hyperboloid, both struck at grazing incidence.

    Soft X-rays reflect efficiently only at grazing incidence, so an X-ray
    telescope illuminates its mirrors far from the axis, on their steep,
    nearly cylindrical flanks.
    The two reflections satisfy the Abbe sine condition far better than one,
    which is what gives the Wolter-I its usable field of view.

    The telescope is fixed by its focal length and radius, both measured at
    the intersection plane where the two mirrors meet
    :cite:p:`VanSpeybroeck1972`, and by the axial length of each mirror.
    The grazing angle follows from the first two, and it is the same on both
    mirrors.
    The intersection plane is :math:`z = 0`, the paraboloid lies upstream of
    it, the hyperboloid downstream, and the focus is at :math:`z = f`.
    The entrance aperture is the annulus between the radius of the telescope
    and the radius of the paraboloid at its upstream end.

    Examples
    --------

    Build a telescope with a focal length of 3 m and draw its mirrors and the path of an annulus of
    rays in the :math:`z`-:math:`y` plane.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import numpy as np
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        import magixs

        telescope = magixs.optics.Telescope(
            focal_length=3 * u.m,
            radius=72 * u.mm,
            length_primary=200 * u.mm,
            length_secondary=200 * u.mm,
        )

        # A grazing mirror is a thin ring, so sample its entrance aperture
        # with a ring of rays halfway across the annulus.
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
            num_pixel=na.Cartesian2dVectorArray(2048, 2048),
            transformation=na.transformations.Cartesian3dTranslation(
                z=telescope.focal_length,
            ),
        )
        grid = optika.vectors.ObjectVectorArray(
            wavelength=15 * u.AA,
            field=na.Cartesian2dVectorArray(0, 0) * u.deg,
            pupil=pupil,
        )
        # The rays start from a plane upstream of the whole primary.
        sky = optika.surfaces.Surface(
            name="sky",
            transformation=na.transformations.Cartesian3dTranslation(z=-400 * u.mm),
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

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
            system.plot(ax=ax, components=("z", "y"), plot_rays=False, color="black")
            na.plt.plot(
                rays.outputs.position,
                ax=ax,
                axis=system.axis_surface,
                components=("z", "y"),
                color="tab:blue",
            )
            ax.set_title(f"grazing angle {telescope.grazing_angle.to(u.deg):.2f}")
    """

    focal_length: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the intersection plane to the focus.
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the mirrors at the intersection plane.
    """

    length_primary: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The axial length of the paraboloid, which ends at the intersection
    plane.
    """

    length_secondary: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The axial length of the hyperboloid, which starts at the intersection
    plane.
    """

    material: None | optika.materials.AbstractMaterial = None
    """
    The material of both mirrors.
    If :obj:`None`, the default, a perfect :class:`optika.materials.Mirror`.
    """
