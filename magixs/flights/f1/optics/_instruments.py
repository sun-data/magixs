import astropy.units as u
import numpy as np
import named_arrays as na
import magixs

__all__ = [
    "design",
]


def _coefficients_spacing() -> dict[int, u.Quantity]:
    """
    The polynomial which gives the ruling spacing along the grating.

    :cite:t:`Champey2022` tabulate the ruling period at four positions
    along the grating (their Table 4), so the cubic which passes through
    all four is used.
    """
    position = [-32, 0, 32, 41] * u.mm
    spacing = [541.6, 476.19, 415.01, 398.45] * u.nm
    degree = len(position) - 1
    coefficients = np.polynomial.polynomial.polyfit(
        x=position.value,
        y=spacing.value,
        deg=degree,
    )
    return {
        power: coefficients[power] * spacing.unit / position.unit**power
        for power in range(degree + 1)
    }


def design(
    num_wavelength: int = 5,
    num_field: int = 5,
    num_pupil: int = 21,
) -> magixs.optics.Instrument:
    """
    The optical design of the instrument for the first flight, with perfect
    mirrors and a perfect grating.

    The values are those published by :cite:t:`Champey2022`, with the
    following exceptions, which are not published and are inferred here.

    - The length of the paraboloid of the telescope is what remains of the
      total length of the instrument, 2513 mm, once the three spans in
      Figure 2 of :cite:t:`Champey2022` are subtracted from it.
    - The length of the hyperboloid of the telescope is taken to be the
      same as that of the paraboloid, which is longer than the beam needs.
    - The length of the spectrometer mirrors is the length of the segment
      over which :cite:t:`Champey2022` quote the figure of their mandrel.
    - The position of the grating is the one where the image of a point is
      the same distance from the grating along and across the dispersion.
      It puts the sensor 645 mm past the midpoint between the spectrometer
      mirrors, as in Figure 2 of :cite:t:`Champey2022`,
      and it puts the ends of the beam on the grating at the first and
      last positions of their Table 4.
    - The width of the grating is a guess which is wider than the beam.
    - The wavelength at the center of the sensor is chosen so that the
      sensor spans both the short end of the passband, 6 Å, and the longest
      wavelength :cite:t:`Savage2023` report, N VI at 29.5 Å.

    Parameters
    ----------
    num_wavelength
        The number of wavelengths in the default grid.
    num_field
        The number of field positions along each axis of the default grid.
    num_pupil
        The number of pupil positions along each axis of the default grid.

    Examples
    --------

    Plot the layout of the instrument in the plane of the dispersion.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import magixs

        instrument = magixs.flights.f1.optics.design()

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
            instrument.system.plot(
                ax=ax,
                components=("z", "x"),
                plot_rays=False,
                color="black",
            )
    """
    width_sector = 34 * u.deg

    return magixs.optics.Instrument(
        telescope_mirror=magixs.optics.TelescopeMirror(
            focal_length=1091 * u.mm,
            radius=73.06 * u.mm,
            length_primary=(2513 - 1091 - 652 - 645) * u.mm,
            length_secondary=125 * u.mm,
            angle_start=-width_sector / 2,
            angle_stop=width_sector / 2,
        ),
        slot=magixs.optics.Slot(
            half_width=na.Cartesian2dVectorArray(3.84, 10.6) * u.mm / 2,
        ),
        spectrometer_mirror=magixs.optics.SpectrometerMirror(
            radius_min=39.21 * u.mm,
            radius_max=41.91 * u.mm,
            length=80 * u.mm,
        ),
        grating=magixs.optics.Grating(
            coefficients_spacing=_coefficients_spacing(),
            diffraction_order=1,
            position_start=-32 * u.mm,
            position_stop=41 * u.mm,
            width=25 * u.mm,
            angle_grazing=2 * u.deg,
            distance=174 * u.mm,
        ),
        sensor=magixs.optics.Sensor(
            wavelength_center=17 * u.AA,
        ),
        distance_spectrometer=652 * u.mm,
        wavelength=na.linspace(6, 24, axis="wavelength", num=num_wavelength) * u.AA,
        field=na.Cartesian2dVectorArray(
            x=na.linspace(-1, 1, axis="field_x", num=num_field),
            y=na.linspace(-1, 1, axis="field_y", num=num_field),
        ),
        pupil=na.Cartesian2dVectorArray(
            x=na.linspace(-1, 1, axis="pupil_x", num=num_pupil),
            y=na.linspace(-1, 1, axis="pupil_y", num=num_pupil),
        ),
    )
