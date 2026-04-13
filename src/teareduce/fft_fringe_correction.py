#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""
fft_fringe_correction.py
========================
Second-order flatfield correction via FFT low-pass filtering.

CLI usage
---------
    python fft_fringe_correction.py input.fits output.fits \
        --freq-radius 25 --tukey-alpha 0.05 --corner-sharpness 4 --kmedian 0 --verbose --plots

Jupyter / script usage
----------------------
    from fft_fringe_correction import correct_fringe

    fringe_norm, data_corrected = correct_fringe(
        data,
        freq_radius      = 25,
        tukey_alpha      = 0.05,
        corner_sharpness = 4,
        kmedian          = 0,
        verbose          = True,
        plots            = True,
    )
"""

import argparse

from astropy.io import fits
from astropy.visualization import ZScaleInterval
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def tukey_radial_2d(ny, nx, alpha=0.30, p=6):
    """
    Build a 2D radial Tukey (tapered cosine) window using the Lp norm.

    Unlike the separable outer-product construction, this window attenuates
    the corners of the image more aggressively than the edge midpoints,
    matching the vignetting pattern of instruments such as CAFOS.

    The Log-Power distance from the centre for normalised coordinates (x,y) ∈ [-1,1]:

        r = ( |x|^p + |y|^p )^(1/p)

    With p > 2 the iso-distance contours are progressively more square-like.
    At a corner (|x|=1, |y|=1): r = 2^(1/p) > 1, so corners always lie in
    the taper or zero region regardless of alpha.

    The Tukey profile applied to r:
        r ≤ (1−α)      → weight = 1          (flat top)
        (1−α) < r ≤ 1  → cosine taper 1 → 0 (smooth suppression)
        r > 1           → weight = 0          (corners zeroed)

    Parameters
    ----------
    ny, nx : int
        Image dimensions in pixels.
    alpha : float
        Fraction of the normalised radius over which the cosine taper acts.
        0 = rectangular (no apodisation); 1 = full Hann-like cosine.
    p : float
        Log-Power norm order. p=2 gives circular iso-contours; higher p gives
        more square-like contours with faster corner attenuation.

    Returns
    -------
    window : ndarray, shape (ny, nx), values in [0, 1].
    """
    y_lin = np.linspace(-1.0, 1.0, ny)
    x_lin = np.linspace(-1.0, 1.0, nx)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    r = (np.abs(Xg) ** p + np.abs(Yg) ** p) ** (1.0 / p)
    return np.where(
        r <= (1.0 - alpha),
        1.0,
        np.where(
            r <= 1.0,
            0.5 * (1.0 + np.cos(np.pi * (r - (1.0 - alpha)) / alpha)),
            0.0,
        ),
    )


def correct_fringe(data, freq_radius=25, tukey_alpha=0.05, corner_sharpness=4, kmedian=0, verbose=False, plots=False):
    """
    Apply a second-order flatfield correction by isolating and removing the
    large-scale fringe / illumination pattern via FFT low-pass filtering.

    The fringe problem is multiplicative:

        I_obs = I_true × P

    where P is the spatially varying sensitivity residual left after the
    first-pass flat-field division.  This function reconstructs P by:

      1. Subtracting the median (removes the DC offset).
      2. Applying a 2D radial Tukey window (suppresses spectral leakage from
         edge discontinuities caused by vignetting).
      3. Computing the 2D FFT and retaining only the low-frequency content
         inside a circular mask of radius ``freq_radius`` (in frequency-space
         pixels), which captures the large-scale fringe pattern.
      4. Inverse-transforming to recover the fringe map in image space.
      5. Optionally replacing the borders of the fringe map with the median-filtered
         value to mitigate edge artefacts (since the Tukey window may not fully
         suppress them, especially if alpha is small).
      6. Normalising the fringe map to unit median so it acts as a pure
         multiplicative correction factor.
      7. Dividing the original image by the normalised fringe map.

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        Input 2D image array (flat-divided CCD frame).  Must be float-
        convertible; will be cast to float64 internally.
    freq_radius : int
        Radius of the circular low-pass mask in frequency-space pixels.
        Larger values capture coarser fringe patterns; smaller values are
        more conservative.  Rule of thumb: r ≈ N / λ where λ is the
        characteristic fringe scale in pixels and N = min(ny, nx).
    tukey_alpha : float
        Fraction of the normalised Log-Power radius over which the cosine taper of
        the Tukey window acts.  0 = rectangular; 1 = Hann-like.
    corner_sharpness : float
        Order p of the Log-Power norm used to build the 2D Tukey window.  Higher
        values make the iso-weight contours more square-like and attenuate
        the corners more aggressively.
    kmedian : int
        Size of the median filter kernel used to replace the borders of the
        resulting fringe map with the median-filtered value to mitigate edge
        artefacts. This is optional since the Tukey window should already suppress
        edge discontinuities, but it can be helpful in some cases. This number
        must be odd and positive; if zero or negative, no median filtering is applied.
    verbose : bool
        If True, print diagnostic information about the correction process.
    plots : bool
        If True, produce and display two diagnostic figures:
          • Figure 1 — Tukey window construction and effect on the image.
          • Figure 2 — FFT fringe correction overview (spectra, fringe map,
            corrected image, residual).
        If False, run silently without opening any windows.

    Returns
    -------
    fringe_pattern : ndarray, shape (ny, nx)
        Reconstructed fringe pattern (before normalisation).
        This is the large-scale pattern isolated by the FFT low-pass filtering,
        living on the same flux scale as the input image
        (i.e. with the median level restored).
    data_corrected : ndarray, shape (ny, nx)
        Corrected image: data / fringe_norm.
    """
    if kmedian < 0:
        raise ValueError("kmedian must be zero or a positive odd integer")
    if kmedian > 0 and kmedian % 2 == 0:
        raise ValueError("kmedian must be an odd integer to have a well-defined centre pixel")

    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape

    if kmedian > min(ny, nx):
        raise ValueError(
            f"kmedian must be smaller than the image dimensions (got kmedian={kmedian}, image size={data.shape})"
        )

    # Subtract median (remove DC offset)
    data_zeromean = data - np.median(data)

    # Build and apply the 2D radial Tukey window
    window_2d = tukey_radial_2d(ny, nx, alpha=tukey_alpha, p=corner_sharpness)
    data_windowed = data_zeromean * window_2d

    # Compute 2D FFT and power spectrum
    # Note: the FFT of a real-valued image is symmetric, so we only need
    # to look at the shifted version to apply the circular mask correctly.
    F_shifted = np.fft.fftshift(np.fft.fft2(data_windowed))
    power_before = np.abs(F_shifted) ** 2

    # Build circular low-pass mask
    cy, cx = ny // 2, nx // 2
    Y_f, X_f = np.ogrid[:ny, :nx]
    dist_f = np.sqrt((X_f - cx) ** 2 + (Y_f - cy) ** 2)
    mask = (dist_f <= freq_radius).astype(np.float64)

    # Apply mask and inverse FFT to recover fringe pattern
    F_masked = F_shifted * mask
    power_after = np.abs(F_masked) ** 2
    fringe_raw = np.real(np.fft.ifft2(np.fft.ifftshift(F_masked)))

    # Restore the median level so the fringe map lives on the same flux scale
    fringe_fft = fringe_raw + np.median(data)

    # Optional: replace borders with median-filtered values to mitigate edge artefacts
    if kmedian > 0:
        if verbose:
            print("Computing median-filtered fringe map for border replacement")
            print("(this may take a moment)")
        time_ini = datetime.now()
        # The median filter will smooth the fringe map and provide a more stable
        # estimate of the large-scale pattern at the edges, which can be affected
        # by artefacts from the FFT masking. The Tukey window should already suppress
        # these artefacts, but this step can help ensure a cleaner correction,
        # especially if the original alpha is small.
        fringe_median = median_filter(data, size=kmedian)
        time_end = datetime.now()
        if verbose:
            print(f"Median filter completed in {(time_end - time_ini).total_seconds():.1f} seconds")
        # Use a slightly larger alpha for the mixing window to ensure
        # smooth blending of the median and FFT fringe maps, especially
        # if the original alpha is small and may not fully suppress edge artefacts.
        tukey_alpha_mix = tukey_alpha + 0.1
        tukey_alpha_mix = min(tukey_alpha_mix, 1.0)  # cap the alpha to avoid exceeding 1.0
        # Define mixing window that transitions from 1 at the centre to 0 at the edges,
        # with a slightly larger alpha to ensure smooth blending.
        window_2d_mix = tukey_radial_2d(ny, nx, alpha=tukey_alpha_mix, p=corner_sharpness)
        # Blend the FFT fringe map and the median-filtered fringe map using the mixing window.
        fringe_pattern = fringe_fft * window_2d_mix + fringe_median * (1 - window_2d_mix)
    else:
        fringe_pattern = fringe_fft

    # Normalise the fringe map to unit median so it acts as a pure
    # multiplicative correction factor (i.e. the median level of the
    # corrected image will match the original).
    fringe_norm = fringe_pattern / np.median(fringe_pattern)
    # Apply the multiplicative correction to the original data (not the zero-mean version)
    data_corrected = data / fringe_norm

    if verbose:
        print(f"Fringe map range : {fringe_norm.min():.4f} – {fringe_norm.max():.4f}")
        print(f"Peak deviation   : {np.max(np.abs(fringe_norm - 1)) * 100:.2f} %")
        print(f"Median corrected : {np.median(data_corrected):.4f}")

    if plots:
        _plot_window(data_zeromean, data_windowed, window_2d, tukey_alpha, corner_sharpness)
        _plot_correction(
            data,
            power_before,
            power_after,
            fringe_norm,
            data_corrected,
            cx,
            cy,
            freq_radius,
            tukey_alpha,
            corner_sharpness,
        )

    return fringe_pattern, data_corrected


# ─────────────────────────────────────────────────────────────────────────────
# Private plotting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _style_ax(ax, title):
    """Apply light-theme styling to a matplotlib axis."""
    ax.set_facecolor("white")
    ax.set_title(title, color="#1a1a2e", fontsize=10, pad=8)
    ax.tick_params(colors="#444444")
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb")
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")


def _add_colorbar(fig, ax, im, label=None):
    """Attach a styled colorbar to an axis."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label, color="#444444", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#444444", labelcolor="#444444")
    return cb


def _stretch(img, contrast=0.25):
    """Return image clipped to DS9-style zscale cuts."""
    interval = ZScaleInterval(contrast=contrast)
    vmin, vmax = interval.get_limits(img)
    return np.clip(img, vmin, vmax)


def _log_power(power, exclude_dc_radius=5):
    """
    Return log10 of the power spectrum with robust display limits.
    The DC peak is excluded from the percentile calculation so the colour
    scale reflects the actual frequency content rather than the DC spike.
    """
    ny, nx = power.shape
    cy2, cx2 = ny // 2, nx // 2
    lp = np.log10(np.clip(power, 1e-10, None))
    Y2, X2 = np.ogrid[:ny, :nx]
    no_dc = np.sqrt((X2 - cx2) ** 2 + (Y2 - cy2) ** 2) > exclude_dc_radius
    return lp, np.percentile(lp[no_dc], 10), np.percentile(lp[no_dc], 99.9)


def _plot_window(data_zeromean, data_windowed, window_2d, tukey_alpha, corner_sharpness):
    """Figure 1: Tukey window construction and effect on the image."""
    ny, nx = window_2d.shape

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    fig.patch.set_facecolor("white")

    titles = [
        f"2D radial Tukey window  (α={tukey_alpha}, p={corner_sharpness})",
        "Log-Power iso-contours for different p values",
        "Window profile  (corner vs. edge midpoint)",
        "Original image  (zero-mean)",
        "Windowed image  (zero-mean)",
        "Window profile  (diagonal cross-section)",
    ]
    for ax, title in zip(axes.flat, titles):
        _style_ax(ax, title)

    # ── (0,0): 2D window with iso-contour at w=0.5 ───────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(window_2d, cmap="viridis", origin="lower", aspect="equal", vmin=0, vmax=1)
    _add_colorbar(fig, ax, im, label="weight  [0 – 1]")
    ax.contour(window_2d, levels=[0.5], colors="#1565c0", linewidths=1.0, linestyles="--")

    # ── (0,1): Log-Power iso-contours for several p values ───────────────────
    ax = axes[0, 1]
    y_c = np.linspace(-1.4, 1.4, 400)
    x_c = np.linspace(-1.4, 1.4, 400)
    Xc, Yc = np.meshgrid(x_c, y_c)
    colors_p = ["#ff6666", "#ffaa44", "#88dd44", "#44bbff", "#cc88ff"]
    p_values = [2, 4, 6, 10, 20]
    for p_val, col in zip(p_values, colors_p):
        r_c = (np.abs(Xc) ** p_val + np.abs(Yc) ** p_val) ** (1.0 / p_val)
        ax.contour(x_c, y_c, r_c, levels=[1.0], colors=[col], linewidths=1.3)
        ax.plot([], [], color=col, linewidth=1.3, label=f"p = {p_val}")
    r_sel = (np.abs(Xc) ** corner_sharpness + np.abs(Yc) ** corner_sharpness) ** (1.0 / corner_sharpness)
    ax.contour(x_c, y_c, r_sel, levels=[1.0], colors=["#1a1a2e"], linewidths=2.0, linestyles="-")
    ax.plot([], [], color="#1a1a2e", linewidth=2.0, label=f"p = {corner_sharpness}  ← selected")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.axvline(0, color="#cccccc", linewidth=0.5)
    ax.set_xlabel("normalised x")
    ax.set_ylabel("normalised y")
    ax.legend(
        fontsize=8, framealpha=0.8, labelcolor="#1a1a2e", facecolor="white", edgecolor="#bbbbbb", loc="upper right"
    )

    # ── (0,2): 1D Tukey profile — edge midpoint vs corner diagonal ────────────
    ax = axes[0, 2]
    t = np.linspace(0, 1.5, 500)

    def _tp(r, alpha):
        return np.where(
            r <= (1.0 - alpha),
            1.0,
            np.where(r <= 1.0, 0.5 * (1.0 + np.cos(np.pi * (r - (1.0 - alpha)) / alpha)), 0.0),
        )

    r_edge = t
    r_corner = t * 2.0 ** (1.0 / corner_sharpness)
    ax.plot(t, _tp(r_edge, tukey_alpha), color="#1565c0", lw=1.3, label="edge midpoint")
    ax.plot(t, _tp(r_corner, tukey_alpha), color="#c04d00", lw=1.3, label="corner diagonal")
    ax.axvline(1.0 - tukey_alpha, color="#888888", lw=0.8, ls=":", label="taper onset (edge)")
    ax.axvline(
        (1.0 - tukey_alpha) / 2.0 ** (1.0 / corner_sharpness),
        color="#c04d00",
        lw=0.8,
        ls=":",
        alpha=0.5,
        label="taper onset (corner)",
    )
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("distance from centre (normalised)")
    ax.set_ylabel("window weight")
    ax.legend(fontsize=8, framealpha=0.8, labelcolor="#1a1a2e", facecolor="white", edgecolor="#bbbbbb")
    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.axhline(1, color="#cccccc", linewidth=0.5, linestyle=":")

    # ── (1,0): original zero-mean image ──────────────────────────────────────
    ax = axes[1, 0]
    im = ax.imshow(_stretch(data_zeromean), cmap="gray", origin="lower", aspect="equal")
    _add_colorbar(fig, ax, im)

    # ── (1,1): windowed image ─────────────────────────────────────────────────
    ax = axes[1, 1]
    im = ax.imshow(_stretch(data_windowed), cmap="gray", origin="lower", aspect="equal")
    _add_colorbar(fig, ax, im)

    # ── (1,2): diagonal cross-section ─────────────────────────────────────────
    ax = axes[1, 2]
    dlen = min(ny, nx)
    idx = np.arange(dlen)
    raw = data_zeromean[idx, idx]
    win = data_windowed[idx, idx]
    wd = window_2d[idx, idx]
    ax.plot(idx, win, color="#ffffff", lw=0.8)
    ymin, ymax = ax.get_ylim()
    ax.plot(idx, raw, color="#888888", lw=0.8, label="zero-mean")
    ax.plot(idx, win, color="#1565c0", lw=0.8, label="windowed")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, dlen - 1)
    ax.set_xlabel("diagonal pixel index")
    ax.set_ylabel("image value")
    # add a secondary y-axis for the window weight and plot it on top
    ax2 = ax.twinx()
    ax2.plot(idx, wd, color="#c04d00", lw=1.0, ls="-", label="window weight")
    ax2.axhline(0.0, color="#c04d00", linewidth=0.5, linestyle=":")
    ax2.axhline(1.0, color="#c04d00", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("window weight", color="#c04d00")
    ax2.tick_params(colors="#c04d00")
    # combine legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=8,
        framealpha=0.8,
        labelcolor="#1a1a2e",
        facecolor="white",
        edgecolor="#bbbbbb",
        loc="lower center",
    )
    fig.suptitle(
        f"Radial Tukey window (α = {tukey_alpha}, p = {corner_sharpness})" " — construction and effect on the image",
        color="#1a1a2e",
        fontsize=13,
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig("tukey_window.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("Figure 1 saved → tukey_window.png")


def _plot_correction(
    data, power_before, power_after, fringe_norm, data_corrected, cx, cy, freq_radius, tukey_alpha, corner_sharpness
):
    """Figure 2: FFT fringe correction overview."""
    ny, nx = data.shape

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    fig.patch.set_facecolor("white")

    titles = [
        "Original  (flat-divided)",
        "Power spectrum  [before mask]",
        "Power spectrum  [after mask]",
        "Fringe pattern  (normalised)",
        "Corrected image",
        "Residual  (original − corrected)",
    ]
    for ax, title in zip(axes.flat, titles):
        _style_ax(ax, title)

    # Wavenumber axes for the power spectrum panels.
    # After fftshift the array runs from -N//2 to N//2-1 in each dimension.
    # Passing 'extent' to imshow replaces pixel indices with wavenumbers.
    # extent = [left, right, bottom, top] in data coordinates.
    spec_extent = [-nx // 2, nx // 2, -ny // 2, ny // 2]

    def _mask_circle(ax, r):
        # After fftshift the DC peak is at wavenumber (0, 0).
        circle = plt.Circle(
            (0, 0),
            r,
            edgecolor="#1565c0",
            facecolor="none",
            linewidth=1.5,
            linestyle=":",
            label=f"mask  r = {r} cycles",
        )
        ax.add_patch(circle)
        ax.legend(
            loc="upper right", fontsize=7, framealpha=0.8, labelcolor="#1a1a2e", facecolor="white", edgecolor="#bbbbbb"
        )

    def _spec_axis_labels(ax):
        ax.set_xlabel("$k_x$  (cycles / image width)", fontsize=8)
        ax.set_ylabel("$k_y$  (cycles / image height)", fontsize=8)

    # ── (0,0): original image ─────────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(_stretch(data), cmap="gray", origin="lower", aspect="equal")
    _add_colorbar(fig, ax, im)

    # ── (0,1): power spectrum before mask ─────────────────────────────────────
    ax = axes[0, 1]
    lp_b, vmin_p, vmax_p = _log_power(power_before)
    im = ax.imshow(lp_b, cmap="inferno", origin="lower", aspect="equal", vmin=vmin_p, vmax=vmax_p, extent=spec_extent)
    _mask_circle(ax, freq_radius)
    _spec_axis_labels(ax)
    _add_colorbar(fig, ax, im, label="log₁₀(power)")

    # ── (0,2): power spectrum after mask ──────────────────────────────────────
    ax = axes[0, 2]
    lp_a, _, _ = _log_power(power_after)
    im = ax.imshow(lp_a, cmap="inferno", origin="lower", aspect="equal", vmin=vmin_p, vmax=vmax_p, extent=spec_extent)
    _mask_circle(ax, freq_radius)
    _spec_axis_labels(ax)
    _add_colorbar(fig, ax, im, label="log₁₀(power)")

    # ── (1,0): normalised fringe map ──────────────────────────────────────────
    ax = axes[1, 0]
    dev = np.max(np.abs(fringe_norm - 1))
    im = ax.imshow(fringe_norm, cmap="RdBu_r", origin="lower", aspect="equal", vmin=1 - dev, vmax=1 + dev)
    _add_colorbar(fig, ax, im, label="multiplicative factor")

    # ── (1,1): corrected image ────────────────────────────────────────────────
    ax = axes[1, 1]
    im = ax.imshow(_stretch(data_corrected), cmap="gray", origin="lower", aspect="equal")
    _add_colorbar(fig, ax, im)

    # ── (1,2): residual map ───────────────────────────────────────────────────
    ax = axes[1, 2]
    im = ax.imshow(_stretch(data - data_corrected), cmap="coolwarm", origin="lower", aspect="equal")
    _add_colorbar(fig, ax, im, label="counts removed")

    fig.suptitle(
        f"FFT low-pass fringe correction  |  "
        f"radial Tukey α = {tukey_alpha},  p = {corner_sharpness}  |  "
        f"mask radius = {freq_radius} px",
        color="#1a1a2e",
        fontsize=13,
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig("fft_fringe_correction.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("Figure 2 saved → fft_fringe_correction.png")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="fft_fringe_correction",
        description=(
            "Second-order flatfield correction via FFT low-pass filtering.\n"
            "Isolates the large-scale fringe / illumination pattern, normalises\n"
            "it to unit median, and divides the input image by the result."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "  # Correct with default parameters and show diagnostic plots\n"
            "  python fft_fringe_correction.py flat_divided.fits corrected.fits --plots\n\n"
            "  # Override all parameters, run silently\n"
            "  python fft_fringe_correction.py flat_divided.fits corrected.fits \\\n"
            "      --freq-radius 20 --tukey-alpha 0.2 --corner-sharpness 8\n"
        ),
    )
    parser.add_argument("input", help="Input FITS file (flat-divided image).")
    parser.add_argument("output", help="Output FITS file (corrected image).")
    parser.add_argument(
        "--freq-radius",
        type=int,
        default=25,
        metavar="R",
        help=(
            "Radius of the circular low-pass mask in frequency-space pixels. "
            "Larger values capture coarser fringe patterns. "
            "Rule of thumb: R ≈ N / λ where λ is the fringe scale in pixels "
            "and N = min(image height, image width).  Default: 25."
        ),
    )
    parser.add_argument(
        "--tukey-alpha",
        type=float,
        default=0.05,
        metavar="A",
        help=(
            "Fraction of the normalised log-Power radius over which the cosine taper "
            "of the Tukey window acts.  0 = rectangular (no apodisation); "
            "1 = full Hann-like cosine.  Default: 0.05."
        ),
    )
    parser.add_argument(
        "--corner-sharpness",
        type=float,
        default=4,
        metavar="P",
        help=(
            "Order p of the log-Power norm used to build the 2D Tukey window.  "
            "Higher values make iso-weight contours more square-like and "
            "attenuate corners more aggressively.  Default: 4."
        ),
    )
    parser.add_argument(
        "--kmedian",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Size of the median filter kernel used to replace the borders of the "
            "resulting fringe map with the median-filtered value to mitigate edge "
            "artefacts.  This is optional since the Tukey window should already "
            "suppress edge discontinuities, but it can be helpful in some cases.  "
            "Default: 0 (no median filtering)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic information about the correction process (default: silent).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Display and save diagnostic figures (default: silent).",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # ── Load input FITS ───────────────────────────────────────────────────────
    print(f"Reading  : {args.input}")
    hdu = fits.open(args.input)
    data = hdu[0].data.astype(np.float64)
    header = hdu[0].header
    ny, nx = data.shape
    if args.verbose:
        print(f"Image    : {ny} × {nx} px")
        print(
            f"Params   : freq_radius={args.freq_radius}, "
            f"tukey_alpha={args.tukey_alpha}, "
            f"corner_sharpness={args.corner_sharpness}, "
            f"kmedian={args.kmedian}"
        )

    # ── Run correction ────────────────────────────────────────────────────────
    fringe_pattern, data_corrected = correct_fringe(
        data,
        freq_radius=args.freq_radius,
        tukey_alpha=args.tukey_alpha,
        corner_sharpness=args.corner_sharpness,
        kmedian=args.kmedian,
        verbose=args.verbose,
        plots=args.plots,
    )

    # ── Save output FITS ──────────────────────────────────────────────────────
    hdu_out = fits.PrimaryHDU(data_corrected.astype(np.float32), header=header)
    hdu_out.header["HISTORY"] = (
        f"FFT low-pass fringe correction: "
        f"radial Tukey alpha={args.tukey_alpha} p={args.corner_sharpness}, "
        f"mask radius={args.freq_radius} px, "
        f"kmedian={args.kmedian}"
    )
    hdu_out.writeto(args.output, overwrite=True)
    print(f"Saved    : {args.output}")


if __name__ == "__main__":
    main()
