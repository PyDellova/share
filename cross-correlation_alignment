from skimage.registration import phase_cross_correlation
from astropy.wcs import FITSFixedWarning
from scipy.ndimage import fourier_shift
from reproject import reproject_interp
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from glob import glob
import numpy as np
import warnings
import cmocean
import os


def apply_shift_to_directory(directory, shift, output_suffix='_wcs_corrected', hdu_index=0):
    """Apply CRPIX shift to all FITS files in the directory."""
    fits_files = glob(os.path.join(directory, '*.fits'))

    print(f"\nApplying shift to {len(fits_files)} FITS files in: {directory}")
    for path in fits_files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FITSFixedWarning)
            with fits.open(path) as hdul:
                hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
                wcs = WCS(hdr)

                # Apply the shift
                wcs.wcs.crpix[0] -= shift[1]
                wcs.wcs.crpix[1] -= shift[0]

                # Update header
                hdr.update(wcs.to_header())

                # Save to new file
                base, ext = os.path.splitext(path)
                out_path = base + output_suffix + ext
                hdul.writeto(out_path, overwrite=True)

                print(f"Saved (updated wcs): {out_path}")


def visualize_contours(ref, target, aligned, levels=5):
    """Plot contours of target over reference, before and after alignment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Before Alignment", "After Alignment"]

    for ax, overlay, title in zip(axes, [target, aligned], titles):
        ax.imshow(ref, cmap=cmocean.cm.rain_r, origin='lower')
        ax.contour(overlay, levels=levels, colors='black', linewidths=1)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def load_fits_image(path, hdu_index=1):
    """Load 2D image data from a FITS file."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FITSFixedWarning)
        with fits.open(path) as hdul:
            data = hdul[hdu_index].data
        if data is None:
            raise ValueError(f"No data in HDU[{hdu_index}] of {path}")
    return data


def prepare_image(data, lower_percentile=1, upper_percentile=99):
    """Clip contrast and normalize image."""
    low = np.percentile(data, lower_percentile)
    high = np.percentile(data, upper_percentile)
    return np.clip(data, low, high)


def align_images_by_cross_correlation(ref_data, target_data, upsample=100):
    """Estimate and apply subpixel shift using phase cross-correlation."""
    shift, error, diffphase = phase_cross_correlation(ref_data, target_data, upsample_factor=upsample)
    shifted_fft = fourier_shift(np.fft.fftn(target_data), shift)
    aligned = np.fft.ifftn(shifted_fft).real
    return aligned, shift


def visualize_alignment(ref, target, aligned):
    """Show side-by-side plots of reference, target, and aligned images."""
    plt.figure(figsize=(12, 4))
    titles = ['Reference', 'Original Target', 'Aligned Target']
    images = [ref, target, aligned]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap=cmocean.cm.rain_r)
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()


def main():
    # Paths
    ref_path = '/home/pdellova/data/JWST/pattern-alignment/NGC_7023/Level3_CLEAR-F212N_i2d.fits'
    target_path = '/home/pdellova/data/JWST/pattern-alignment/NGC_7023/NIRSpec.fits'

    # Load data and WCS
    ref_data = load_fits_image(ref_path)
    target_data = load_fits_image(target_path, hdu_index=0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FITSFixedWarning)

        with fits.open(ref_path) as hdul:
            ref_header = hdul[1].header
            ref_wcs = WCS(ref_header)

        with fits.open(target_path) as hdul:
            target_header = hdul[1].header
            target_wcs = WCS(target_header)
            target_shape = hdul[1].data.shape

    # Reproject reference image to match target
    ref_data, _ = reproject_interp((ref_data, ref_wcs), target_wcs, shape_out=target_shape)

    # Clean NaNs
    ref_data[np.isnan(ref_data)] = np.nanmedian(ref_data)
    target_data[np.isnan(target_data)] = np.nanmedian(target_data)

    # Normalize both images
    ref_prepared = prepare_image(ref_data)
    target_prepared = prepare_image(target_data)

    # Align using phase cross-correlation
    aligned_data, shift = align_images_by_cross_correlation(ref_prepared, target_prepared)

    print(f"Estimated shift (y, x): {shift}")

    # Extract pixel scale (degrees/pixel)
    cdelt1 = abs(target_header.get('CDELT1', 1))  # in degrees/pixel
    cdelt2 = abs(target_header.get('CDELT2', 1))  # in degrees/pixel

    # Convert pixel shift to arcseconds
    shift_arcsec_x = shift[1] * cdelt1 * 3600  # shift[1] = x
    shift_arcsec_y = shift[0] * cdelt2 * 3600  # shift[0] = y

    print(f"Angular shift: ΔRA = {shift_arcsec_x:.3f}\"  ΔDec = {shift_arcsec_y:.3f}\"")

    # Adjust WCS reference pixel (CRPIX)
    wcs_aligned = target_wcs.deepcopy() 

    # Shift in pixels (note: CRPIX is 1-based in FITS, but this shift is relative)
    wcs_aligned.wcs.crpix[0] -= shift[1]  # X axis
    wcs_aligned.wcs.crpix[1] -= shift[0]  # Y axis

    # Write original (unaligned) data but with corrected WCS
    hdu_wcs_aligned = fits.PrimaryHDU(data=load_fits_image(target_path, hdu_index=0), header=wcs_aligned.to_header())
    hdu_wcs_aligned.writeto('/home/pdellova/data/JWST/pattern-alignment/NGC_7023/target_with_corrected_wcs.fits', overwrite=True)
    print("Saved: WCS-corrected target FITS (no resampling, no reprojection)")

    # Optionally apply this shift to all files in the same directory
    alignment_dir = "/home/pdellova/data/JWST/pattern-alignment/NGC_7023/to_align"
    apply_shift_to_directory(
        directory=alignment_dir,
        shift=shift,
        output_suffix='_wcs_corrected'
    )

    # Show images
    visualize_alignment(ref_prepared, target_prepared, aligned_data)

    # Overlay contour comparison
    visualize_contours(ref_prepared, target_prepared, aligned_data)


if __name__ == "__main__":
    main()
