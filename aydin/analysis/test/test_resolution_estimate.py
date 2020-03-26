from scipy.signal import convolve2d

from aydin.analysis.resolution_estimate import resolution_estimate_2d
from aydin.io.datasets import camera, add_noise
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def test_resolution_estimate():

    camera_image = camera()

    def noise(image):
        return add_noise(image, intensity=100, variance=0.01, sap=0.001)

    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    print(psf_xyz_array.shape)
    in_focus_psf_kernel = psf_xyz_array[0]
    defocused_psf_kernel = psf_xyz_array[7]

    in_focus_image = noise(convolve2d(camera_image, in_focus_psf_kernel, 'same'))
    out_of_focus_image = noise(convolve2d(camera_image, defocused_psf_kernel, 'same'))

    si_if, sip_if, cp_if, resolution_if, = resolution_estimate_2d(in_focus_image)
    si_of, sip_of, cp_of, resolution_of, = resolution_estimate_2d(out_of_focus_image)

    print(f"resolution_if={resolution_if}")
    print(f"resolution_of={resolution_of}")

    # TODO: this is not really robust, we need something better...
    # Idea: we could denoise and then estimate resolution....
