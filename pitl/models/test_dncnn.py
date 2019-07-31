import os
scriptname = os.path.basename(__file__)[:-3]
import time

import numpy as np
from matplotlib import pyplot as plt
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from models.dncnn import DNCNN

image = camera().astype(np.float32)
image = rescale_intensity(image, in_range='image', out_range=(0, 1))

intensity = 5
np.random.seed(0)
noisy = np.random.poisson(image * intensity) / intensity
noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0).astype(np.float32)
noisy = np.expand_dims(np.expand_dims(noisy, axis=2), axis=0)

model = DNCNN(noisy.shape[1:], 10)
history = model.fit(noisy, EStop_patience=15, ReduceLR_patience=10)

hist = history.history
epoch = len(hist['loss'])
savepath = '/microDL/pitl/src/pitl/examples/output_data/{}_keru{}'.format(scriptname, 3)
if not os.path.exists(savepath):
    os.makedirs(savepath)

out = model.predict(noisy)
out = rescale_intensity(out.reshape(512, 512), in_range='image', out_range=(0, 1))
PSNR_out = psnr(image, out)
SSIM_out = ssim(image, out)
PSNR_nsy = psnr(image, noisy.reshape(512, 512))
SSIM_nsy = ssim(image, noisy.reshape(512, 512))

plt.figure()
plt.imshow(out, cmap='gray')
plt.title('Unet PSNR_out: {:.2f}, SSIM_out: {:.2f} \n PSNR_nsy: {:.2f}, SSIM_nsy: {:.2f}'.format(PSNR_out, SSIM_out, PSNR_nsy, SSIM_nsy))
plt.savefig(os.path.join(savepath, 'Poiss{}_ep{}.png'.format(intensity, epoch)), dpi=300)

plt.figure()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Original')
plt.subplot(222)
plt.imshow(noisy.reshape(512, 512), cmap='gray')
plt.axis('off')
plt.title('Noisy \nPSNR={:.2f}, SSMI={:.2f}'.format(PSNR_nsy, SSIM_nsy))
plt.subplot(224)
plt.imshow(out, cmap='gray')
plt.axis('off')
plt.title('Unet \nPSNR={:.2f}, SSMI={:.2f}'.format(PSNR_out, SSIM_out))
plt.subplots_adjust(left=0.11, right=0.9, top=0.91, bottom=0.02, hspace=0.25, wspace=0.2)
plt.savefig(os.path.join(savepath, 'Poiss{}_ep{}_4comp.png'.format(intensity, epoch)), dpi=300)
plt.show()