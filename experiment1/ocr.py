import PIL
from PIL import Image
import cv2
import numpy as np

from handlers.imageInhancer import ImageEnhancer

ie = ImageEnhancer()
im = Image.open('./data/images/adhaar1.png')
resized = ie.resize(im, 300)
enhanced = ie.enhanceAndMonoChrome(im, 3.5)
enhanced.save('./results/enhanced-adhaar1.jpeg', dpi=(300, 300))
denoised = ie.denoise('./results/enhanced-adhaar1.jpeg')
cv2.imwrite('./results/denoised-adhaar1.jpeg', denoised)
#deskewedImg = ie.getDeskewedImage('/Users/ravi/Documents/Learning/image-preprocessing/image-manipulation/enhanced-pancard-test-cw.jpeg')
#cv2.imwrite('/Users/ravi/Documents/Learning/image-preprocessing/image-manipulation/deskewed-pancard-test-cw2.jpeg', deskewedImg)

#deskewedImg = ie.deskew('/Users/ravi/Documents/Learning/image-preprocessing/image-manipulation/enhanced-pancard-test-cw.jpeg', angle)
#cv2.imwrite('/Users/ravi/Documents/Learning/image-preprocessing/image-manipulation/deskewed-pancard-test-cw.jpeg', deskewedImg)
#binarized = ie.binarize('./images/dob2.jpeg')
#cv2.imwrite('./images/binarized-dob2.jpeg', binarized)
