import PIL
from PIL import Image
from PIL import ImageEnhance
import cv2
import numpy as np
import math
import imutils

class ImageEnhancer:

    #resizes image and retains aspect ratio
    def resizeWithAspectRatio(self, img, width = None, height = None, inter = cv2.INTER_AREA):
        image = cv2.imread(img , cv2.IMREAD_COLOR)
        resized = imutils.resize(image, width=300)
        return resized

    # resize the image
    def resize(self, image, basewidth):
        wpercent = (basewidth/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        newImg = image.resize((basewidth,hsize), Image.ANTIALIAS)
        return newImg

    #increase sharpness and convert to monochrome
    def enhanceAndMonoChrome(self, image, factor):
        img = ImageEnhance.Sharpness(image)
        newImg = img.enhance(factor)
        return newImg.convert('L')

    #denoise
    def denoise(self, imageName):
        img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        denImg = cv2.fastNlMeansDenoising(img, None, 10, 7, 15)
        return denImg

    #assuming the image is in monochrome
    def backgroundWhite(self, image):
        black = (0,0,0)
        white = (255,255,255)
        threshold = (100,100,100)
        pixels = img.getdata()
        newPixels = []
        for pixel in pixels:
            if pixel < threshold:
                newPixels.append(black)
            else:
                newPixels.append(white)
        newImg = Image.new("RGB",img.size)
        newImg.putdata(newPixels)
        return newImg



    #binarize use with care
    def binarize(self, imageName):
        cim_num = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        ret, cim_num_t = cv2.threshold(cim_num, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cim_num_t

    def getOrientation(self, im):
        orientation = 0
        try:
            exif = im._getexif()
            print(exif)
        except Exception:
            exif = None
        if exif:
            orientation = exif.get(0X0112)
        return orientation

    def exif_orientation(self, im, orientation):
        if orientation == 2:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            im = im.rotate(180)
        elif orientation == 4:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            im = im.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            im = im.rotate(-90)
        elif orientation == 7:
            im = im.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            im = im.rotate(90)
        return im
