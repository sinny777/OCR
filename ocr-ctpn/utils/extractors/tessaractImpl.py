import cv2
import numpy as np
import pytesseract

class TessaractImpl(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG

    def extractData(self, boxes):
        result = {}
        for idx, box in enumerate(boxes):
            dataImg = cv2.cvtColor(box["boxImg"], cv2.COLOR_BGR2GRAY)
            # Apply dilation and erosion to remove some noise
            kernel = np.ones((1, 1), np.uint8)
            dataImg = cv2.dilate(dataImg, kernel, iterations=1)
            dataImg = cv2.erode(dataImg, kernel, iterations=1)
            if self.CONFIG["preprocess"] == "thresh":
        	    #  Apply threshold to get image with only black and white
        	    # dataImg = cv2.adaptiveThreshold(dataImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        	    dataImg = cv2.threshold(dataImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        	# make a check to see if median blurring should be done to remove
        	# noise
            elif self.CONFIG["preprocess"] == "blur":
                dataImg = cv2.medianBlur(dataImg, 3)

        	# in order to apply Tesseract v4 to OCR text we must supply
        	# (1) a language, (2) an OEM flag of 4, indicating that the we
        	# wish to use the LSTM neural net model for OCR, and finally
        	# (3) an OEM value, in this case, 7 which implies that we are
        	# treating the ROI as a single line of text
            tessaractConfig = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(dataImg, config=tessaractConfig)
            print(text)
            if text.strip() and len(text) > 5:
                result[idx] = text
                # cv2.imshow(result[idx], box["boxImg"])
                # cv2.waitKey(0)

        return result
