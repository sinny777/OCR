
#!/usr/bin/env python

"""Preprocessing before OCR."""
import numpy as np
import random
import re
import json

import os.path
from os import path

# from imutils.object_detection import non_max_suppression
# import pytesseract
import argparse
import cv2

class Preprocess(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG

    def getBoxes(self, bbx_data):
        width = bbx_data['imageWidth']
        height = bbx_data['imageHeight']
        if len(bbx_data['points']) == 4:
            #Regular BBX has 4 points of the rectangle.
            xmin = width*min(bbx_data['points'][0][0], bbx_data['points'][1][0], bbx_data['points'][2][0], bbx_data['points'][3][0])
            ymin = height * min(bbx_data['points'][0][1], bbx_data['points'][1][1], bbx_data['points'][2][1],
                               bbx_data['points'][3][1])

            xmax = width * max(bbx_data['points'][0][0], bbx_data['points'][1][0], bbx_data['points'][2][0],
                               bbx_data['points'][3][0])
            ymax = height * max(bbx_data['points'][0][1], bbx_data['points'][1][1], bbx_data['points'][2][1],
                               bbx_data['points'][3][1])

        else:
            #OCR BBX format has 'x','y' in one point.
            # We store the left top and right bottom as point '0' and point '1'
            xmin = int(bbx_data['points'][0]['x']*width)
            ymin = int(bbx_data['points'][0]['y']*height)
            xmax = int(bbx_data['points'][1]['x']*width)
            ymax = int(bbx_data['points'][1]['y']*height)

        return (int(xmin), int(ymin), int(xmax), int(ymax))


    def crop_image_texts(self, type, imgPath):
        # print(self.load_dirty_json(self.configPath))
        image = cv2.imread(imgPath)
        boxes = []
        if self.CONFIG["annotations"]["type"] == "PAN_FORMAT1":
            for annotation in self.CONFIG["annotations"]["annotation"]:
                box = {"name": annotation["label"]}
                # box["points"] = self.getBoxes(annotation)
                startX, startY, endX, endY = self.getBoxes(annotation)
                orig = cv2.resize(image, (annotation['imageWidth'], annotation['imageHeight']))
                # cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # cv2.putText(output, box["name"], (startX, startY - 20),	cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                # cv2.imshow("Text Detection", output)
                boxImg = orig[startY:endY, startX:endX]
                box["img"] = boxImg
                boxes.append(box)
            return boxes
