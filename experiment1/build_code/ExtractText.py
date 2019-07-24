
# python build_code/ExtractText.py --data_dir data --result_dir results --config_file config.json

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import environ

import numpy as np
import PIL
from PIL import Image
import cv2
import json
import pytesseract

from handlers.preprocess import Preprocess
from handlers.tessaractImpl import TessaractImpl

FLAGS = None


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def set_config():
    # print(FLAGS)
    if (FLAGS.data_dir[0] == '$'):
      DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
      DATA_DIR = FLAGS.data_dir
    if (FLAGS.result_dir[0] == '$'):
      RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
      RESULT_DIR = FLAGS.result_dir

    with open(os.path.join(DATA_DIR, FLAGS.config_file), 'r') as f:
        RUN_CONFIG = json.load(f)

    # ensure_dir(RESULT_DIR)

    global CONFIG
    CONFIG = {
                "DATA_DIR": DATA_DIR,
                "RESULT_DIR": RESULT_DIR,
                "RUN_CONFIG": RUN_CONFIG
             }

def extractData():
    pp = Preprocess(CONFIG["RUN_CONFIG"])
    # config = pp.getConfig()

    imgPath = CONFIG["DATA_DIR"] + "/images/test/pan2.jpg"
    boxes = pp.crop_image_texts("PAN_FORMAT1", imgPath)

    if (FLAGS.framework == "tessaract"):
        tessaractiImpl = TessaractImpl(CONFIG["RUN_CONFIG"]["TESSARACT_CONFIG"])
        result = tessaractiImpl.extractData(boxes)

    print(result)


def main():
    set_config()
    extractData()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--config_file', type=str, default='config.json', help='Run Configuration file name')
  parser.add_argument('--framework', type=str, default='tessaract', help='OCR Framework to use')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start Processing.....")
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  main()
