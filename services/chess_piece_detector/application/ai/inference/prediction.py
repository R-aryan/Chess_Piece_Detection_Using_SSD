import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import pathlib
import glob
import matplotlib.pyplot as plt
import cv2


from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

from services.chess_piece_detector.settings import Settings


class ChessDetector:
    def __init__(self):
        self.settings = Settings
        # loading the model weights
        self.model = tf.saved_model.load(self.settings.MODEL_WEIGHTS_PATH)

        self.category_index = label_map_util.create_category_index_from_labelmap(self.settings.PATH_TO_LABELS,
                                                                                 use_display_name=True)

    def load_image_into_numpy_array(self, path):
        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

