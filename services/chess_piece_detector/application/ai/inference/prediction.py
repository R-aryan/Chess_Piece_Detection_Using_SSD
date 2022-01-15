import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
from datetime import datetime
import pathlib
import glob
import matplotlib.pyplot as plt
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
from utils.utils import encodeImageIntoBase64

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

    def __load_image_into_numpy_array(self, path):
        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def __run_inference_for_single_image(self, model, image):
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        output_dict = model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        return self.__post_process(output_dict, image)

    def __post_process(self, output_dict, image):
        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > self.settings.detection_mask_threshold,
                                               tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def __visualize_inference(self, output_dict, image_np):
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        return self.__write_output_to_directory(image_np)

    def __write_output_to_directory(self, image):
        output_image_name = "output_image_" + str(datetime.now()).split(':')[-1] + ".jpg"
        output_filename = self.settings.OUTPUT_IMAGE_PATH + output_image_name
        cv2.imwrite(output_filename, image)
        open_coded_base64 = encodeImageIntoBase64(output_filename)

        return open_coded_base64

    def run(self, image_path):
        image_np = self.__load_image_into_numpy_array(image_path)
        output_dict = self.__run_inference_for_single_image(self.model, image_np)

        output_base64 = self.__visualize_inference(output_dict, image_np)

        result = {"image": output_base64.decode('utf-8')}
        return result
