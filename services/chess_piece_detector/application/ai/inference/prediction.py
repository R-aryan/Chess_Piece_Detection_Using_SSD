import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from datetime import datetime
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

from utils.utils import encodeImageIntoBase64
from services.chess_piece_detector.settings import Settings

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


class ChessDetector:
    def __init__(self):
        self.settings = Settings
        # loading the model weights
        self.settings.logger.info("Loading Model Weights to--->" + str(self.settings.DEVICE))
        self.model = tf.saved_model.load(self.settings.MODEL_WEIGHTS_PATH)

        self.category_index = label_map_util.create_category_index_from_labelmap(self.settings.PATH_TO_LABELS,
                                                                                 use_display_name=True)
        self.settings.logger.info("Model Weights Loaded to--->" + str(self.settings.DEVICE) + "--Successfully--!!")

    def __load_image_into_numpy_array(self, path):
        self.settings.logger.info("Loading image--" + path + "--as a numpy array--!!")
        try:
            img_data = tf.io.gfile.GFile(path, 'rb').read()
            image = Image.open(BytesIO(img_data))
            (im_width, im_height) = image.size
            self.settings.logger.info("Image-->" + path + "--Loaded as numpy array successfully--!!")
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)
        except BaseException as ex:
            ex = str(ex)
            self.settings.logger.error("Error occurred while loading Image--> " + path + "--as numpy array--!!" + ex)
            return ex

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
        self.settings.logger.info("Writing Out Image to Directory -->"+self.settings.OUTPUT_IMAGE_PATH)
        try:
            output_image_name = "output_image_" + str(datetime.now()).split(':')[-1] + ".jpg"
            output_filename = self.settings.OUTPUT_IMAGE_PATH + output_image_name
            cv2.imwrite(output_filename, image)
            self.settings.logger.info(
                "Output Image stored in directory -- " + self.settings.OUTPUT_IMAGE_PATH + "----with image name--"
                + output_image_name + "--Successfully--!!")
            open_coded_base64 = encodeImageIntoBase64(output_filename)
            return open_coded_base64

        except BaseException as ex:
            ex = str(ex)
            self.settings.logger.error(
                "Following Error occurred while writing Output image to directory--> " + ex)
            return ex

    def predict(self, image_path):
        self.settings.logger.info("Performing Inference on Image-->" + str(image_path))
        try:
            image_np = self.__load_image_into_numpy_array(image_path)
            output_dict = self.__run_inference_for_single_image(self.model, image_np)

            output_base64 = self.__visualize_inference(output_dict, image_np)

            result = {"image": output_base64.decode('utf-8')}
            self.settings.logger.info("Inference on Image-->" + str(image_path) + "--performed successfully--!!")
            return result

        except BaseException as ex:
            ex = str(ex)
            self.settings.logger.error(
                "Following Error occurred while performing inference on  Image--> " + image_path + "---" + ex)
            return ex
