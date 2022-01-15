import os

from common.logging.console_logger import ConsoleLogger
import tensorflow as tf


class Settings:
    sep = None
    if os.name == 'nt':
        sep = "\\"
    else:
        sep = "/"

    PROJECT_NAME = 'Chess_Piece_Detection_Using_SSD'

    root_path = os.getcwd().split(PROJECT_NAME)[0] + PROJECT_NAME + sep
    APPLICATION_PATH = root_path + "services" + sep + "chess_piece_detector" + sep + "application" + sep
    # print(APPLICATION_PATH)
    # setting up logs path
    LOGS_DIRECTORY = root_path + "services" + sep + "chess_piece_detector" + sep + "logs" + sep + "logs.txt"

    # input image path
    INPUT_IMAGE_PATH = "../images/input_images/"
    # output image path
    OUTPUT_IMAGE_PATH = "../images/output_images/"

    # setting up logger
    logger = ConsoleLogger(filename=LOGS_DIRECTORY)

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    MODEL_WEIGHTS_PATH = APPLICATION_PATH + "ai" + sep + "weights" + sep + "converted_trained_weights" + sep + "saved_model"

    # Path to label map file
    PATH_TO_LABELS = APPLICATION_PATH + "ai" + sep + "data" + sep + "labelmap.pbtxt"

    detection_mask_threshold = 0.5

    if len(tf.config.list_physical_devices('GPU')) > 0:
        DEVICE = 'GPU'
    else:
        DEVICE = 'CPU'


