import os

from common.logging.console_logger import ConsoleLogger


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
