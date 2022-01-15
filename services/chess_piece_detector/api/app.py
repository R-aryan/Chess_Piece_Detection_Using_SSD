from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from datetime import datetime

from services.chess_piece_detector.application.ai.inference.prediction import ChessDetector
from services.chess_piece_detector.application.ai.utils.utils import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        image_name = "input_image_" + str(datetime.now()).split(':')[-1] + ".jpg"
        chess_detector.settings.logger.info("Received Post Request for inference--!!")
        decodeImage(image, image_name, chess_detector.settings.INPUT_IMAGE_PATH)
        chess_detector.settings.logger.info(
            "Image stored in directory -- " + chess_detector.settings.INPUT_IMAGE_PATH + "--with image name--" + str(
                image_name))
        result = chess_detector.predict(chess_detector.settings.INPUT_IMAGE_PATH + image_name)
        return jsonify(result)
    except BaseException as ex:
        chess_detector.settings.logger.error("Following Error occurred while inference---!!", str(ex))
        return jsonify(str(ex))


if __name__ == "__main__":
    chess_detector = ChessDetector()
    port = 8000
    app.run(host='127.0.0.1', port=port)

