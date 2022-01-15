import base64


def decodeImage(imgstring, fileName, image_path):
    try:
        imgdata = base64.b64decode(imgstring)
        with open(image_path + fileName, 'wb') as f:
            f.write(imgdata)
            f.close()
    except BaseException as ex:
        print(ex)


def encodeImageIntoBase64(croppedImagePath):
    try:
        with open(croppedImagePath, "rb") as f:
            return base64.b64encode(f.read())
    except BaseException as ex:
        print(ex)

