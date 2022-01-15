# Chess Piece Detection Using SSD

# Cards Detection Using FASTER RCNN

- End to end object detection project using SSD(Single Shot Detection).
- The training is done using [TFOD2.0(Tensorflow object detection) framework](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/).
- The application is served as an REST API using Flask.


# Steps to run the application 

### Setting up  virtual environment.

- What is [**Virtual Environment in python ?**](https://www.geeksforgeeks.org/python-virtual-environment/)
- [Create virtual environment in python](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/)
- [Create virtual environment Anaconda](https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/)
- create a virtual environment and install [requirements.txt](https://github.com/R-aryan/Chess_Piece_Detection_Using_SSD/blob/main/requirements.txt)

> pip install -r requirements.txt

- After setting up the virtual environment download the trained weights from [here](https://drive.google.com/drive/folders/1HndHDyQXGSR56E2OPujYaErWjBIs3jpZ?usp=sharing).
- After downloading the trained weights place it under the directory [**services/chess_piece_detector/application/ai/weights/**](https://github.com/R-aryan/Chess_Piece_Detection_Using_SSD/tree/main/services/chess_piece_detector/application/ai)
- After performing the above steps go to [services/chess_piece_detector/api](https://github.com/R-aryan/Chess_Piece_Detection_Using_SSD/tree/main/services/chess_piece_detector/api) and run **app.py**
> python app.py
- After running the **app.py** the web app can be accessed at **http://127.0.0.1:8000/** copy this url and paste it in your browser.
- The UI will look like the following.

![Sample UI](https://github.com/R-aryan/Chess_Piece_Detection_Using_SSD/blob/main/msc/sample-input.png)
  <br>
  <br>

- The picture can be uploaded using the **upload** button and after uploading the image click on **predict** to perform inference.
- Sample Input

![Sample Input](https://github.com/R-aryan/Chess_Piece_Detection_Using_SSD/blob/main/msc/smaple_input_1.png)
  <br>
  <br>

- Sample Output

![Sample output](https://github.com/R-aryan/Chess_Piece_Detection_Using_SSD/blob/main/msc/sample_output.png)
  <br>
  <br>

- The application logs can also be found [here](https://github.com/R-aryan/Chess_Piece_Detection_Using_SSD/tree/main/services/chess_piece_detector/logs).
 
