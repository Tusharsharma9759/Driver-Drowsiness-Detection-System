# Driver_drowsiness_system_CNN
This is a system which can detect the drowsiness of the driver using CNN - Python, OpenCV

The aim of this is system to reduce the number of accidents on the road by detecting the drowsiness of the driver and warning them using an alarm. 

Here, we used Python, OpenCV, Keras(tensorflow) to build a system that can detect features from the face of the drivers and alert them if ever they fall asleep while while driving. The system dectects the eyes and prompts if it is closed or open. If the eyes are closed for 3 seconds it will play the alarm to get the driver's attention, to stop cause its drowsy.We have build a CNN network which is trained on a dataset which can detect closed and open eyes. Then OpenCV is used to get the live fed from the camera and run that frame through the CNN model to process it and classify wheather it opened or closed eyes.

## Setup
To set the model up:<br />
Pre-install all the required libraries <br />1) OpenCV<br />
                                       2) Keras<br />
                                       3) Numpy<br />
                                       4) Pandas<br />
                                       5) OS<br />
Download the Dataset from the link given below and edit the address in the notebook accordingly.<br />
Run the Jupyter Notebook and add the model name in detect_drowsiness.py file in line 20.<br />

## The Dataset
The dataset which was used is a subnet of a dataset from(https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)<br />
it has 4 folder which are <br />1) Closed_eyes - having 726 pictures<br />
                          2) Open_eyes - having 726 pictures<br />
                          3) Yawn - having 725 pictures<br />
                          4) no_yawn - having 723 pictures<br />

<img width="1440" alt="Screenshot 2023-07-08 at 11 50 03 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/4abd117a-eeb2-47b7-9b98-cd462ac44bb1">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 12 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/819639b1-c7f5-420e-9cf0-779a0e03d252">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 18 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/3ada4315-9ffb-4ea1-bbc7-34d95e92e773">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 22 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/7e4ded6b-a66a-42ff-aee6-b93cce4f00f5">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 28 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/87e78100-c77d-4008-998d-cc7a8a6a0e01">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 35 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/92303017-fd0c-4e58-9ad5-46935b848399">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 42 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/5dadc8f6-6710-4e21-b651-8d7084d1966f">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 49 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/2a8c0358-2fd2-4e1c-9570-01f00ed2c5c7">
<img width="1440" alt="Screenshot 2023-07-08 at 11 50 58 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/cfb1543e-1be1-4951-bd2a-45314196e8f8">
<img width="1440" alt="Screenshot 2023-07-08 at 11 51 14 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/0bad4987-1146-42d7-b9db-ea2f445666b5">
<img width="1440" alt="Screenshot 2023-07-08 at 11 51 21 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/44ca4ac9-39bc-4832-b972-56bddd84c7db">
<img width="1440" alt="Screenshot 2023-07-08 at 11 51 25 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/a6990012-f6cc-4f6e-9580-b0712721f83e">
<img width="1440" alt="Screenshot 2023-07-08 at 11 51 41 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/96044c62-7af4-446a-8e85-8028b4df6c76">
<img width="1440" alt="Screenshot 2023-07-08 at 11 51 45 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/7e0265b2-00d0-45a2-96e8-63adb0e4608b">
<img width="1440" alt="Screenshot 2023-07-08 at 11 51 55 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/f608ba81-811f-4259-91ce-2dff9080bbaa">
<img width="1440" alt="Screenshot 2023-07-08 at 11 51 59 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/6221aad7-fb3b-4b29-b739-a73ac67040f2">
<img width="1440" alt="Screenshot 2023-07-08 at 11 52 04 PM" src="https://github.com/Tusharsharma9759/Driver-Drowsiness-Detection-System/assets/114795523/c5a9323c-ab78-4fce-a22e-a76cb9bd14d6">










## The Convolution Neural Network
![CNN](https://user-images.githubusercontent.com/16632408/159187014-4bc4b70e-98d6-4313-873f-997ded2eff27.png)

## Accuracy 
We did 50 epochs, to get a good accuracy from the model i.e. 98% for training accuracy and 96% for validation accuracy.
![Graph](https://user-images.githubusercontent.com/16632408/159187004-92a72662-ddfe-471d-8bd6-65a3593a70a1.png)

## The Output 
1. Open Eyes<br />
![Open_eyes](https://user-images.githubusercontent.com/16632408/159187179-b557ab8e-fb8c-4408-850b-417893014f8c.png)
2. Close Eyes<br />
Here we detect wheater the eyes are closed and count the number of frames for which the eyes were closed (which is 10 frame) greater then that the Alarm will ring and the WARNING sign is displayed.
![Closed_eyes](https://user-images.githubusercontent.com/16632408/159187305-68cbdee3-8325-4216-85e3-7dbb66a429fb.png)


