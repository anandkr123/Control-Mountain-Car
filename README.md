# Control the Mountain-Car-v0 Gym env using your webcam


## Below are the three gestures chosen for controlling the car.

- push left (class 0)

![image](https://user-images.githubusercontent.com/23450113/222438191-e409b709-ecb2-4691-8c1c-c66b2727f3c2.png)

- push right (class 2)

![image](https://user-images.githubusercontent.com/23450113/222438309-d5faf6b2-7754-4da8-9881-0a96c8e7f6ff.png)

- no action (class 1)

![image](https://user-images.githubusercontent.com/23450113/222438364-6c94f2d9-351b-4d29-a4f4-8278062e4f11.png)

*Specific gestures were chosen because all of them are very different from each other and it will help the neural network model to easily differentiate one class from another*

### Dataset properties 

- Prepared using OpenCV, captures frames from the camera
- 500 images of each class
- Images converted to binary with a threshold of 40.
- Images resized to 56 x 100 dimension to be fed into CNN model.

### Code flow and execution steps

- create_dataset.py --> script to create the dataset. It opens the camera, the user can place fingers or hands or the type of sample images user wants in their dataset. Run the script through the terminal and specify the **directory name** to save the captured images.

e.g. python create_dataset.py --folder_name <folder_name>

- resize_images.py --> for  pre-processing images

Run the script through the terminal and 
- specify original images folder,
- folder where to save the resized binary images and 
- the threshold value to be used.

e.g. original images path → Dataset/left/{left_1,left_2 . png}
     resized images path → resized/left/{left_1,left_2, . Png}

python  resize_images.py --images_original <original images folder> --images_resized <folder to save resize images> --threshold <threshold value>


### *run the model.py* script which reads the images from the resized folder and creates a Neural network model and starts training.

### *run prediction.py* 

- It will open a window with the front camera. bring the hand in the green box in the opened window.
- Do the gesture as listed in the first answer to control the car. 
  -Index finger to the left – left push to the car
  -Index finger up – right push to the car
  -Five fingers – no-push

- Precautions --> Make sure the fist or hand is not visible when giving actions for first two classes as it might miss-classify it as the third class (five fingers).
- The hand in the green box captures captures frame and calibrates the background until a given number of frames are captured.  
- The user can press **‘s’ to start the recording** and the camera will capture the gesture done in the green box to do a certain action based on the prediction.
- As soon as the user presses the ‘s’ button,  the hand region is segmented from the top right frame of the screen and the thresholded image is saved in the local disk.
- The thresholded image is again read from the local disk.
- The image is fed into neural network for predictions.
- Based on predictions, the corresponding action is done on MountainCar.
- Following, there is a pause of 1 second done by the script so the user can easily switch actions and the green box doesn’t misinterpret the prediction from the last action.
- For every action, the mountain car moves 5 times for that actions, so as to provide a better visualization to the user of the movement of the car.


### Realtime actions on the mountain car
[Task_1_MountainCar.webm](https://user-images.githubusercontent.com/23450113/222442477-21cec4ca-aeee-4551-b17b-6576778892bd.webm)

