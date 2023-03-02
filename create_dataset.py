import cv2
import argparse
import os

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def dataset_creation(dir):
    # Open the device at the ID 0
    # Use the camera ID based on
    cap = cv2.VideoCapture(0)
    # Check if camera was opened correctly
    if not (cap.isOpened()):
        print("Could not open video device")

    # Capture frame-by-frame
    count = 1
    while (True):
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        # Display the resulting frame
        clone = frame.copy()
        cv2.imshow("Video Feed", clone)
        cv2.imwrite(join_dir(cwd, dir, f'{dir}_{count}.png'), frame)
        count = count + 1
        # Waits for a user input to quit the application
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


desc = "Dataset creation"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-f', '--folder_name', type=str, default='images',
                    help='Specify directory name to save the data set images ')

args = parser.parse_args()

if args.folder_name:
    check_dir(args.folder_name)
    dataset_creation(args.folder_name)
else:
    exit()
