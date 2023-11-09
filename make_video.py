import cv2
import os

import pathlib_variable_names as my_vars

##############
# Core from: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
##############

def make_video(image_folder, video_save_folder, video_name):
    video_name = video_name + ".avi"
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video_path = str(video_save_folder.joinpath(video_name))
    video = cv2.VideoWriter(video_path, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def make_state_to_dest():
    image_folder = my_vars.stateToDestP
    video_name = "State_Moving_To_Dest"
    video_save_folder = my_vars.vidsFolder
    make_video(image_folder, video_save_folder, video_name)


def make_state_to_state():
    
    image_folder = my_vars.orbsToStateP 
    image_folder = my_vars.orbsToStateF
    video_name = "State_1_Trans_to_State_2"
    video_save_folder = my_vars.vidsFolder
    make_video(image_folder, video_save_folder, video_name)

if __name__ == "__main__":
    #make_state_to_dest()
    make_state_to_state()
