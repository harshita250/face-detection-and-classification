import glob
import sys
import os
import dlib
from PIL import Image
import cv2
import numpy as np


fps = 30
cropped_path = './data'
resized_path = './resized'
resolution = 1080
width = 216
height = 168

def face_detection():
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()

    for f in os.listdir(sys.argv[1]):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(sys.argv[1] + '/' + f)  # numpy
        image = Image.open(sys.argv[1] + '/' + f)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces. -- SIDE NOTE : DLIB DOESN'T WORK WELL IF THE IMAGES ARE SMALL, MTCNN WORKS BETTER
        dets = detector(img,
                        1)  # 0.8 detects most of the faces but also detects false positives. 0.8 is basically confidence score
        # detector.run(img,1,-1) -- multiple faces detection
        # print(dets)
        # print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            cropped = image.crop((d.left(), d.top(), d.right(), d.bottom()))
            name = f[:-4]
            print(name)
            cropped.save(r'cropped/' + name + '.jpg')
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        # dlib.hit_enter_to_continue()


def resize_imgs(path):
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
            im = Image.open(os.path.join(path, file))

            imResize = im.resize((width, height))
            imResize.save(r'resized/' + file, 'JPEG', quality=1080)
            #im.save(r'resized/' + file, 'JPEG')


def generate_video():
    image_folder = r'C:\Users\harsh\anaconda3\envs\face-recognition\code\overlay'  # make sure to use your folder
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width,_ = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (width, height))

    # Appending the images to the video one by one
    for image in images:
        count = 0
        while count <= 5:
            video.write(cv2.imread(os.path.join(image_folder, image)))
            #print(count)
            count = count + 1
        # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


if __name__ == "__main__":
    #face_detection()
    resize_imgs(cropped_path)
    #generate_video()


#create multiple video objects and try concatenating all the vid objects

