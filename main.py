from ultralytics import YOLO
import cv2
import time
import pyvirtualcam
import numpy as np


def add_background(bg_image_path, orig_image, masks):
    st = time.time()
    bg_image = cv2.imread(bg_image_path)
    _, width, height = masks.shape
    bg_image = cv2.resize(bg_image, (height, width))
    orig_image = cv2.resize(orig_image, (height, width))
    st = time.time()
    #print("Resize1", time.time()-st)
    st = time.time()
    res = masks[0]
    #print("Resize2", time.time()-st)

    st1 = time.time()
    condition = np.stack(
        (res,), axis=-1) > 0.5
    bg_image = np.where(condition, orig_image, bg_image)

    #print("Adding back", time.time()-st1)
    return bg_image


# Load a model
model = YOLO('yolov8n-seg.pt')
cap = cv2.VideoCapture(0)
video_frames = []
start_main = time.time()
size_image = cv2.imread("innopolis.jpeg")
i = 0
with pyvirtualcam.Camera(width=640, height=480, fps=20, fmt=pyvirtualcam.PixelFormat('24BG')) as cam:
    print(f'Using virtual camera: {cam.device}')

    while cap.isOpened():
        try:
            st = time.time()
            ret, frame = cap.read()
            if frame is None:
                break

            width, height, _ = frame.shape
            start = time.time()
            results = model(frame)
            #print("Inference:", time.time()-start)
            res_plotted = results[0].plot()
            #print("Resulted size", res_plotted.shape)
            #print("After masks:", time.time()-start)
            start = time.time()
            im = add_background("innopolis.jpeg", frame, results[0].masks.masks)
            #print("After adding background:", time.time()-start)
            print("Frame Rate", 1/(time.time()-st))
            #cv2.imshow("", im)
            #key = cv2.waitKey(1)
            #if key == ord('q'):
            #    break
            cam.send(im)
            #cam.sleep_until_next_frame()
            cv2.imwrite("resulted.jpeg", im)
        except:
            pass


print(time.time() - start_main)
