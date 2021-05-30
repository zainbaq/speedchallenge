import cv2 as cv
import numpy as np
import os

def play_frames(path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            img = cv.imread(path+filename)
            cv.imshow('image', img)
            continue
        else:
            continue

def gray_to_dense_optical_flow(prev_gray, gray, mask):
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    return rgb, mask

def video_to_optical_flow_frames(source, show=False, write_path=None):
    cap = cv.VideoCapture(source)
    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)
    
    # Sets image saturation to maximum
    mask[..., 1] = 255
    frame_num = 0
    data = {}
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        
        # Converts each frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        rgb, mask = gray_to_dense_optical_flow(prev_gray, gray, mask)
        data[frame_num] = rgb

        # Opens a new window and displays the output frame
        if show:
            cv.imshow("frame", frame)
            cv.imshow("dense optical flow", rgb)
        
        if write_path != None:
            cv.imwrite(f"{write_path+frame_num}.jpg", rgb)

        # Updates previous frame
        prev_gray = gray
        frame_num = frame_num + 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # closes all windows
    cap.release()
    cv.destroyAllWindows()

    return data

