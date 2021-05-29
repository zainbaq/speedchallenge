import utils
import cv2 as cv
import numpy as np
import argparse
import os
import torch as torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import SpeedDetector
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
from torchvision import transforms
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', nargs="?", default='checkpoints/checkpoint.pt', help='path to checkpoint')
parser.add_argument('--scaler_path', nargs="?", default='data/scalers/scaler.sav', help='path to scaler.sav')
parser.add_argument('--label_path', nargs="?", default='data/labels/train.txt', help='path to labels.txt')
parser.add_argument('--video_path', nargs="?", default='data/raw_video/train.mp4', help='path to raw video')

args = parser.parse_args()

scaler = pickle.load(open(args.scaler_path, 'rb'))
labels = pd.read_csv(args.label_path, header=None)
VIDEO_PATH = args.video_path

def play_predictions(model, transform, source, show=False):

    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(source)
    # ret = a boolean return value from
    # getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    
    # Converts frame to grayscale because we only need the luminance channel for
    # detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    
    # Creates an image filled with zero intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(first_frame)
    
    # Sets image saturation to maximum
    mask[..., 1] = 255
    frame_num = 0
    while(cap.isOpened()):
        
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        ret, frame = cap.read()
        
        # Opens a new window and displays the input
        # frame
        
        # Converts each frame to grayscale - we previously 
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Calculates dense optical flow by Farneback method
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

        rgb_transformed = transform(rgb).unsqueeze(0)
        predicted_speed =  model.predict(rgb_transformed)

        print(scaler.inverse_transform(predicted_speed)[0][0], labels.values[frame_num])
        
        # Opens a new window and displays the output frame
        if show:
            cv.imshow("input", frame)
            cv.imshow("dense optical flow", rgb)
            
        # Updates previous frame
        prev_gray = gray
        frame_num = frame_num + 1
        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()

def main():
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((150, 480)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])  
    ])

    checkpoint = torch.load(args.model_path)
    loaded_model = SpeedDetector(hidden_size=64)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    play_predictions(loaded_model, image_transforms, VIDEO_PATH, show=True)



if __name__ == '__main__':
    main()