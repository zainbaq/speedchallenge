# %%
import utils
import cv2 as cv
import numpy as np
import argparse
from preprocess import DashCamDataset
import torch as torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from model import SpeedDetector
from torch.utils.data import random_split
from torchvision import transforms
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
# %%

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', nargs="?", default='checkpoints/checkpoint.pt', help='path to checkpoint')
parser.add_argument('--scaler_path', nargs="?", default='data/scalers/scaler.sav', help='path to scaler.sav')
parser.add_argument('--label_path', nargs="?", default='data/labels/train.txt', help='path to labels.txt')
parser.add_argument('--video_path', nargs="?", default='data/raw_video/train.mp4', help='path to raw video')

args = parser.parse_args()

scaler = pickle.load(open(args.scaler_path, 'rb'))
labels = pd.read_csv(args.label_path, header=None)
VIDEO_PATH = args.video_path

# %%
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

def add_prediction_text(image, prediction, label):
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale, thickness = 1, 1
    color = (0, 255, 255)
    frame_with_text = cv.putText(
        image, 
        f'Predicted: {prediction:.2f}', 
        (50, 50), font, fontScale, color, thickness, cv.LINE_AA
        )
    frame_with_text = cv.putText(
        image, 
        f'Actual: {label:.2f}', 
        (50, 100), font, fontScale, color, thickness, cv.LINE_AA
        )
    error = label - prediction
    error_c = (0, 255, 0) if error > 0 else (0, 0, 255)
    frame_with_text = cv.putText(
        image, 
        f'Error: {label-prediction:.2f}', 
        (50, 150), font, fontScale, error_c, thickness, cv.LINE_AA
        )
    return frame_with_text


def play_predictions(predictions, labels, source, show=False):

    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(source)
    # ret = a boolean return value from
    # getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    
    # Convert first frame to grayscale
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
        
        # Convert frame to gray scale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Get optical flow between frames
        rgb, mask = gray_to_dense_optical_flow(prev_gray, gray, mask)
        
        # Apply transform
        frame_with_text = add_prediction_text(frame, predictions[frame_num], labels[frame_num])
        
        if show:
            cv.imshow("input", frame_with_text)
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

# %%

def main():

    # %%
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((150, 480)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])  
    ])

    dataset = DashCamDataset(
        'data/rgb/', 
        'data/labels/train.txt',
        image_transforms,
        'norm'
        )
    
    # %%
    dataloader = DataLoader(dataset, batch_size=64)
    checkpoint = torch.load('checkpoints/checkpoint.pt') #args.model_path)
    loaded_model = SpeedDetector(hidden_size=64)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    print('Predicting...')
    predictions = []
    for sample in tqdm(dataloader):
        p = loaded_model.predict(sample['image'])
        predictions.append(p)
    predictions = torch.cat(predictions)
    print(predictions.shape)
    
    # %%
    unscaled_predictions = list(dataset.scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).reshape(-1))

    labels = list(dataset.scaler.inverse_transform(
        np.array(dataset.y).reshape(-1, 1)
    ).reshape(-1))

    rmse = mean_squared_error(unscaled_predictions, labels) ** 0.5
    print(f'RMSE: {rmse}')

    # %%
    fig, ax = plt.subplots()
    ax.plot(labels, label='t')
    ax.plot(unscaled_predictions, label='p', alpha=0.4)
    plt.legend()
    plt.show()
    # %%
    play_predictions(unscaled_predictions, labels, 'data/raw_video/train.mp4', show=True)
    # %%


if __name__ == '__main__':
    main()