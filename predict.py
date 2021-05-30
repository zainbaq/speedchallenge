# %%
import cv2 as cv
import numpy as np
import argparse
from modules.preprocess import DashCamDataset
from modules.model import SpeedDetector
from modules.utils import gray_to_dense_optical_flow
import torch as torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
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
parser.add_argument('--live', nargs="?", default=False, help='live or computed predictions')

args = parser.parse_args()

LABELS = pd.read_csv(args.label_path, header=None).values.reshape(-1)
VIDEO_PATH = args.video_path
MODEL_PATH = args.model_path

# %%

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

def make_predictions_from_dataloader(model, dataloader):
    print('Predicting...')
    predictions = []
    for sample in tqdm(dataloader):
        p = model.predict(sample['image'])
        predictions.append(p)
    predictions = torch.cat(predictions)
    return predictions

def detect_via_dataset(model, transforms, unscale=True):
    dataset = DashCamDataset(
        'data/rgb/', 
        'data/labels/train.txt',
        transforms,
        'norm'
        )
    dataloader = DataLoader(dataset, batch_size=64)

    predictions = make_predictions_from_dataloader(model, dataloader)

    # scaler = pickle.load(open('data/scalers/scaler.sav', 'rb'))
    unscaled_predictions = list(dataset.scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).reshape(-1))

    labels = list(dataset.scaler.inverse_transform(
        np.array(dataset.y).reshape(-1, 1)
    ).reshape(-1))

    rmse = mean_squared_error(unscaled_predictions, labels) ** 0.5
    print(f'RMSE: {rmse}')
    return unscaled_predictions, labels, rmse

def live_predictions(model, labels, transforms, source, show=False):
    cap = cv.VideoCapture(source)
    scaler = pickle.load(open('data/scalers/scaler.sav', 'rb'))
    ret, first_frame = cap.read()

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rgb, mask = gray_to_dense_optical_flow(prev_gray, gray, mask)
        rgb_ = transforms(rgb)
        prediction = model(rgb_.unsqueeze(0)).detach().numpy()
        prediction_ = scaler.inverse_transform(prediction).item()
        frame_with_text = add_prediction_text(frame, prediction_, labels[frame_num])
        if show:
            cv.imshow('video', frame_with_text)
            cv.imshow('input', rgb)
        prev_gray = gray
        frame_num = frame_num + 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

def play_predictions(predictions, labels, source, show=False):

    cap = cv.VideoCapture(source)
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

        ret, frame = cap.read()
        # Convert frame to gray scale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Get optical flow between frames
        rgb, mask = gray_to_dense_optical_flow(prev_gray, gray, mask)
        
        # Add text to frame
        frame_with_text = add_prediction_text(frame, predictions[frame_num], labels[frame_num])
        
        if show:
            cv.imshow("input", frame_with_text)
            cv.imshow("dense optical flow", rgb)
            
        # Updates previous frame
        prev_gray = gray
        frame_num = frame_num + 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

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

    checkpoint = torch.load(MODEL_PATH)
    loaded_model = SpeedDetector(hidden_size=64)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    if args.live:
        live_predictions(
            loaded_model, 
            LABELS,
            image_transforms, 
            VIDEO_PATH, 
            show=True
            )
    else:
        predictions, labels, _ = detect_via_dataset(loaded_model, image_transforms)
        play_predictions(predictions, labels, VIDEO_PATH, show=True)

    # %%

if __name__ == '__main__':
    main()