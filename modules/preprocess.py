import cv2 as cv
import os
import torch as torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from modules import utils
import argparse

class DashCamDataset(Dataset):
    
    def normalize(self, arr, scalerfile, return_scaler=True):
        scaler = StandardScaler()
        normed = scaler.fit_transform(arr.reshape(-1, 1))
        
        if scalerfile != None:
            # save scaler
            pickle.dump(scaler, open(scalerfile, 'wb'))
        
        if return_scaler:
            return normed, scaler
        else:
            return normed   

    def frames_to_numpy(self, path):
        data = {}
        print('Loading frames and labels...')
        for i, filename in tqdm(enumerate(os.listdir(path))):
            if filename.endswith(".jpg"):
                img = cv.imread(path+filename)
                data[i] = img
                continue
            else:
                continue
        return data

    def __init__(self, video_path, labels_path, transform=None, target_transform=None, scalerfile='data/scalers/speed_scaler.sav'):
        self.frame_labels = pd.read_csv(labels_path)
        self.frame_labels.columns = ['speed']
        self.transform = transform
        self.target_transform = target_transform
        
        if self.target_transform == 'norm':
            self.y, self.scaler = self.normalize(self.frame_labels.speed.values, scalerfile)
        elif self.target_transform == None:
            self.y = self.frame_labels.speed.values.reshape(-1, 1)
        
        if video_path.endswith('/') == False:
            print('Reading video and computing optical flow...')
            utils.video_to_optical_flow_frames(video_path, 'data/rgb/')
            self.x = self.frames_to_numpy(video_path)
        else:
            self.x = self.frames_to_numpy(video_path)

        if len(self.x) != len(self.y):
            print(f'X ({len(self.x)}) and y ({len(self.y)}) are misaligned')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)
        label = torch.Tensor(label)

        sample = {"image": image, "label": label}
        
        return sample

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', nargs="?", default=None, help='path to save optical flow frames')
    parser.add_argument('--show', nargs="?", default=False, help='show frames or not')
    parser.add_argument('--video_path', nargs="?", default='data/raw_video/train.mp4', help='path to video')

    args = parser.parse_args()
    
    utils.video_to_optical_flow_frames(args.video_path, args.show, args.target_path)
    
if __name__ == '__main__':
    main()

    
