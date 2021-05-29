# %%
from numpy.core.fromnumeric import shape
from torch._C import device
from preprocess import DashCamDataset
from model import SpeedDetector
import utils
import argparse
import torch as torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
# %%

# Arguments
parser = argparse.ArgumentParser()

# path to directory containing preprocessed images
parser.add_argument(
    '--frames_path', 
    nargs='?',
    default='data/rgb/', 
    help='path to flow frames'
    )
# path to dashcam video
parser.add_argument(
    '--video_path', 
    nargs='?',
    default='data/raw_video/train.mp4', 
    help='path to raw video to be preprocessed.'
    )
# path to labels.txt containing a float value for each frame in video
parser.add_argument(
    '--labels_path', 
    nargs='?',
    default='data/labels/train.txt', 
    help='path to frame wise labels'
    )
# batch size for training
parser.add_argument(
    '--batch_size',
    nargs="?",
    default=60,
    help='data loader batch size',
    type=int
)
# number of epochs to train for
parser.add_argument(
    '--n_epochs',
    nargs="?",
    default=5,
    help='number of epochs to train over',
    type=int
)
# path to checkpoint to load, None if retraining
parser.add_argument(
    '--checkpoint_path',
    nargs='?',
    default=None,
    help='path to checkpoint to be trained',
    type=str,
    required=False
)
# learning rate for training
parser.add_argument(
    '--lr',
    nargs="?",
    default=3e-4,
    type=int,
    required=False,
    help='learning rate for model training'
)
# train/validation split fraction
parser.add_argument(
    '--t_split',
    nargs="?",
    default=0.8,
    type=float,
    required=False,
    help='train/validation split fraction'
)
parser.add_argument(
    '--save_name',
    required=True,
    default='checkpoint.pt',
    help='desired name for checkpoint',
    type=str
)
args = parser.parse_args()

# device to train on
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
S_PATH = 'checkpoints/' + args.save_name

print(f'device: {DEVICE}')

def main():
    # %%
    # Image transforms
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((150, 480)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])

    dataset = DashCamDataset(
        args.frames_path,
        args.labels_path, 
        transform=image_transforms,
        target_transform='norm'
    )
    # %%
    # Train / Test split
    train_len = int(len(dataset)*args.t_split)
    lengths = [train_len, len(dataset)-train_len]

    trainset, validset = random_split(dataset, lengths)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)
    # %%
    # Model
    hidden_size = 64
    model = SpeedDetector(hidden_size=hidden_size).to(DEVICE)

    # Load checkpoints if passed in argparse
    if args.checkpoint_path != None:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Move tensors to training device
    criterion = nn.MSELoss().to(DEVICE)
    # %%

    # Training
    for epoch in range(args.n_epochs):

        # Train
        train_loss = 0.0
        model.train()
        for batch_idx, batch in tqdm(enumerate(trainloader)):
            image, label = batch['image'].to(DEVICE), batch['label'].to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)   
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(trainloader)

        # Validate
        valid_loss = 0.0
        model.eval()
        for batch_idx, batch in tqdm(enumerate(validloader)):

            with torch.no_grad():
                image, label = batch['image'].to(DEVICE), batch['label'].to(DEVICE)

                output = model(image)
                error = criterion(output, label)

                valid_loss += error.item()
        valid_loss = valid_loss / len(validloader)
        
        print(f'{epoch} - train: {train_loss}, valid: {valid_loss}')

        # Save checkpoint
        model.save_checkpoint(epoch, optimizer, train_loss, S_PATH)
            
if __name__ == '__main__':
    main()