import torch
import torch.nn as nn

# Model Definition
class SpeedDetector(nn.Module):
        def __init__(self, hidden_size):
                super().__init__()

                self.hidden_size = hidden_size
                self.conv1 = nn.Conv2d(3, 6, 3)
                self.norm1 = nn.BatchNorm2d(6)
                
                self.conv2 = nn.Conv2d(6, 12, 8)
                self.norm2 = nn.BatchNorm2d(12)

                self.conv3 = nn.Conv2d(12, 24, 7)
                self.norm3 = nn.BatchNorm2d(24)

                self.conv4 = nn.Conv2d(24, 36, 4)
                self.norm4 = nn.BatchNorm2d(36)
                
                self.relu = nn.ReLU()
                self.pool = nn.AvgPool2d(2)
                self.dropout = nn.Dropout(0.5)

                self.fc1 = nn.Linear(4680, self.hidden_size)
                self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
                self.fc3 = nn.Linear(self.hidden_size, 1)

        def forward(self, x):
                
                # Convolution layers
                # print(x.shape)
                x = self.relu(self.pool(self.norm1(self.conv1(x))))
                # print(x.shape)

                x = self.relu(self.pool(self.norm2(self.conv2(x))))
                # print(x.shape)

                x = self.relu(self.pool(self.norm3(self.conv3(x))))
                # print(x.shape)

                x = self.relu(self.pool(self.norm4(self.conv4(x))))
                # print(x.shape)

                # fully connected layers
                x = x.view(x.size(0), -1)
                # print(x.shape)
                
                x = self.dropout(x)
                x = self.relu(self.fc1(x))
                # print(x.shape)

                x = self.dropout(x)
                x = self.fc2(x)
                # print(x.shape)

                x = self.dropout(x)
                x = self.fc3(x)
                # print(x.shape)
                return x

        def predict(self, image):
                self.eval()
                with torch.no_grad():
                        prediction = self.forward(image)
                return prediction
        
        def save_checkpoint(self, epoch, optimizer, loss, f_path):
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, f_path)
