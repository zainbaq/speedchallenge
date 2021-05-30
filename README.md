# Dash Cam Speed Detector

Convolutional neural network trained on frames from dash cam .MP4 footage to regress driver speed.

Implemented dense optical flow to capture differences between frames. These mutated frames are then passed into the model.

The network consists of 4 convolutional layers:
- 3 x 6
- 6 x 12
- 12 x 24
- 24 x 36

These layers are then feed into a relu activated feed forward network for prediction.

![Screenshot 2021-05-29 183248 - Copy](https://user-images.githubusercontent.com/46573513/120086408-9e704d80-c0ac-11eb-9efc-87ad4dfa2eed.png)

![Screenshot 2021-05-29 183248](https://user-images.githubusercontent.com/46573513/120086409-a03a1100-c0ac-11eb-9157-bff2220e0c69.png)
