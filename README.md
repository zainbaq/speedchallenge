# Dash Cam Speed Detector

Convolutional neural network trained on frames from dash cam .MP4 footage to regress driver speed.

Implemented dense optical flow to capture differences between frames. These mutated frames are then passed into the model.

The network consists of 4 convolutional layers:
- 3 x 6
- 6 x 12
- 12 x 24
- 24 x 36

These layers are then feed into a relu activated feed forward network for prediction.
