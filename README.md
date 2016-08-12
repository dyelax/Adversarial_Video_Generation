# Adversarial Video Generation
This project implements a generative adversarial network to predict future frames of video, as detailed in "Deep Multi-Scale Video Prediction Beyond Mean Square Error" by Mathieu, Couprie & LeCun. 

## Results
I trained and tested my network on a dataset of frames from games of Ms. PacMan. 

While these error measures did not show significant difference between the adversarial and non-adversarial networks

The following example shows how quickly the non-adversarial network becomes fuzzy and loses definition of the sprites. The adversarial network exhibits this behavior to an extent, but is much better at maintaining crisp representations of at least some sprites throughout the frames.

![Comparison 1](https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/4_Comparison.gif)

The following example shows how the adversarial network is able to 

![Comparison 2](https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/5_Comparison.gif)

While the adversarial network is clearly better in terms of sharpness and consistency over time, the non-adversarial network does generate some spectacular failures:

![Rainbows!!!](https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/rainbow_NonAdv.gif)

## Usage

1. Clone or download this repository.
2. Prepare your data:
  - If you want to replicate my results, you can [downloaded the Ms. PacMan dataset here](https://drive.google.com/open?id=0Byf787GZQ7KvV25xMWpWbV9LdUU).
  - If you would like to train on your own videos, preprocess them so that they are directories of frame sequences as structured below. (Neither the names nor the image extensions matter, only the structure):
  ```
    - Test
      - Video 1
        - frame1.png
        - frame2.png
        - frame ...
        - frameN.png
      - Video ...
      - Video N
        - ...
    - Train
      - Video 1
        - frame ...
      - Video ...
      - Video N
        - frame ...
  ```
3. Process training data:
  - 
