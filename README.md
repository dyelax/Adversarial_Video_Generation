# Adversarial Video Generation
This project implements a generative adversarial network to predict future frames of video, as detailed in "Deep Multi-Scale Video Prediction Beyond Mean Square Error" by Mathieu, Couprie & LeCun.

Adversarial generation uses two networks – a generator and a discriminator – to improve the sharpness of generated images. Given the past four frames of video, the generator learns to generate accurate predictions for the next frame. Given either a generated or a real-world image, the discriminator learns to correctly classify between generated and real. The two networks "compete," with the generator attempting to fool the discriminator into classifying its output as real. This forces the generator to create frames that are very similar to what real frames in the domain might look like.

## Results and Comparison
I trained and tested my network on a dataset of frames sequences from Ms. PacMan. To compare adversarial training vs. non-adversarial, I trained an adversarial network for 500,000 steps on both the generator and discriminator and a non-adversarial network for 1,000,000 steps (as the non-adversarial network runs about twice as fast). Training took around 24 hours for each network, using a GTX 980TI GPU.

Using the error measurements outlined in the paper (Peak Signal to Noise Ratio and Sharp Difference) did not show significant difference between the adversarial and non-adversarial networks. I believe this is because sequential frames from the Ms. PacMan dataset have no motion in the majority of pixels. While I could not replicate the paper's results numerically, it is clear that adversarial training produces a qualitative improvement in the sharpness of the generated frames, especially over long time spans.

In the following examples, I ran the networks recursively for 64 frames. (i.e. The input to generate the first frame was [input1, input2, input3, input4], the input to generate the second frame was [input2, input3, input4, generated1], etc.). As the networks are not fed actions from the original game, the goal is not to line up perfectly with the ground truth images (as they couldn't possbily know what to do at intersections), but to maintain a crisp and likely representation of the world.

The following example exhibits how quickly the non-adversarial network becomes fuzzy and loses definition of the sprites. The adversarial network exhibits this behavior to an extent, but is much better at maintaining sharp representations of at least some sprites throughout the sequence.

![Comparison 1](https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/4_Comparison.gif)

This example shows how the adversarial network is able to keep a sharp representation of PacMan around multiple turns, while the non-adversarial network fails to do so.

![Comparison 2](https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/5_Comparison.gif)

While the adversarial network is clearly superior in terms of sharpness and consistency over time, the non-adversarial network does generate some fun/spectacular failures:

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
  


https://drive.google.com/open?id=0Byf787GZQ7KvR2JvMUNIZnFlbm8
