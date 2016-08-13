# Adversarial Video Generation
This project implements a generative adversarial network to predict future frames of video, as detailed in ["Deep Multi-Scale Video Prediction Beyond Mean Square Error"](https://arxiv.org/abs/1511.05440) by Mathieu, Couprie & LeCun.

Adversarial generation uses two networks – a generator and a discriminator – to improve the sharpness of generated images. Given the past four frames of video, the generator learns to generate accurate predictions for the next frame. Given either a generated or a real-world image, the discriminator learns to correctly classify between generated and real. The two networks "compete," with the generator attempting to fool the discriminator into classifying its output as real. This forces the generator to create frames that are very similar to what real frames in the domain might look like.

## Results and Comparison
I trained and tested my network on a dataset of frame sequences from Ms. Pac-Man. To compare adversarial 
training vs. non-adversarial, I trained an adversarial network for 500,000 steps on both the generator and 
discriminator, and I trained  a non-adversarial network for 1,000,000 steps (as the non-adversarial network 
runs about twice as fast). Training took around 24 hours for each network, using a GTX 980TI GPU.

In the following examples, I ran the networks recursively for 64 frames. (i.e. The input to generate the first frame was [input1, input2, input3, input4], the input to generate the second frame was [input2, input3, input4, generated1], etc.). As the networks are not fed actions from the original game, they cannot predict much of the true motion (such as in which direction Ms. Pac-Man will turn). Thus, the goal is not to line up perfectly with the ground truth images, but to maintain a crisp and likely representation of the world.

The following example exhibits how quickly the non-adversarial network becomes fuzzy and loses definition of the sprites. The adversarial network exhibits this behavior to an extent, but is much better at maintaining sharp representations of at least some sprites throughout the sequence:

<img src="https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/4_Comparison.gif" width="100%" />

This example shows how the adversarial network is able to keep a sharp representation of Ms. Pac-Man around multiple turns, while the non-adversarial network fails to do so:

<img src="https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/5_Comparison.gif" width="100%" />

While the adversarial network is clearly superior in terms of sharpness and consistency over time, the non-adversarial network does generate some fun/spectacular failures:

<img src="https://github.com/dyelax/Adversarial_Video_Generation/raw/master/Results/Gifs/rainbow_NonAdv.gif" width="50%" />

Using the error measurements outlined in the paper (Peak Signal to Noise Ratio and Sharp Difference) did not show significant difference between adversarial and non-adversarial training. I believe this is because sequential frames from the Ms. Pac-Man dataset have no motion in the majority of pixels. While I could not replicate the paper's results numerically, it is clear that adversarial training produces a qualitative improvement in the sharpness of the generated frames, especially over long time spans. You can view the loss and error statistics by running `tensorboard --logdir=./Results/Summaries/` from the root of this project.

## Usage

1. Clone or download this repository.
2. Prepare your data:
  - If you want to replicate my results, you can [download the Ms. Pac-Man dataset here](https://drive.google.com/open?id=0Byf787GZQ7KvV25xMWpWbV9LdUU). Put this in a directory named `Data/` in the root of this project for default behavior. Otherwise, you will need to specify your data location using the options outlined in parts 3 and 4.
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
  - The network trains on random 32x32 pixel crops of the input images, filtered to make sure that most clips have some movement in them. To process your input data into this form, run the script `python process_data` from the `Code/` directory with the following options:
  ```
  -n/--num_clips= <# clips to process for training> (Default = 5000000)
  -t/--train_dir= <Directory of full training frames>
  -c/--clips_dir= <Save directory for processed clips>
                  (I suggest making this a hidden dir so the filesystem doesn't freeze
                   with so many files. DON'T `ls` THIS DIRECTORY!)
  -o/--overwrite  (Overwrites the previous data in clips_dir)
  -H/--help       (prints usage)
  ```
  - This can take a few hours to complete, depending on the number of clips you want.
  
4. Train/Test:
  - If you want to plug-and-play with the Ms. Pac-Man dataset, you can [download my trained models here](https://drive.google.com/open?id=0Byf787GZQ7KvR2JvMUNIZnFlbm8). Load them using the `-l` option. (e.g. `python avg_runner.py -l ./Models/Adversarial/model.ckpt-500000`).
  - Train and test your network by running `python avg_runner.py` from the `Code/` directory with the following options:
  ```
  -l/--load_path=    <Relative/path/to/saved/model>
  -t/--test_dir=     <Directory of test images>
  -r--recursions=    <# recursive predictions to make on test>
  -a/--adversarial=  <{t/f}> (Whether to use adversarial training. Default=True)
  -n/--name=         <Subdirectory of ../Data/Save/*/ in which to save output of this run>
  -O/--overwrite     (Overwrites all previous data for the model with this save name)
  -T/--test_only     (Only runs a test step -- no training)
  -H/--help          (Prints usage)
  --stats_freq=      <How often to print loss/train error stats, in # steps>
  --summary_freq=    <How often to save loss/error summaries, in # steps>
  --img_save_freq=   <How often to save generated images, in # steps>
  --test_freq=       <How often to test the model on test data, in # steps>
  --model_save_freq= <How often to save the model, in # steps>
  ```
