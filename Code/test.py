import numpy as np
import constants as c
from utils import normalize_frames, get_test_batch
from glob import glob
from scipy.misc import imread, imsave
import os

def save_batch(batch, num_rec_out):
    # TEST
    for clip_num, clip in enumerate(batch):
        for frame_num in xrange(c.HIST_LEN + num_rec_out):
            imsave(c.get_dir('TEST/' + str(clip_num) + '/') + str(frame_num) + '.png',
                   clip[:, :, frame_num * 3:(frame_num + 1) * 3])

def get_full_clips(data_dir, num_clips, num_rec_out=1):
    """
    Loads a batch of random clips from the unprocessed train or test data.

    @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
    @param num_clips: The number of clips to read.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape
             [num_clips, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    clips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (3 * (c.HIST_LEN + num_rec_out))])

    # get num_clips random episodes
    ep_dirs = np.random.choice(glob(data_dir + '*'), num_clips)
    print ep_dirs

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = glob(os.path.join(ep_dir, '*'))
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]
        print clip_num
        print clip_frame_paths

        # read in frames
        for frame_num, frame_path in enumerate(clip_frame_paths):
            frame = imread(frame_path, mode='RGB')
            norm_frame = normalize_frames(frame)

            clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = norm_frame

    # TEST
    save_batch(clips, num_rec_out)

    return clips

get_full_clips('../Data/Ms_Pacman/Test/', 1, num_rec_out=1)

# def test():
#     """
#     Runs one test step on the generator network.
#     """
#     batch = get_test_batch(c.BATCH_SIZE, num_rec_out=2)
#     save_batch(batch, 2)
#
#     # self.g_model.test_batch(
#     #     batch, self.global_step, num_rec_out=2)
#
# test()
