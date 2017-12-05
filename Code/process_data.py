import numpy as np
import getopt
import sys
from glob import glob
import os
from six.moves import xrange

import constants as c
from utils import process_clip
import queue
import threading

taskQueue = queue.Queue()


def worker(id):
    print('Worker', id, 'started.')
    while not taskQueue.empty():
        clip_num = taskQueue.get()
        if clip_num is None:
            break
        clip = process_clip()
        np.savez_compressed(c.TRAIN_DIR_CLIPS + str(clip_num), clip)

        if (clip_num + 1) % 100 == 0:
            print('Worker %d: Processed %d clips' % (id, clip_num + 1))
    print('Worker', id, 'finished.')



def process_training_data(num_clips):
    """
    Processes random training clips from the full training data. Saves to TRAIN_DIR_CLIPS by
    default.

    @param num_clips: The number of clips to process. Default = 5000000 (set in __main__).

    @warning: This can take a couple of hours to complete with large numbers of clips.
    """
    num_prev_clips = len(glob(c.TRAIN_DIR_CLIPS + '*'))

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(c.NUM_WORKERS)]

    for clip_num in xrange(num_prev_clips, num_clips + num_prev_clips):
        taskQueue.put(clip_num)
    for i in range(c.NUM_WORKERS):
        taskQueue.put(None)
    for i in range(c.NUM_WORKERS):
        workers[i].start()
    for i in range(c.NUM_WORKERS):
        workers[i].join()


def usage():
    print('Options:')
    print('-w/--num_workers= <# threads to process>')
    print('-n/--num_clips=   <# clips to process for training> (Default = 5000000)')
    print('-t/--train_dir=   <Directory of full training frames>')
    print('-c/--clips_dir=   <Save directory for processed clips>')
    print("                  (I suggest making this a hidden dir so the filesystem doesn't freeze")
    print("                   with so many files. DON'T `ls` THIS DIRECTORY!)")
    print('-o/--overwrite    (Overwrites the previous data in clips_dir)')
    print('-H/--help         (Prints usage)')


def main():
    ##
    # Handle command line input
    ##

    num_clips = 5000000

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'w:n:t:c:oH',
                                ['num_workers=', 'num_clips=', 'train_dir=', 'clips_dir=', 'overwrite', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-w', '--num_workers'):
            c.NUM_WORKERS = int(arg)
        if opt in ('-n', '--num_clips'):
            num_clips = int(arg)
        if opt in ('-t', '--train_dir'):
            c.TRAIN_DIR = c.get_dir(arg)
        if opt in ('-c', '--clips_dir'):
            c.TRAIN_DIR_CLIPS = c.get_dir(arg)
        if opt in ('-o', '--overwrite'):
            c.clear_dir(c.TRAIN_DIR_CLIPS)
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)

    # set train frame dimensions
    assert os.path.exists(c.TRAIN_DIR)
    c.FULL_HEIGHT, c.FULL_WIDTH = c.get_train_frame_dims()

    ##
    # Process data for training
    ##

    process_training_data(num_clips)


if __name__ == '__main__':
    main()
