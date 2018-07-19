import tensorflow as tf
import getopt
import sys
import os

from utils import get_train_batch, get_test_batch
import constants as c
from g_model import GeneratorModel
from d_model import DiscriminatorModel


class AVGRunner:
    def __init__(self, num_steps, model_load_path, num_test_rec):
        """
        Initializes the Adversarial Video Generation Runner.

        @param num_steps: The number of training steps to run.
        @param model_load_path: The path from which to load a previously-saved model.
                                Default = None.
        @param num_test_rec: The number of recursive generations to produce when testing. Recursive
                             generations use previous generations as input to predict further into
                             the future.
        """

        self.global_step = 0
        self.num_steps = num_steps
        self.num_test_rec = num_test_rec

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR, graph=self.sess.graph)

        if c.ADVERSARIAL:
            print 'Init discriminator...'
            self.d_model = DiscriminatorModel(self.sess,
                                              self.summary_writer,
                                              c.TRAIN_HEIGHT,
                                              c.TRAIN_WIDTH,
                                              c.SCALE_CONV_FMS_D,
                                              c.SCALE_KERNEL_SIZES_D,
                                              c.SCALE_FC_LAYER_SIZES_D)

        print 'Init generator...'
        self.g_model = GeneratorModel(self.sess,
                                      self.summary_writer,
                                      c.TRAIN_HEIGHT,
                                      c.TRAIN_WIDTH,
                                      c.FULL_HEIGHT,
                                      c.FULL_WIDTH,
                                      c.SCALE_FMS_G,
                                      c.SCALE_KERNEL_SIZES_G)

        print 'Init variables...'
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())

        # if load path specified, load a saved model
        if model_load_path is not None:
            self.saver.restore(self.sess, model_load_path)
            print 'Model restored from ' + model_load_path

    def train(self):
        """
        Runs a training loop on the model networks.
        """
        for i in xrange(self.num_steps):
            if c.ADVERSARIAL:
                # update discriminator
                batch = get_train_batch()
                print 'Training discriminator...'
                self.d_model.train_step(batch, self.g_model)

            # update generator
            batch = get_train_batch()
            print 'Training generator...'
            self.global_step = self.g_model.train_step(
                batch, discriminator=(self.d_model if c.ADVERSARIAL else None))

            # save the models
            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print '-' * 30
                print 'Saving models...'
                self.saver.save(self.sess,
                                c.MODEL_SAVE_DIR + 'model.ckpt',
                                global_step=self.global_step)
                print 'Saved models!'
                print '-' * 30

            # test generator model
            if self.global_step % c.TEST_FREQ == 0:
                self.test()

    def test(self):
        """
        Runs one test step on the generator network.
        """
        batch = get_test_batch(c.BATCH_SIZE, num_rec_out=self.num_test_rec)
        self.g_model.test_batch(
            batch, self.global_step, num_rec_out=self.num_test_rec)


def usage():
    print 'Options:'
    print '-l/--load_path=    <Relative/path/to/saved/model>'
    print '-t/--test_dir=     <Directory of test images>'
    print '-r/--recursions=   <# recursive predictions to make on test>'
    print '-a/--adversarial=  <{t/f}> (Whether to use adversarial training. Default=True)'
    print '-n/--name=         <Subdirectory of ../Data/Save/*/ in which to save output of this run>'
    print '-s/--steps=        <Number of training steps to run> (Default=1000001)'
    print '-O/--overwrite     (Overwrites all previous data for the model with this save name)'
    print '-T/--test_only     (Only runs a test step -- no training)'
    print '-H/--help          (Prints usage)'
    print '--stats_freq=      <How often to print loss/train error stats, in # steps>'
    print '--summary_freq=    <How often to save loss/error summaries, in # steps>'
    print '--img_save_freq=   <How often to save generated images, in # steps>'
    print '--test_freq=       <How often to test the model on test data, in # steps>'
    print '--model_save_freq= <How often to save the model, in # steps>'


def main():
    ##
    # Handle command line input.
    ##

    load_path = None
    test_only = False
    num_test_rec = 1  # number of recursive predictions to make on test
    num_steps = 1000001
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'l:t:r:a:n:s:OTH',
                                ['load_path=', 'test_dir=', 'recursions=', 'adversarial=', 'name=',
                                 'steps=', 'overwrite', 'test_only', 'help', 'stats_freq=',
                                 'summary_freq=', 'img_save_freq=', 'test_freq=',
                                 'model_save_freq='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-l', '--load_path'):
            load_path = arg
        if opt in ('-t', '--test_dir'):
            c.set_test_dir(arg)
        if opt in ('-r', '--recursions'):
            num_test_rec = int(arg)
        if opt in ('-a', '--adversarial'):
            c.ADVERSARIAL = (arg.lower() == 'true' or arg.lower() == 't')
        if opt in ('-n', '--name'):
            c.set_save_name(arg)
        if opt in ('-s', '--steps'):
            num_steps = int(arg)
        if opt in ('-O', '--overwrite'):
            c.clear_save_name()
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)
        if opt in ('-T', '--test_only'):
            test_only = True
        if opt == '--stats_freq':
            c.STATS_FREQ = int(arg)
        if opt == '--summary_freq':
            c.SUMMARY_FREQ = int(arg)
        if opt == '--img_save_freq':
            c.IMG_SAVE_FREQ = int(arg)
        if opt == '--test_freq':
            c.TEST_FREQ = int(arg)
        if opt == '--model_save_freq':
            c.MODEL_SAVE_FREQ = int(arg)

    # set test frame dimensions
    assert os.path.exists(c.TEST_DIR)
    c.FULL_HEIGHT, c.FULL_WIDTH = c.get_test_frame_dims()

    ##
    # Init and run the predictor
    ##

    runner = AVGRunner(num_steps, load_path, num_test_rec)
    if test_only:
        runner.test()
    else:
        runner.train()


if __name__ == '__main__':
    main()
