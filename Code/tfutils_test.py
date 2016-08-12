from tfutils import *

imgs = tf.constant(np.ones([2, 2, 2, 3]))
sess = tf.Session()


# noinspection PyClassHasNoInit,PyMethodMayBeStatic
class TestPad:
    def test_rb(self):
        res = sess.run(batch_pad_to_bounding_box(imgs, 0, 0, 4, 4))
        assert np.array_equal(res, np.array([[[[1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]
                                              ],
                                             [[[1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]
                                              ]], dtype=float))

    def test_center(self):
        res = sess.run(batch_pad_to_bounding_box(imgs, 1, 1, 4, 4))
        assert np.array_equal(res, np.array([[[[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]
                                              ],
                                             [[[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [1, 1, 1],
                                               [1, 1, 1],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]
                                              ]], dtype=float))


padded = batch_pad_to_bounding_box(imgs, 1, 1, 4, 4)


# noinspection PyClassHasNoInit
class TestCrop:
    def test_rb(self):
        res = sess.run(batch_crop_to_bounding_box(padded, 0, 0, 2, 2))
        assert np.array_equal(res, np.array([[[[0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [1, 1, 1]]],
                                             [[[0, 0, 0],
                                               [0, 0, 0]],
                                              [[0, 0, 0],
                                               [1, 1, 1]]]]))

    def test_center(self):
        res = sess.run(batch_crop_to_bounding_box(padded, 1, 1, 2, 2))
        assert np.array_equal(res, np.ones([2, 2, 2, 3]))
