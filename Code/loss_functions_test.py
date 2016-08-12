from loss_functions import *

sess = tf.Session()
BATCH_SIZE = 2
NUM_SCALES = 5
MAX_P      = 5
MAX_ALPHA  = 1


# noinspection PyClassHasNoInit
class TestBCELoss:
    def test_false_correct(self):
        targets = tf.constant(np.zeros([5, 1]))
        preds = 1e-7 * tf.constant(np.ones([5, 1]))
        res = sess.run(bce_loss(preds, targets))

        log_con = np.log10(1 - 1e-7)
        res_tru = -1 * np.sum(np.array([log_con] * 5))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_false_incorrect(self):
        targets = tf.constant(np.zeros([5, 1]))
        preds = tf.constant(np.ones([5, 1])) - 1e-7
        res = sess.run(bce_loss(preds, targets))

        log_con = np.log10(1e-7)
        res_tru = -1 * np.sum(np.array([log_con] * 5))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_false_half(self):
        targets = tf.constant(np.zeros([5, 1]))
        preds = 0.5 * tf.constant(np.ones([5, 1]))
        res = sess.run(bce_loss(preds, targets))

        log_con = np.log10(0.5)
        res_tru = -1 * np.sum(np.array([log_con] * 5))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_true_correct(self):
        targets = tf.constant(np.ones([5, 1]))
        preds = tf.constant(np.ones([5, 1])) - 1e-7
        res = sess.run(bce_loss(preds, targets))

        log = np.log10(1 - 1e-7)
        res_tru = -1 * np.sum(np.array([log] * 5))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_true_incorrect(self):
        targets = tf.constant(np.ones([5, 1]))
        preds = 1e-7 * tf.constant(np.ones([5, 1]))
        res = sess.run(bce_loss(preds, targets))

        log = np.log10(1e-7)
        res_tru = -1 * np.sum(np.array([log] * 5))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_true_half(self):
        targets = tf.constant(np.ones([5, 1]))
        preds = 0.5 * tf.constant(np.ones([5, 1]))
        res = sess.run(bce_loss(preds, targets))

        log = np.log10(0.5)
        res_tru = -1 * np.sum(np.array([log] * 5))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))


# noinspection PyClassHasNoInit
class TestLPLoss:
    def test_same_images(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_tru = 0
        for i in xrange(1, NUM_SCALES + 1):
            scale_preds.append(tf.constant(np.ones([BATCH_SIZE, 2**i, 2**i, 3])))
            scale_truths.append(tf.constant(np.ones([BATCH_SIZE, 2**i, 2**i, 3])))

        for p in xrange(1, MAX_P + 1):
            res = sess.run(lp_loss(scale_preds, scale_truths, p))
            assert res == res_tru, 'failed on p = %d' % p

    def test_opposite_images(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_tru = 0
        for i in xrange(1, NUM_SCALES + 1):
            scale_preds.append(tf.constant(np.zeros([BATCH_SIZE, 2**i, 2 ** i, 3])))
            scale_truths.append(tf.constant(np.ones([BATCH_SIZE, 2**i, 2 ** i, 3])))

            res_tru += BATCH_SIZE * 2**i * 2**i * 3

        for p in xrange(1, MAX_P + 1):
            res = sess.run(lp_loss(scale_preds, scale_truths, p))
            assert res == res_tru, 'failed on p = %d' % p

    def test_some_correct(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_tru = 0
        for i in xrange(1, NUM_SCALES + 1):
            # generate batch of 3-deep identity matrices
            preds = np.empty([BATCH_SIZE, 2**i, 2**i, 3])
            imat = np.identity(2**i)
            for elt in xrange(BATCH_SIZE):
                preds[elt] = np.dstack([imat, imat, imat])

            scale_preds.append(tf.constant(preds))
            scale_truths.append(tf.constant(np.zeros([BATCH_SIZE, 2**i, 2**i, 3])))

            res_tru += BATCH_SIZE * 2**i * 3

        for p in xrange(1, MAX_P + 1):
            res = sess.run(lp_loss(scale_preds, scale_truths, p))
            assert res == res_tru, 'failed on p = %d' % p

    def test_l_high(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_tru = 0
        for i in xrange(1, NUM_SCALES + 1):
            # opposite images
            preds = np.empty([BATCH_SIZE, 2**i, 2**i, 3])
            preds.fill(3)
            scale_preds.append(tf.constant(preds))
            scale_truths.append(tf.constant(np.zeros([BATCH_SIZE, 2**i, 2**i, 3])))

            res_tru += BATCH_SIZE * 2**i * 2**i * 3

        for p in xrange(1, MAX_P + 1):
            res = sess.run(lp_loss(scale_preds, scale_truths, p))
            assert res == res_tru * (3**p), 'failed on p = %d' % p


# noinspection PyClassHasNoInit
class TestGDLLoss:
    def test_same_uniform(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_tru = 0
        for i in xrange(1, NUM_SCALES + 1):
            scale_preds.append(tf.ones([BATCH_SIZE, 2 ** i, 2 ** i, 3]))
            scale_truths.append(tf.ones([BATCH_SIZE, 2 ** i, 2 ** i, 3]))

        for a in xrange(1, MAX_ALPHA + 1):
            res = sess.run(gdl_loss(scale_preds, scale_truths, a))
            assert res == res_tru, 'failed on alpha = %d' % a

    def test_same_nonuniform(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_tru = 0
        for i in xrange(1, NUM_SCALES + 1):
            # generate batch of 3-deep identity matrices
            arr = np.empty([BATCH_SIZE, 2 ** i, 2 ** i, 3])
            imat = np.identity(2 ** i)
            for elt in xrange(BATCH_SIZE):
                arr[elt] = np.dstack([imat, imat, imat])

            scale_preds.append(tf.constant(arr, dtype=tf.float32))
            scale_truths.append(tf.constant(arr, dtype=tf.float32))

        for a in xrange(1, MAX_ALPHA + 1):
            res = sess.run(gdl_loss(scale_preds, scale_truths, a))
            assert res == res_tru, 'failed on alpha = %d' % a

    # TODO: Not 0 loss as expected because the 1s array is padded by 0s, so there is some gradient.
    def test_diff_uniform(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_tru = 0
        for i in xrange(1, NUM_SCALES + 1):
            scale_preds.append(tf.zeros([BATCH_SIZE, 2 ** i, 2 ** i, 3]))
            scale_truths.append(tf.ones([BATCH_SIZE, 2 ** i, 2 ** i, 3]))

            # every diff should have an abs value of 1, so no need for alpha handling
            res_tru += BATCH_SIZE * 2 ** i * 2 * 3

        for a in xrange(1, MAX_ALPHA + 1):
            res = sess.run(gdl_loss(scale_preds, scale_truths, a))
            assert res == res_tru, 'failed on alpha = %d' % a

    def test_diff_one_uniform_one_not(self):
        # generate scales
        scale_preds = []
        scale_truths = []

        res_trus = np.zeros(MAX_ALPHA - 1)
        for i in xrange(1, NUM_SCALES + 1):
            # generate batch of 3-deep matrices with 3s on the diagonals
            preds = np.empty([BATCH_SIZE, 2 ** i, 2 ** i, 3])
            imat = np.identity(2 ** i) * 3
            for elt in xrange(BATCH_SIZE):
                preds[elt] = np.dstack([imat, imat, imat])

            scale_preds.append(tf.constant(preds, dtype=tf.float32))
            scale_truths.append(tf.zeros([BATCH_SIZE, 2 ** i, 2 ** i, 3]))

            # every diff has an abs value of 3, so we can multiply that, raised to alpha
            # for each alpha check, times the number of diffs in a batch:
            # BATCH_SIZE * (diffs to left + down) * (diffs from up and right) * (# 3s in height) *
            # (# channels)
            num_diffs = BATCH_SIZE * 2 * 2 * 2**i * 3

            for a in xrange(1, MAX_ALPHA):
                res_trus[a] += num_diffs * 3**a

        for a, res_tru in enumerate(res_trus):
            res = sess.run(gdl_loss(scale_preds, scale_truths, a + 1))
            assert res == res_tru, 'failed on alpha = %d' % (a + 1)


# noinspection PyClassHasNoInit
class TestAdvLoss:
    def test_false_correct(self):
        # generate scales
        scale_preds = []
        targets = tf.constant(np.zeros([5, 1]))

        res_tru = 0
        log_con = np.log10(1 - 1e-7)
        for i in xrange(NUM_SCALES):
            scale_preds.append(1e-7 * tf.constant(np.ones([5, 1])))
            res_tru += -1 * np.sum(np.array([log_con] * 5))

        res = sess.run(adv_loss(scale_preds, targets))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_false_incorrect(self):
        scale_preds = []
        targets = tf.constant(np.zeros([5, 1]))

        res_tru = 0
        log_con = np.log10(1e-7)
        for i in xrange(NUM_SCALES):
            scale_preds.append(tf.constant(np.ones([5, 1])) - 1e-7)
            res_tru += -1 * np.sum(np.array([log_con] * 5))

        res = sess.run(adv_loss(scale_preds, targets))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_false_half(self):
        scale_preds = []
        targets = tf.constant(np.zeros([5, 1]))

        res_tru = 0
        log_con = np.log10(0.5)
        for i in xrange(NUM_SCALES):
            scale_preds.append(0.5 * tf.constant(np.ones([5, 1])))
            res_tru += -1 * np.sum(np.array([log_con] * 5))

        res = sess.run(adv_loss(scale_preds, targets))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_true_correct(self):
        scale_preds = []
        targets = tf.constant(np.ones([5, 1]))

        res_tru = 0
        log = np.log10(1 - 1e-7)
        for i in xrange(NUM_SCALES):
            scale_preds.append(tf.constant(np.ones([5, 1])) - 1e-7)
            res_tru += -1 * np.sum(np.array([log] * 5))

        res = sess.run(adv_loss(scale_preds, targets))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_true_incorrect(self):
        scale_preds = []
        targets = tf.constant(np.ones([5, 1]))

        res_tru = 0
        log = np.log10(1e-7)
        for i in xrange(NUM_SCALES):
            scale_preds.append(1e-7 * tf.constant(np.ones([5, 1])))
            res_tru += -1 * np.sum(np.array([log] * 5))

        res = sess.run(adv_loss(scale_preds, targets))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))

    def test_true_half(self):
        scale_preds = []
        targets = tf.constant(np.ones([5, 1]))

        res_tru = 0
        log = np.log10(0.5)
        for i in xrange(NUM_SCALES):
            scale_preds.append(0.5 * tf.constant(np.ones([5, 1])))
            res_tru += -1 * np.sum(np.array([log] * 5))

        res = sess.run(adv_loss(scale_preds, targets))
        assert np.array_equal(np.around(res, 7), np.around(res_tru, 7))
