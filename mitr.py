# mitr.py
"""
  weakly supervised learning of a
  linear boundary

"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

def mitr(data_gen, batch_size, input_dim, ngroups):

    input = tf.placeholder(tf.float32, [None, input_dim])
    # fraction of training sample containing positive instances
    pos_class_ratio = tf.placeholder(tf.float32, name='pos_class_ratio')
    # group indices present
    group_labels_input = tf.placeholder(tf.int32, [None])
    # sort inputs by group label index
    values, indices = tf.nn.top_k(group_labels_input, k=tf.shape(group_labels_input)[0], sorted=True)
    input = tf.gather(input, indices[::-1], axis=0)
    unique_group_labels_input, unique_group_enum = tf.unique(values[::-1])
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def lr(x):
        w = tf.Variable(tf.random_normal((2, 1)), dtype=tf.float32)
        b = tf.Variable(tf.random_normal((1,)), dtype=tf.float32)
        return tf.nn.sigmoid(tf.add(tf.matmul(x, w), b))

    def gpp(l, lh):
        l = tf.expand_dims(l, -1)
        lh = tf.expand_dims(lh, -1)
        l_sq = tf.reduce_sum(tf.square(l))
        lh_sq = tf.reduce_sum(tf.square(lh))
        l_lh_dp = tf.matmul(tf.transpose(l), lh)
        return tf.reduce_sum(l_sq - 2 * l_lh_dp + lh_sq)

    def pp(y):
        return pr_sq_diff(y)

    def pr_sq_diff(x):
        x_sq = tf.reduce_sum(tf.square(x), axis=-1)
        x_dp = tf.matmul(x, tf.transpose(x))
        return tf.abs(x_sq - 2 * x_dp + tf.transpose(x_sq))

    def rbf_kernal(x):
        gamma = 1.0
        return tf.exp(-gamma * pr_sq_diff(x))

    def aggpred(y):
        return tf.segment_mean(y[:, 0], unique_group_enum)

    N = tf.constant(batch_size, dtype=tf.float32)
    lambda_c = tf.constant(1.0, dtype=tf.float32)
    K = tf.constant(ngroups, dtype=tf.float32)

    ypred = lr(input)
    loss_1 = 1.0 / tf.square(N) * tf.reduce_sum(rbf_kernal(input) * pp(ypred))
    loss_2 = lambda_c / K * gpp(tf.cast(unique_group_labels_input, tf.float32), aggpred(ypred))
    loss = loss_1 + loss_2
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)
    writer = tf.summary.FileWriter("mitr_logs")
    saver = tf.train.Saver()
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss1", loss_1)
        tf.summary.scalar("loss2", loss_2)
        tf.summary.histogram("positive class ratio", pos_class_ratio)

        summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cpkt = tf.train.get_checkpoint_state(os.path.dirname("mitr" + "/checkpoint"))
        if cpkt and cpkt.model_checkpoint_path:
            saver.restore(sess, cpkt.model_checkpoint_path)
            print("Loaded checkpointed model:\n {} for {}.".format(cpkt.model_checkpoint_path,
                                                                   "mitr"))

        fig, axes = plt.subplots(nrows=4, ncols=4)
        fig.set_size_inches(11, 20)

        plt_ix = 0
        for X, group_labels, y, ratio in data_gen(batch_size):
            _, ls, ls1, ls2, step, summary, posrt = sess.run([optimizer, loss, loss_1, loss_2, global_step, summary_op, pos_class_ratio],
                                   feed_dict={input: X, group_labels_input: group_labels, pos_class_ratio: ratio})
            if step % 300 == 0:
                print("Loss: {} Loss1: {} Loss2: {}".format(ls, ls1, ls2))
                print("Fraction of sample data in positive class: {}".format(ratio))
                saver.save(sess, "mitr" + "/" + "mitr", global_step=step)
                writer.add_summary(summary, step)
                writer.flush()
                print("Saved to {}".format("mitr" + "/" + "mitr"))
                heatmap_pred = sess.run(ypred, feed_dict={input: heatmap_input(), group_labels_input: group_labels})
                heatmap(heatmap_pred, axes.flatten()[plt_ix])
                axes.flatten()[plt_ix].set_title("step: {}".format(step))
                plt_ix += 1
                if plt_ix > 15:
                    plt.savefig('heatmap.png', dpi=100)
                    print("Saved heatmap.png")
                    sys.exit(0)



def sample_decision_boundary_small_uniform(batch_size):
    """ Sample random small squares uniformly to learn
    linear decision surface in multi-instance learning. """
    def boundary_label(e):
        if e[1] + e[0] > 0:
            return 1
        return 0

    sq_len = 1.5

    while True:
        p = np.random.uniform(-5, 5, size=(2))
        x = np.random.uniform(0, sq_len, size=(100, 2)) + p
        y = np.apply_along_axis(boundary_label, 1, x).astype(np.float32)
        n_in_group = np.sum(y)
        ratio = n_in_group * 1.0 / batch_size
        y_group = np.zeros((batch_size,))
        y_group[:] = ratio
        yield x, y_group, y, ratio


def heatmap(heatmap_values, axes):
    """ heatmap_values is given in order rows then
        columns: 00, 01, 02,..etc
    """

    n = 100

    data = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            data[(i, j)] = heatmap_values[n * i + j]
    axes.pcolor(data, cmap=plt.cm.Blues)


def heatmap_input():
    l = 10.0
    n = 100
    column_labels = np.linspace(-l, l, n)
    row_labels = np.linspace(-l, l, n)
    x = np.zeros(shape=(n**2, 2))
    for i in range(n):
        for j in range(n):
            x[i * n + j] = [row_labels[i], column_labels[j]]
    return x

if __name__ == "__main__":
    mitr(sample_decision_boundary_small_uniform, 100, 2, 2)