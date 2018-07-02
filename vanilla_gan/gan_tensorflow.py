"""Generative Adversarial Nets."""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    """将权值初始化到一个较好的情况."""
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 784])

"""从大的网络架构层面进行创新(相比较与前面的CNN，RNN在网络内部连接层面进行创新)."""
# 初始化权值和偏置矩阵，D网络为一个三层网络
# 最终生成一个实数（是否限制在0-1之间？）
D_W1 = tf.Variable(xavier_init([784, 128]))
# 一维数据是如何运行的？shape = [128] 而非 shape = [128, 1] / [1, 128]
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
# 将参数构建成一个list（也可以将参数构成一个字典dict（））
theta_D = [D_W1, D_W2, D_b1, D_b2]

"""从100维噪声数据生成784维的手写数字数据."""
# 为网络提供随机噪声输入
Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    """生成服从均匀分布在[-1, 1]之间的随机矩阵."""
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    """使用三层网络对输入噪声进行非线性变换，输出生成数据."""
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    # 使网络输出数据分布在[-1, 1]之间
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    """使用三层网络对输入数据的来源进行辨别，输出为数据为真的概率."""
    # 在进行前向传播的时候tf中的加法与符号 + 混合使用，是否会造成隐藏问题
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


def plot(samples):
    """对生成样本数据进行可视化（绘图）."""
    # 参数设置及意义？
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# 使用PyCharm提交文件(可以直接在PyCharm中直接commit到Github中)
"""对于loss的计算与公式不同
将输入数据分为真实样本与虚假样本
将两类数据计算得到的loss相加
从而得到真实loss"""
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)
# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
"""使用 0 1 进行labels标记，0：fake 1：real
此处在实现loss函数：
在分类器loss上，是loss越小越好
loss越小，表明二者标签相同（即将真实数据与采样数据均正确分类）
在生成器loss上，loss越小越好
loss越小，表明可以欺骗分类器（将采样数据判定为真）"""
# 真实数据判定为真
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# 采样数据判定为假
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
# 采样数据判定为真
G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
"""需要指定var_list，在每次迭代中更新的参数不同，且只是部分更新"""
# 在此处使用Adam的默认参数
# learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../../data/MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 对于D的循环迭代次数
K = 5

save_path = os.path.join('../../', 'out_G_' + str(K) + '/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

i = 0

for it in range(1000000):
    # 每1000次迭代进行一次抽样输出（每次采样为16个样本）
    if it % 1000 == 0:
        samples = sess.run(G_sample,
                           feed_dict={Z: sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig(save_path + '{}.png'.format(str(i).zfill(3)),
                    bbox_inches='tight')
        i += 1
        plt.close(fig)

    # X_mb, _ = mnist.train.next_batch(mb_size)
    """进行一次分类器优化 进行一次生成器优化
    在分类器优化时，feed真实数据与采样数据
    在生成器优化时，仅feed采样数据"""
    # 在此处对源代码进行修改：使得每对D进行k此迭代后，对G进行一次迭代
    """根据论文算法，在进行K次循环之中，每次都要对真实数据和采样数据进行重新采样，
    而非每次均使用相同的样本进行训练。
    在使用相同的样本训练时，有以下问题：
    1）在K = 1 时，生成样本在0-9之间均匀分布，即在一次输出16个采样中据各个数字
    2）在K = 3 时，生成样本在前期训练较差时，数字分布较为均匀，在训练趋于稳定时，生成样本数据16个均为相同数据
    如：在实验中存在全为1和9的情况
    3）在K= 5 时，生成样本情况与K = 3类似
    当G进行多次训练时：
    1）K = 1 时，网络输出正常
    2）K = 3 时，网络正常，训练较慢
    3）K = 5 时，网络训练失败，输出类似噪声
    D loss: 4.206e-07
    G_loss: 14.71
    随着K值增大，网络的训练难度增大"""

    X_mb, _ = mnist.train.next_batch(mb_size)
    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    for k in range(K):
        # 在每次循环中进行重新采样
        # X_mb, _ = mnist.train.next_batch(mb_size)
        # _, D_loss_curr = sess.run([D_solver, D_loss],
        #                           feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={Z: sample_Z(mb_size, Z_dim)})
# 每1000次迭代，输出一侧分类器与生成器的loss
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
