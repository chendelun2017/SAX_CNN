import tensorflow as tf
import numpy as np
import os
import sys

#直接把print保存在TXT文件中，这样我就可以不用复制黏贴了哈哈哈！
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


train_dir = './img'
np.random.seed(1)   #shuffle打乱的数据顺序不变

def get_files(file_dir):
    c0 = []
    label_c0 = []
    c1 = []
    label_c1 = []
    c2 = []
    label_c2 = []
    c3 = []
    label_c3 = []
    c4 = []
    label_c4 = []
    c5 = []
    label_c5 = []
    c6 = []
    label_c6 = []
    c7 = []
    label_c7 = []
    c8 = []
    label_c8 = []
    c9 = []
    label_c9 = []

    for file in os.listdir(file_dir+'/c0'):
        c0.append(file_dir+'/c0'+'/'+file)
        label_c0.append(0)
    for file in os.listdir(file_dir+'/c1'):
        c1.append(file_dir+'/c1'+'/'+file)
        label_c1.append(1)
    for file in os.listdir(file_dir+'/c2'):
        c2.append(file_dir+'/c2'+'/'+file)
        label_c2.append(2)
    for file in os.listdir(file_dir+'/c3'):
        c3.append(file_dir+'/c3'+'/'+file)
        label_c3.append(3)
    for file in os.listdir(file_dir+'/c4'):
        c4.append(file_dir+'/c4'+'/'+file)
        label_c4.append(4)
    for file in os.listdir(file_dir+'/c5'):
        c5.append(file_dir+'/c5'+'/'+file)
        label_c5.append(5)
    for file in os.listdir(file_dir+'/c6'):
        c6.append(file_dir+'/c6'+'/'+file)
        label_c6.append(6)
    for file in os.listdir(file_dir+'/c7'):
        c7.append(file_dir+'/c7'+'/'+file)
        label_c7.append(7)
    for file in os.listdir(file_dir+'/c8'):
        c8.append(file_dir+'/c8'+'/'+file)
        label_c8.append(8)
    for file in os.listdir(file_dir+'/c9'):
        c9.append(file_dir+'/c9'+'/'+file)
        label_c9.append(9)
    print('There are %d c0\nThere are %d c1\nThere are %d c2\nThere are %d c3\nThere are %d c4\nThere are %d c5\nThere are %d c6\nThere are %d c7\nThere are %d c8\nThere are %d c9' % (len(c0), len(c1), len(c2), len(c3),len(c4), len(c5), len(c6), len(c7),len(c8), len(c9)))

    image_list = np.hstack((c0, c1, c2, c3, c4, c5, c6, c7, c8, c9))
    label_list = np.hstack((label_c0, label_c1, label_c2, label_c3, label_c4, label_c5, label_c6, label_c7, label_c8, label_c9))
    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # # 从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   #read img from a queue
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size=batch_size,
                                                num_threads = 32,
                                                capacity = capacity)
    return image_batch, label_batch


def inference(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 1, 6],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[6],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pooling1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 6, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides  =[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pooling2')

    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 120],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[120],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[120, 84],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[84],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[84, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


#%%
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy


def run_training():

    train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    train_logits = inference(train_batch, BATCH_SIZE , N_CLASSES)
    train_loss = losses(train_logits, train_label_batch)
    train_op = trainning(train_loss, learning_rate)
    train__acc = evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 10 == 0:
                print('Step %d, train loss = %.6f, train accuracy = %.6f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 100 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


from PIL import Image
import matplotlib.pyplot as plt

def evaluate_all_image():
    '''Test one image against the saved models and parameters
    '''
    n = len(test)
    c0_orig, c1_orig, c2_orig, c3_orig, c4_orig, c5_orig, c6_orig, c7_orig, c8_orig, c9_orig = 0,0,0,0,0,0,0,0,0,0
    c0_true, c1_true, c2_true, c3_true, c4_true, c5_true, c6_true, c7_true, c8_true, c9_true = 0,0,0,0,0,0,0,0,0,0
    c0_false, c1_false, c2_false, c3_false, c4_false, c5_false, c6_false, c7_false, c8_false, c9_false = 0,0,0,0,0,0,0,0,0,0

    for i in range(n):
        img_dir = test[i]
        if test_label[i] == 0:
            c0_orig += 1
        elif test_label[i] == 1:
            c1_orig += 1
        elif test_label[i] == 2:
            c2_orig += 1
        elif test_label[i] == 3:
            c3_orig += 1
        elif test_label[i] == 4:
            c4_orig += 1
        elif test_label[i] == 5:
            c5_orig += 1
        elif test_label[i] == 6:
            c6_orig += 1
        elif test_label[i] == 7:
            c7_orig += 1
        elif test_label[i] == 8:
            c8_orig += 1
        else:
            c9_orig += 1

        image = Image.open(img_dir)

        image_array = np.array(image)

        with tf.Graph().as_default():
            BATCH_SIZE = 1
            N_CLASSES = 10

            image = tf.cast(image_array, tf.float32)
            image = tf.reshape(image, [1, 32, 32, 1])
            logit = inference(image, BATCH_SIZE, N_CLASSES)

            logit = tf.nn.softmax(logit)

            x = tf.placeholder(tf.float32, shape=[32, 32])

            saver = tf.train.Saver()

            with tf.Session() as sess:

                # print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')

                prediction = sess.run(logit, feed_dict={x: image_array})
                max_index = np.argmax(prediction)
                # print(prediction)
                if max_index == 0:
                    if max_index == test_label[i]:
                        c0_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c0_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (test_label[i], max_index, prediction[:, 0]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 1:
                    if max_index == test_label[i]:
                        c1_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c1_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (test_label[i], max_index, prediction[:, 1]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 2:
                    if max_index == test_label[i]:
                        c2_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c2_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 2]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 3:
                    if max_index == test_label[i]:
                        c3_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c3_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 3]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 4:
                    if max_index == test_label[i]:
                        c4_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c4_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 4]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 5:
                    if max_index == test_label[i]:
                        c5_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c5_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 5]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 6:

                    if max_index == test_label[i]:
                        c6_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c6_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 6]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 7:

                    if max_index == test_label[i]:
                        c7_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c7_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 7]))
                        confusion_mat[test_label[i]][max_index] += 1
                elif max_index == 8:

                    if max_index == test_label[i]:
                        c8_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c8_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 8]))
                        confusion_mat[test_label[i]][max_index] += 1
                else:
                    if max_index == test_label[i]:
                        c9_true += 1
                        confusion_mat[test_label[i]][max_index] += 1
                    else:
                        c9_false += 1
                        print('The real label is %d, and the predict label is %d, with possibility %.6f' % (
                        test_label[i], max_index, prediction[:, 9]))
                        confusion_mat[test_label[i]][max_index] += 1
    print(c0_orig, c1_orig, c2_orig, c3_orig, c4_orig, c5_orig, c6_orig, c7_orig, c8_orig, c9_orig)
    print(c0_true, c1_true, c2_true, c3_true, c4_true, c5_true, c6_true, c7_true, c8_true, c9_true)
    print(c0_false, c1_false, c2_false, c3_false, c4_false, c5_false, c6_false, c7_false, c8_false, c9_false)


if __name__ == '__main__':
    #把print写入到txt
    sys.stdout = Logger("result.txt")
    #定义一个误差矩阵
    confusion_mat = np.zeros([10, 10])

    N_CLASSES = 10
    IMG_W = 32  # resize the image, if the input image is too large, training will be very slow.
    IMG_H = 32
    BATCH_SIZE = 512 #512,1024
    CAPACITY = 1024
    MAX_STEP = 1510  # with current parameters, it is suggested to use MAX_STEP>10k
    learning_rate = 0.005  # with current parameters, it is suggested to use learning rate<0.0001

    train_dir = './img/'
    logs_train_dir = './log/log_lenet/'

    data, data_label = get_files(train_dir)
    num = int(len(data) / 10)  # 设置交叉验证集

    #第一个交叉验证
    train, train_label, test, test_label = [], [], [], []
    train = data[num + 1:num * 10 + 1]
    train_label = data_label[num + 1:num * 10 + 1]

    test = data[1:num+1]
    test_label = data_label[1:num+1]
    run_training()
    evaluate_all_image()

    #第二个交叉验证
    tf.reset_default_graph()    #重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num+1]
    train = data[num*2+1:num*10 + 1]
    train_label = data_label[1:num+1]
    train_label = data_label[num*2+1:num*10 + 1]

    test = data[num:num*2+1]
    test_label = data_label[num:num*2+1]

    run_training()
    evaluate_all_image()

    # 第三个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*2 + 1]
    train_label = data_label[1:num*2 + 1]
    train = data[num*3 + 1:num*10 + 1]
    train_label = data_label[num * 3 + 1:num * 10 + 1]

    test = data[num*2 + 1:num * 3 + 1]
    test_label = data_label[num*2 + 1:num * 3 + 1]

    run_training()
    evaluate_all_image()

    # 第4个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*3 + 1]
    train_label = data_label[1:num*3 + 1]
    train = data[num*4 + 1:num*10 + 1]
    train_label = data_label[num * 4 + 1:num * 10 + 1]

    test = data[num*3 + 1:num * 4 + 1]
    test_label = data_label[num*3 + 1:num * 4 + 1]

    run_training()
    evaluate_all_image()

    # 第5个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*4 + 1]
    train_label = data_label[1:num*4 + 1]
    train = data[num*5 + 1:num*10 + 1]
    train_label = data_label[num * 5 + 1:num * 10 + 1]

    test = data[num*4 + 1:num * 5 + 1]
    test_label = data_label[num*4 + 1:num * 5 + 1]

    run_training()
    evaluate_all_image()

    # 第6个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*5 + 1]
    train_label = data_label[1:num*5 + 1]
    train = data[num*6 + 1:num*10 + 1]
    train_label = data_label[num * 6 + 1:num * 10 + 1]

    test = data[num*5 + 1:num * 6 + 1]
    test_label = data_label[num*5 + 1:num * 6 + 1]

    run_training()
    evaluate_all_image()

    # 第7个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*6 + 1]
    train_label = data_label[1:num*6 + 1]
    train = data[num*7 + 1:num*10 + 1]
    train_label = data_label[num * 7 + 1:num * 10 + 1]

    test = data[num*6 + 1:num * 7 + 1]
    test_label = data_label[num*6 + 1:num * 7 + 1]

    run_training()
    evaluate_all_image()

    # 第8个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*7 + 1]
    train_label = data_label[1:num*7 + 1]
    train = data[num*8 + 1:num*10 + 1]
    train_label = data_label[num * 8 + 1:num * 10 + 1]

    test = data[num*7 + 1:num * 8 + 1]
    test_label = data_label[num*7 + 1:num * 8 + 1]

    run_training()
    evaluate_all_image()

    # 第9个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*8 + 1]
    train_label = data_label[1:num*8 + 1]
    train = data[num*9 + 1:num*10 + 1]
    train_label = data_label[num * 9 + 1:num * 10 + 1]

    test = data[num*8 + 1:num * 9 + 1]
    test_label = data_label[num*8 + 1:num * 9 + 1]

    run_training()
    evaluate_all_image()

    # 第10个交叉验证
    tf.reset_default_graph()  # 重启tensorflow，否则会报错
    train, train_label, test, test_label = [], [], [], []

    train = data[1:num*9 + 1]
    train_label = data_label[1:num*9 + 1]

    test = data[num*9 + 1:num * 10 + 1]
    test_label = data_label[num*9 + 1:num * 10 + 1]

    run_training()
    evaluate_all_image()

    #输出误差矩阵,x方向为预测类，y方向为实际类
    print(confusion_mat)