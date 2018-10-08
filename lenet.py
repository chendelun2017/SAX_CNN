import tensorflow as tf
import numpy as np
import os

train_dir = './img'

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
    # 从打乱的temp中再取出list（img和lab）
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
    image = tf.image.per_image_standardization(image)
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
        # norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
        #                   beta=0.75, name='norm1')

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
        # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
        #                   beta=0.75, name='norm2')
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
    train_dir = './img/'
    logs_train_dir = './img/log/'

    train, train_label = get_files(train_dir)
    train_batch, train_label_batch = get_batch(train,
                                               train_label,
                                               IMG_W,
                                               IMG_H,
                                               BATCH_SIZE,
                                               CAPACITY)
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

def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)

   plt.imshow(image)
   plt.show()
   image = np.array(image)
   return image


def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   test_dir = './img/'
   test, test_label = get_files(test_dir)
   image_array = get_one_image(test)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 10

       image = tf.cast(image_array, tf.float32)
       # image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 32, 32, 1])
       logit = inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[32, 32])

       # you need to change the directories to yours.
       logs_train_dir = './img/log/'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           print(prediction)
           if max_index == 0:
               print('This is a c0 with possibility %.6f' % prediction[:, 0])
           elif max_index == 1:
               print('This is a c1 with possibility %.6f' % prediction[:, 1])
           elif max_index == 2:
               print('This is a c2 with possibility %.6f' % prediction[:, 2])
           elif max_index == 3:
               print('This is a c3 with possibility %.6f' % prediction[:, 3])
           elif max_index == 4:
               print('This is a c4 with possibility %.6f' % prediction[:, 4])
           elif max_index == 5:
               print('This is a c5 with possibility %.6f' % prediction[:, 5])
           elif max_index == 6:
               print('This is a c6 with possibility %.6f' % prediction[:, 6])
           elif max_index == 7:
               print('This is a c7 with possibility %.6f' % prediction[:, 7])
           elif max_index == 8:
               print('This is a c8 with possibility %.6f' % prediction[:, 8])
           else:
               print('This is a c9 with possibility %.6f' % prediction[:, 9])


def get_test_files(file_dir):
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
    # temp = np.array([image_list, label_list])
    # temp = temp.transpose()
    # np.random.shuffle(temp)
    # # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]

    return image_list, label_list


def evaluate_all_image():
    '''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    test_dir = './img/'
    train, train_label = get_test_files(test_dir)
    # image_array = get_one_image(train)
    n = len(train)
    for i in range(n):
        img_dir = train[i]
        print('         ',i)
        image = Image.open(img_dir)

        image = image.resize([32, 32])
        image_array = np.array(image)

        with tf.Graph().as_default():
            BATCH_SIZE = 1
            N_CLASSES = 10

            image = tf.cast(image_array, tf.float32)
            # image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, 32, 32, 1])
            logit = inference(image, BATCH_SIZE, N_CLASSES)

            logit = tf.nn.softmax(logit)

            x = tf.placeholder(tf.float32, shape=[32, 32])

            # you need to change the directories to yours.
            logs_train_dir = './img/log/'

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
                    print('This is a c0 with possibility %.6f' % prediction[:, 0])
                elif max_index == 1:
                    print('This is a c1 with possibility %.6f' % prediction[:, 1])
                elif max_index == 2:
                    print('This is a c2 with possibility %.6f' % prediction[:, 2])
                elif max_index == 3:
                    print('This is a c3 with possibility %.6f' % prediction[:, 3])
                elif max_index == 4:
                    print('This is a c4 with possibility %.6f' % prediction[:, 4])
                elif max_index == 5:
                    print('This is a c5 with possibility %.6f' % prediction[:, 5])
                elif max_index == 6:
                    print('This is a c6 with possibility %.6f' % prediction[:, 6])
                elif max_index == 7:
                    print('This is a c7 with possibility %.6f' % prediction[:, 7])
                elif max_index == 8:
                    print('This is a c8 with possibility %.6f' % prediction[:, 8])
                else:
                    print('This is a c9 with possibility %.6f' % prediction[:, 9])


if __name__ == '__main__':
    N_CLASSES = 10
    IMG_W = 32  # resize the image, if the input image is too large, training will be very slow.
    IMG_H = 32
    BATCH_SIZE = 1024
    CAPACITY = 2048
    MAX_STEP = 5000  # with current parameters, it is suggested to use MAX_STEP>10k
    learning_rate = 0.009  # with current parameters, it is suggested to use learning rate<0.0001

    run_training()
    # evaluate_one_image()
    # evaluate_all_image()