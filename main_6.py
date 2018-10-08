import tensorflow as tf
import numpy as np
import os

train_dir = 'H:/delun/project_1/tensorflow/img/img2'

#%%
def get_files(file_dir):
    s1 = []
    label_s1 = []
    s2 = []
    label_s2 = []
    s3 = []
    label_s3 = []
    s4 = []
    label_s4 = []

    for file in os.listdir(file_dir+'/s1'):
        s1.append(file_dir+'/s1'+'/'+file)
        label_s1.append(0)
    for file in os.listdir(file_dir+'/s2'):
        s2.append(file_dir+'/s2'+'/'+file)
        label_s2.append(1)
    for file in os.listdir(file_dir + '/s3'):
        s3.append(file_dir+'/s3'+'/'+file)
        label_s3.append(2)
    for file in os.listdir(file_dir + '/s4'):
        s4.append(file_dir+'/s4'+'/'+file)
        label_s4.append(3)
    print('There are %d s1\nThere are %d s2\nThere are %d s3\nThere are %d s4' % (len(s1), len(s2),len(s3),len(s4)))
    # 把s1,s2,s3,s4合起来组成一个list（img和lab）
    image_list = np.hstack((s1, s2, s3, s4))
    label_list = np.hstack((label_s1, label_s2, label_s3, label_s4))
    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # 从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


#%%
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

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   #read img from a queue
    # 将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################
    # 数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # 如果想看到正常的图片，请注释掉78行（标准化）和 89行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    # image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    # 生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size=batch_size,
                                                num_threads = 32,
                                                capacity = capacity)
    # 重新排列label，行数为[batch_size]
    # label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


#%%
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
                                  shape=[11, 11, 3, 96],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[96],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pooling1')
        # norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
        #                   beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 96, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides  =[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
        #                   beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pooling2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 256, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv2')

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)

    # pool5
    with tf.variable_scope('pooling5_lrn') as scope:
        pool5 = tf.nn.max_pool(conv5,
                            ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1],
                            padding='VALID',
                            name='pool5')

    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool5, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[1024, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[512, n_classes],
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
    train_dir = 'H:/delun/project_1/tensorflow/img/img2'
    logs_train_dir = 'H:/delun/project_1/tensorflow/log_6/'

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


#%% Evaluate one image
# when training, comment the following codes.


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

   image = image.resize([224, 224])
   plt.imshow(image)
   plt.show()
   image = np.array(image)
   return image

def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   train_dir = 'H:/delun/project_1/tensorflow/img/img2_test'
   train, train_label = get_files(train_dir)
   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 4

       image = tf.cast(image_array, tf.float32)
       # image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 224, 224, 3])
       logit = inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[224, 224, 3])

       # you need to change the directories to yours.
       logs_train_dir = 'H:/delun/project_1/tensorflow/log_5/'

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
               print('This is a s1 with possibility %.6f' % prediction[:, 0])
           elif max_index == 1:
               print('This is a s2 with possibility %.6f' % prediction[:, 1])
           elif max_index == 2:
               print('This is a s3 with possibility %.6f' % prediction[:, 2])
           else:
               print('This is a s4 with possibility %.6f' % prediction[:, 3])


def get_test_files(file_dir):
    s1 = []
    label_s1 = []
    s2 = []
    label_s2 = []
    s3 = []
    label_s3 = []
    s4 = []
    label_s4 = []

    for file in os.listdir(file_dir+'/s1'):
        s1.append(file_dir+'/s1'+'/'+file)
        label_s1.append(0)
    for file in os.listdir(file_dir+'/s2'):
        s2.append(file_dir+'/s2'+'/'+file)
        label_s2.append(1)
    for file in os.listdir(file_dir + '/s3'):
        s3.append(file_dir+'/s3'+'/'+file)
        label_s3.append(2)
    for file in os.listdir(file_dir + '/s4'):
        s4.append(file_dir+'/s4'+'/'+file)
        label_s4.append(3)
    print('There are %d s1\nThere are %d s2\nThere are %d s3\nThere are %d s4' % (len(s1), len(s2),len(s3),len(s4)))
    # 把s1,s2,s3,s4合起来组成一个list（img和lab）
    image_list = np.hstack((s1, s2, s3, s4))
    label_list = np.hstack((label_s1, label_s2, label_s3, label_s4))
    # 利用shuffle打乱顺序
    # temp = np.array([image_list, label_list])
    # temp = temp.transpose()
    # np.random.shuffle(temp)
    # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def evaluate_all_image():
    '''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    train_dir = 'H:/delun/project_1/tensorflow/img/img2_test'
    train, train_label = get_test_files(train_dir)
    # image_array = get_one_image(train)
    n = len(train)
    for i in range(n):
        img_dir = train[i]
        print('         ',i)
        image = Image.open(img_dir)

        image = image.resize([224, 224])
        image_array = np.array(image)

        with tf.Graph().as_default():
            BATCH_SIZE = 1
            N_CLASSES = 4

            image = tf.cast(image_array, tf.float32)
            # image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, 224, 224, 3])
            logit = inference(image, BATCH_SIZE, N_CLASSES)

            logit = tf.nn.softmax(logit)

            x = tf.placeholder(tf.float32, shape=[224, 224, 3])

            # you need to change the directories to yours.
            logs_train_dir = 'H:/delun/project_1/tensorflow/log_6/'

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
                   print('s1 with possibility %.2f' % prediction[:, 0])
                elif max_index == 1:
                   print('s2 with possibility %.2f' % prediction[:, 1])
                elif max_index == 2:
                   print('s3 with possibility %.2f' % prediction[:, 2])
                else:
                   print('s4 with possibility %.2f' % prediction[:, 3])
#%%


if __name__ == '__main__':
    N_CLASSES = 4
    IMG_W = 224  # resize the image, if the input image is too large, training will be very slow.
    IMG_H = 224
    BATCH_SIZE = 256
    CAPACITY = 512
    MAX_STEP = 5000 # with current parameters, it is suggested to use MAX_STEP>10k
    learning_rate = 0.001 # with current parameters, it is suggested to use learning rate<0.0001

    # run_training()
    # evaluate_one_image()
    evaluate_all_image()
