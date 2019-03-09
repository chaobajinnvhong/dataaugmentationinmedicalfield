import tensorflow as tf
import numpy as np
import csv

# data=pd.read_csv('4.csv')
# dataset = pd.DataFrame(columns=data.columns)
# for i in range(5):
#     print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#     temp=pd.read_csv('%d.csv'%i)
#     dataset=pd.concat([dataset,temp])
#     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# dataset.to_csv('ll.csv',index=False)
# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# dataset=pd.read_csv('ll.csv')
# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# print(dataset)



# csv_file = csv.reader(open('0.csv', 'r'))
# for i in csv_file:
#     print(i)
#     break


#
# ss=[]
# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# csv_file = csv.reader(open('ll.csv', 'r'))
# j=0
# for i in csv_file:
#     # print(i)
#     j+=1
#     if j%10000==0:
#         print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
def watch(object):
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(object))

def hebing01234():
    ss=[]
    csv_file = csv.reader(open('0.csv', 'r'))
    for i in csv_file:
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        ss.append(i)
        break
    for i in range(5):
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        csv_file = csv.reader(open('%d.csv' % i, 'r'))
        j = 0
        for line in csv_file:
            if j == 0:
                j += 1
                continue
            ss.append(line)
    tt = ss[1:]
    print("ok")
    return tt

def shujuload(tt):
    all = []
    label = []
    data_hot = []
    for leibie0 in np.random.randint(0, 358680,20000):
        all.append(leibie0)
    for leibie1 in np.random.randint(358680,469112,20000):
        all.append(leibie1)
    for leibie2 in np.random.randint(469112,625352,20000):
        all.append(leibie2)
    for leibie3 in np.random.randint(625352,794072,20000):
        all.append(leibie3)
    for leibie4 in np.random.randint(794072,1011128,20000):
        all.append(leibie4)
    np.random.shuffle(all)
    # for leibie0 in np.random.randint(0, 265,200):
    #     all.append(leibie0)
    # for leibie1 in np.random.randint(265,314,200):
    #     all.append(leibie1)
    # for leibie2 in np.random.randint(314,372,200):
    #     all.append(leibie2)
    # for leibie3 in np.random.randint(372,434,200):
    #     all.append(leibie3)
    # for leibie4 in np.random.randint(434,507,200):
    #     all.append(leibie4)
    # np.random.shuffle(all)
    # print(all)
    for k in all:
        label.append(float(tt[k][-1]))
        zhong = []
        for t in range(len(tt[0]) - 1):  #
            zhong.append(float(tt[k][t]))
        for _ in range(8):
            zhong.append(0.0)
        data_hot.append(zhong)
    label = np.array(label, dtype='float32')
    label_hot = tf.one_hot(label, depth=5)
    return data_hot,label_hot

def testload():
    kk = csv.reader(open('0.csv', 'r'))
    ss = []
    label=[]
    i=0
    for e in kk:
       if i ==0:
           i+=1
           continue
       ss.append(e)
    ss=np.array(ss)
    for i in range(len(ss)):
        label.append(float(ss[i][-1]))
    label = np.array(label,dtype='float32')
    testlabel_hot = tf.one_hot(label, depth=5)
    # watch(label_hot)
    test_hot = []

    for j in range(len(ss)):  # 507
        zhong = []
        for i in range(len(ss[0]) - 1):  #
            zhong.append(float(ss[j][i]))
        for k in range(8):
            zhong.append(0.0)
        test_hot.append(zhong)
    test_hot = np.array(test_hot,dtype='float32')
    return test_hot,testlabel_hot


def testload1(tt):
    all = []
    label = []
    data_hot = []
    for leibie0 in np.random.randint(0, 100000, 200):
        all.append(leibie0)
    for leibie1 in np.random.randint(360000, 400000, 200):
        all.append(leibie1)
    for leibie2 in np.random.randint(469112, 500000, 200):
        all.append(leibie2)
    for leibie3 in np.random.randint(625352, 700000, 200):
        all.append(leibie3)
    for leibie4 in np.random.randint(794072, 900000, 200):
        all.append(leibie4)
    np.random.shuffle(all)
    for k in all:
        label.append(float(tt[k][-1]))
        zhong = []
        for t in range(len(tt[0]) - 1):  #
            zhong.append(float(tt[k][t]))
        for _ in range(8):
            zhong.append(0.0)
        data_hot.append(zhong)
    label = np.array(label, dtype='float32')
    label_hot = tf.one_hot(label, depth=5)
    return data_hot, label_hot

def testload2():
    kk = csv.reader(open('rawdata1sort.csv', 'r'))
    ss = []
    label = []
    i = 0
    for e in kk:
        if i == 0:
            i += 1
            continue
        ss.append(e)
    ss = np.array(ss)
    for i in range(len(ss)):
        if i==20:
            for i in range(265,285):
                label.append(float(ss[i][-1]))
            for i in range(314,334):
                label.append(float(ss[i][-1]))
            for i in range(370,390):
                label.append(float(ss[i][-1]))
            for i in range(428,448):
                label.append(float(ss[i][-1]))
            break
        label.append(float(ss[i][-1]))
    print(len(label))
    label = np.array(label, dtype='float32')
    testlabel_hot = tf.one_hot(label, depth=5)
    # watch(label_hot)

    test_hot = []

    for j in range(len(ss)):  # 507
        if j==20:
            for j in range(265,285):
                zhong = []
                for i in range(len(ss[0]) - 1):  #
                    zhong.append(float(ss[j][i]))
                for k in range(8):
                    zhong.append(0.0)
                test_hot.append(zhong)
            for j in range(314,334):
                zhong = []
                for i in range(len(ss[0]) - 1):  #
                    zhong.append(float(ss[j][i]))
                for k in range(8):
                    zhong.append(0.0)
                test_hot.append(zhong)
            for j in range(370,390):
                zhong = []
                for i in range(len(ss[0]) - 1):  #
                    zhong.append(float(ss[j][i]))
                for k in range(8):
                    zhong.append(0.0)
                test_hot.append(zhong)
            for j in range(428,448):
                zhong = []
                for i in range(len(ss[0]) - 1):  #
                    zhong.append(float(ss[j][i]))
                for k in range(8):
                    zhong.append(0.0)
                test_hot.append(zhong)
            break
        zhong = []
        for i in range(len(ss[0]) - 1):  #
            zhong.append(float(ss[j][i]))
        for k in range(8):
            zhong.append(0.0)
        test_hot.append(zhong)
    test_hot = np.array(test_hot, dtype='float32')
    return test_hot, testlabel_hot

# test_hot,testlabel_hot=testload2()


def batch(data,label,num):
    data=data[num:num+10000]
    label=label[num:num+10000]
    return data,label

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 144])   # 12x12
ys = tf.placeholder(tf.float32, [None, 5])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1,12, 12, 1])

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,8]) # patch 5x5
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2 = weight_variable([5,5, 8, 16]) # patch 5x5
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #
h_pool2 = max_pool_2x2(h_conv2)

## fc1 layer ##
W_fc1 = weight_variable([3*3*16, 64])
b_fc1 = bias_variable([64])
h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([64, 5])
b_fc2 = bias_variable([5])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)



if __name__ == '__main__':
    tt = hebing01234()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for batch in range(1950):
            data_hot,label_hot = shujuload(tt)
            test_hot, testlabel_hot = testload2()
            label_hot = label_hot.eval(session=sess)
            testlabel_hot = testlabel_hot.eval(session=sess)
            for i in range(50):
                batch_xs, batch_ys = data_hot,label_hot
                sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.7})
                out = open('result.csv', 'a', newline='')
                csv_write = csv.writer(out, dialect='excel')
                kk=compute_accuracy(test_hot, testlabel_hot)
                print(kk,"      " )
                csv_write.writerow((str(kk)))
    #
    # with tf.Session() as sess:
    #     tt = hebing01234()
    #     data_hot, label_hot = shujuload(tt)
    #     test_hot, testlabel_hot = testload()
    #     label_hot = label_hot.eval(session=sess)
    #     testlabel_hot = testlabel_hot.eval(session=sess)
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #     # j=0
    #     for i in range(10000):
    #         batch_xs, batch_ys = data_hot,label_hot
    #         # j += 10000
    #         # if j==1000000:
    #         #     j=0
    #         sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.7})
    #         kk = compute_accuracy(test_hot, testlabel_hot)
    #         print(kk)
    #         # out = open('result.csv', 'a', newline='')
    #         # csv_write = csv.writer(out, dialect='excel')
    #         # kk=compute_accuracy(test_hot, testlabel_hot)
    #         # csv_write.writerow((str(kk)))
    #
    #
    #