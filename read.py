import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import collections
import datetime
import os
import sys


def read_poems(poems_file='data/poems.txt'):

    poems=[]
    with open(poems_file,'r',encoding='utf-8') as f:
        #读取每一行
        for line in f.readlines():
            #print(line)
            #会产生ValueException，需要用try Exception捕捉一下
            try:
                #使用split将诗的名字和诗的内容分离，名字后面就用不到了
                title,content=line.strip().split(':')
                #去除空格
                content=content.replace(' ','')
                #防止出现奇怪的字符
                if '_' in content or '(' in content or \
                '（'in content or '《' in content or \
                '['in content : continue
                if len(content)<5 or len(content)>80:
                    continue
                #设置起始字符和结束字符，方便到时候神经网络读取
                content='['+content+']'
                poems.append(content)
            except Exception as e:
                pass
    #对长度进行排序
    poems=sorted(poems,key=lambda x:len(x))
    #print(len(poems))
    #print(poems)
    words=[]
    #分离成字符
    for poem in poems:
        words+=[word for word in poem]
    #print(words)
    #统计每个字符出现的次数然后排序
    counter=collections.Counter(words)
    count_table=sorted(counter.items(),key=lambda x:-x[1])
    #print(count_table)
    words,t=zip(*count_table)
    #print(words)
    #因为后面要把那些长度短的诗补上空格，所以
    # 把空格也加到我们的字符库中
    words=words[:len(words)]+(' ',)
    #print(words)
    #这两个字典非常关键，神经网络无法识别字符，通过这两个字典可以完成字符与数值的双向转换
    word_to_id=dict(zip(words,range(len(words))))
    id_to_word=dict(zip(word_to_id.values(),word_to_id.keys()))
    #生成诗的向量，把所有诗的每一个字符都用特定的数字表示
    poems_vec=[[word_to_id[word] for word in poem] for poem in poems]
    print(word_to_id)
    #print(poems_vec)
    #返回诗向量和两个字典
    return poems_vec,word_to_id,id_to_word


def create_batch(batch_size,poems_vec,word_to_id):

    n_chunk=len(poems_vec)//batch_size
    x_batch=[]
    y_batch=[]
    for i in range(n_chunk):
        start_index=i*batch_size
        end_index=start_index+batch_size
        batches=poems_vec[start_index:end_index]
        length=max(map(len,batches))
        x_data = np.full((batch_size, length), word_to_id.get(' '), np.int32)
        for row in range(batch_size):
            x_data[row,:len(batches[row])]=batches[row]
        y_data=np.copy(x_data)
        y_data[:,:-1]=x_data[:,1:]
        x_batch.append(x_data)
        y_batch.append(y_data)
    return x_batch,y_batch


def model(input_data,output_data,word_size):

    rnn_size=128
    num_layers=2
    batch_size=64
    learning_rate=0.01

    outs={}
    cell_fun=tf.nn.rnn_cell.BasicLSTMCell
    cell=cell_fun(rnn_size,state_is_tuple=True)
    cell=tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers,state_is_tuple=True)
    initial_state=cell.zero_state(batch_size,tf.float32)
    embedding=tf.Variable(tf.random_uniform([word_size+1,rnn_size],-1.0,1.0))
    inputs=tf.nn.embedding_lookup(embedding,input_data)

    outputs,last_state=tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
    outputs=tf.reshape(outputs,[-1,rnn_size])

    w=tf.Variable(tf.truncated_normal([rnn_size,word_size+1]))
    b=tf.Variable(tf.zeros(shape=[word_size+1]))
    logits=tf.nn.bias_add(tf.matmul(outputs,w),b)

    labels=tf.one_hot(tf.reshape(output_data,[-1]),depth=word_size+1)
    loss=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    mean_loss=tf.reduce_mean(loss)
    opti=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

    outs['initial_state']=initial_state
    outs['outputs']=outputs
    outs['mean_loss']=mean_loss
    outs['loss']=loss
    outs['last_state']=last_state
    outs['train']=opti

    return outs


def train():

    epochs=100

    poems_vec, word_to_id, id_to_word = read_poems()
    x_batch, y_batch = create_batch(batch_size=64, poems_vec=poems_vec, word_to_id=word_to_id)

    batch_size = 64

    poems_vec, word_to_id, id_to_word = read_poems()
    input_data=tf.placeholder(tf.int32,[batch_size,None])
    output_data=tf.placeholder(tf.int32,[batch_size,None])

    outs = model(input_data=input_data,output_data=output_data,word_size=len(id_to_word))

    saver=tf.train.Saver(tf.global_variables())
    init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        start_epoch=0
        checkpoint=tf.train.latest_checkpoint('model/')

        if checkpoint:
            saver.restore(sess,checkpoint)
            print('载入数据点{}'.format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('开始训练')

        try:
            for epoch in range(start_epoch,epochs):
                n = 0
                n_chunk = len(poems_vec) // batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        outs['mean_loss'],
                        outs['last_state'],
                        outs['train']
                    ], feed_dict={input_data: x_batch[n], output_data: y_batch[n]})
                    n += 1
                    print('迭代周期: %d , 样本: %d , 损失函数: %.6f' % (epoch, batch, loss))
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join('model/','model_prefix'), global_step=epoch)
        except KeyboardInterrupt:
            print('开始手动保存')
            saver.save(sess, os.path.join('model/', 'model_prefix'), global_step=epoch)
            print('手动保存了保存点{}.'.format(epoch))


def predict(input_data,word_size):
    rnn_size = 128
    num_layers = 2

    outs = {}
    cell_fun = tf.nn.rnn_cell.BasicLSTMCell
    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    initial_state = cell.zero_state(1, tf.float32)
    embedding = tf.Variable(tf.random_uniform([word_size + 1, rnn_size], -1.0, 1.0))
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    outputs = tf.reshape(outputs, [-1, rnn_size])

    w = tf.Variable(tf.truncated_normal([rnn_size, word_size + 1]))
    b = tf.Variable(tf.zeros(shape=[word_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(outputs, w), b)

    pred=tf.nn.softmax(logits=logits)

    outs['initial_state']=initial_state
    outs['last_state']=last_state
    outs['pred']=pred

    return outs


def converse(pred,id_to_word):
    t=np.cumsum(pred)
    s=np.sum(pred)
    sample=int(np.searchsorted(t,np.random.rand(1)*s))
    if sample>len(id_to_word):
        sample=len(id_to_word)-1
    return id_to_word[sample]


def write(begin_word):

    batch_size=1
    print('大诗人思考中...')
    poems_vec, word_to_id, id_to_word = read_poems()

    input_data=tf.placeholder(tf.int32,[batch_size,None])

    outs=predict(input_data=input_data,word_size=len(id_to_word))

    saver=tf.train.Saver(tf.global_variables())
    init=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)

        checkpoint=tf.train.latest_checkpoint('model/')
        saver.restore(sess,checkpoint)

        x = np.array([list(map(word_to_id.get, '['))])
        [pred,last_state]=sess.run([outs['pred'],outs['last_state']],
                                   feed_dict={input_data:x})
        if begin_word:
            word=begin_word
        else:
            word=converse(pred,id_to_word)

        poem=''
        while word!=']':
            poem+=word
            x=np.zeros((1,1))
            x[0,0]=word_to_id.get(word)
            [pred, last_state] = sess.run([outs['pred'], outs['last_state']],
                            feed_dict={input_data:x,outs['initial_state']:last_state})
            word=converse(pred,id_to_word)
        return poem


if __name__=='__main__':
    # train()
    begin_word=str(input())
    poem=write(begin_word)
    print(poem)
    print(len(poem))