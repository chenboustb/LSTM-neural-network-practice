import collections

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
    print(len(poems))
    print(poems)
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
    #因为后面要把那些长度短的诗补上空格，所以把空格也加到我们的字符库中
    words=words[:len(words)]+(' ',)
    #print(words)
    #这两个字典非常关键，神经网络无法识别字符，通过这两个字典可以完成字符与数值的双向转换
    word_to_id=dict(zip(words,range(len(words))))
    id_to_word=dict(zip(word_to_id.values(),word_to_id.keys()))
    #生成诗的向量，把所有诗的每一个字符都用特定的数字表示
    poems_vec=[[word_to_id[word] for word in poem] for poem in poems]
    print(poems_vec)
    #返回诗向量和两个字典
    return poems_vec,word_to_id,id_to_word

if __name__=='__main__':
    read_poems()