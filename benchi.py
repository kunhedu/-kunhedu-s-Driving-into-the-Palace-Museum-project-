import logging
import re

import jieba

from gensim import corpora
from gensim.models import LdaModel
from openpyxl import load_workbook
import pyLDAvis.gensim_models
wb = load_workbook("benchixin.xlsx")



sheet = wb.get_sheet_by_name("Sheet1")
raw_corpus = []
for i in sheet["A"]:
  # print(i.value, end=" ")    # c1 c2 c3 c4 c5 c6 c7 c8 c9 c10     <-C列中的所有值
  try:
    raw_corpus.append(i.value)

  except:
    continue



# for sentence in raw_corpus:
#     print(list(jieba.cut(sentence)))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
corpus = []
jieba.add_word('大奔')
jieba.add_word('闭馆日')
jieba.add_word('露小宝')
jieba.add_word('西华门')
jieba.add_word('忠旺集团')
jieba.add_word('长春理工大学')
jieba.add_word('副处长')
jieba.add_word('副院长')
jieba.add_word('未通过')
jieba.add_word('红三代')
jieba.add_word('柠檬酸')
jieba.add_word('撒欢儿')
jieba.add_word('晒照片')
jieba.add_word('未取得')
jieba.add_word('白岩松')
jieba.add_word('停职检查')
jieba.add_word('权贵')
jieba.add_word('大奔')
jieba.add_word('大G')
jieba.add_word('千万名表')
jieba.add_word('吃瓜')
jieba.add_word('内景')
jieba.add_word('热搜')
jieba.add_word('特权')
jieba.add_word('非物质')
jieba.add_word('文化遗产')
jieba.add_word('内景视频')

stop_words = [
    '的', '了', '我', '你', '这', '也', '是', '谁', '呢', '后', '么',
    '啊', '哎', '该', '被', '着', '就', '可', '还', '去', '却','她','跟',
    '对','向','吗','与','并','露','年','能','上','曾','因','但','到','给','过','抖','说',
    '为','一个','已','不是','这个','让','我们','还是','评','更','真','还会',
    '好','们','要','个','又','下','会','是否','什么','那么','就是','这种',
    '门','从','前','这么','如何','听','及','自己','可以','这样','没有','或','多',
    '来','很','还有','事情','只是','看到','时','把','拍','称','而','活动','作为',
    '那件','等','门','看','事','住','早','另','那些','因为','得','将','来',
    '开进','进入','时候','那','娶','逼','你们','人家','引起','这些','任何','进去',
    '人们','太和','继续','其','怎么','应该','临时','接受','还原','大家','如果','他们',
    '他','存在','刚刚','发','里','瓜','广场','那个','他','开大','既然','怎么办',
    '想','涉及','问题','汽车','公司','照片','哪里','至少','交代','解释','晒','或者',
    '显示','原来','发声','西华门','必须','而且','可能','一样','一定','像','之',
    '大家','大众','公众','一下','一点','根本','出来','拿','事儿','总之','结果','声明',
    '一直','社会','文化','觉得','属于','罢了','而已','请','所以','所','这是','属实',
    '不要','没事','开着','居然','依然','很多','非常','可是','一边','下来','需要','用户','均',
    '文物','位于','关于','国家', '说放车','发生','东西','伏虎','玻璃','停车场','希望','现代','通道',
    '简直','而是','中福','发文','知道','事件','现在','不能','原定','故宫','女子','相关','发布','评论','当年',
    '网友','当事人','三颗','我院','周一','开车','这件','起来','真的','露小宝','已经','过去','但是',
    '新闻','女主','微博','学校','中国','过场','到底','高露','还要','关心','为什么','玻璃','是不是',
    '关注','方面','多年','发现','亲历者','面前','保卫','分管','竟然','破坏','也许','清楚','有余','并且',
    '晒开','导演',
    
    
        ]  # 定义停止词

def not_stop_word(word):
    if word in stop_words:
        return False
    elif len(word) <= 1:
        return False
    return True


for sentence in raw_corpus:
    sentence = ''.join(re.findall(r'[\u4e00-\u9fa5]+', sentence))  # 仅保留中文
    
    corpus.append([item for item in jieba.cut(sentence) if not_stop_word(item)])  # 去掉停止词
for sentence in corpus:
    print(sentence)


def wordcount(word):
    c = 0
    for l in corpus:
        for w in l:
            if w == word:
                c += 1
    return c

print(wordcount('特权'))  
print(wordcount('权贵'))  
print(wordcount('红三代'))  
print(wordcount('豪宅'))  


dictionary = corpora.Dictionary(corpus)
# dictionary.save('qzone.dict')  # 把字典存储下来，可以在以后直接导入
print(dictionary)

#dictionary.filter_extremes(no_below=1)  # 删掉只在不超过20个文本中出现过的词，删掉在50%及以上的文本都出现了的词
# # dictionary.filter_tokens(['一个'])  # 这个函数可以直接删除指定的词
dictionary.compactify()  # 去掉因删除词汇而出现的空白

corpus = [dictionary.doc2bow(s) for s in corpus]
corpora.MmCorpus.serialize('corpus_bow.mm', corpus)  # 存储语料库



def wordcount(word):
    try:
        idxx = dictionary.token2id[word]
    except:
        return -1
    c = 0
    for l in corpus:
        for w in l:
            if w[0] == idxx:
                c += w[1]
    return c


print("============================================")
print(wordcount('特权'))  
print(wordcount('权贵'))  
print(wordcount('红三代'))  
print(wordcount('豪宅')) 

num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = 80  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

model.save('qzone.model')  # 将模型保存到硬盘

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.80f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)



vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
# 需要的三个参数都可以从硬盘读取的，前面已经存储下来了
pyLDAvis.save_html(vis, 'lda.html')
pyLDAvis.show(vis)

# pyLDAvis.save_html(vis, 'lda.html')
