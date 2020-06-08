import pandas as pd
import numpy as np

def get_explain_dic():
    dic={}
    dic['CC']=  'Coordinating conjunction 连接词'
    dic['CD']=  'Cardinal number  基数词'
    dic['DT']=  'Determiner  限定词'
    dic['EX']=  'Existential there 存在句'
    dic['FW']=  'Foreign word 外来词'
    dic['IN']=  'Preposition or subordinating conjunction 介词或从属连词'
    dic['JJ']=  'Adjective 形容词或序数词'
    dic['JJR']= 'Adjective, comparative 形容词比较级'
    dic['JJS']= 'Adjective, superlative 形容词最高级'
    dic['LS']=  'List item marker 列表标示'
    dic['MD']=  'Modal 情态助动词'
    dic['NN']=  'Noun, singular or mass 常用名词 单数形式'
    dic['NNS']= 'Noun, plural  常用名词 复数形式'
    dic['NNP']= 'Proper noun, singular  专有名词，单数形式'
    dic['NNPS']='Proper noun, plural  专有名词，复数形式'
    dic['PDT']= 'Predeterminer 前位限定词'
    dic['POS']= 'Possessive ending 所有格结束词'
    dic['PRP']= 'Personal pronoun 人称代词'
    dic['PRP$']='Possessive pronoun 所有格代名词'
    dic['RB']=  'Adverb 副词'
    dic['RBR']= 'Adverb, comparative 副词比较级'
    dic['RBS']= 'Adverb, superlative 副词最高级'
    dic['RP']=  'Particle 小品词'
    dic['SYM']= 'Symbol 符号'
    dic['TO']=  'to 作为介词或不定式格式'
    dic['UH']=  'Interjection 感叹词'
    dic['VB']=  'Verb, base form 动词基本形式'
    dic['VBD']= 'Verb, past tense 动词过去式'
    dic['VBG']= 'Verb, gerund or present participle 动名词和现在分词'
    dic['VBN']= 'Verb, past participle 过去分词'
    dic['VBP']= 'Verb, non-3rd person singular present 动词非第三人称单数'
    dic['VBZ']= 'Verb, 3rd person singular present 动词第三人称单数'
    dic['WDT']= 'Wh-determiner 限定词'
    dic['WP']=  'Wh-pronoun 代词'
    dic['WP$']= 'Possessive wh-pronoun 所有格代词'
    dic['WRB']= 'Wh-adverb 疑问代词'
    return dic

def get_data(data_size=0):
    df = pd.read_table('traindata.txt', header=None, encoding='gb2312', sep='\n', index_col=None)
    df['word'] = df[0].apply(lambda x: str(x).split('/')[0].strip())
    df['tag'] = df[0].apply(lambda x: str(x).split('/')[1].strip())
    df.drop(columns=[0], inplace=True)
    if (data_size > 0):
        # 只取一小部分资料
        del_list = list(range(data_size, df.shape[0]))
        df.drop(index=del_list, inplace=True)

    return df


def get_transfer_mat(df):
    words_uniq = df['word'].unique()
    tags_uniq = df['tag'].unique()
    pi_series = pd.Series(data=0, index=tags_uniq)
    A_tag_tag_mat = pd.DataFrame(data=0, index=tags_uniq, columns=tags_uniq)  # 给定一个tag,出现另一个tag的几率,隐变量,状态转移矩阵
    B_tag_word_mat = pd.DataFrame(data=0, index=tags_uniq, columns=words_uniq)  # 给定一个tag,出现word的几率,观测变量转移矩阵

    words = df['word']
    tags = df['tag']
    pre_tag = None
    for i in range(df.shape[0]):
        print(i)
        word = words[i]
        tag = tags[i]
        B_tag_word_mat.loc[tag, word] += 1
        if (pre_tag is None):
            pi_series[tag] += 1
        else:
            A_tag_tag_mat.loc[pre_tag, tag] += 1
        if (word == '.'):
            pre_tag = None
        else:
            pre_tag = tag

    # 将 pi归一化
    pi_sum = pi_series.sum()
    pi_series = pi_series / pi_sum

    # 将 A B 归一化
    A_tag_tag_mat['sum'] = 0
    B_tag_word_mat['sum'] = 0
    for tag in tags_uniq:
        sum_of_tag=A_tag_tag_mat.loc[tag, :].sum()
        A_tag_tag_mat.loc[tag, 'sum'] = sum_of_tag
        B_tag_word_mat.loc[tag, 'sum'] = sum_of_tag
    for tag in tags_uniq:
        A_tag_tag_mat[tag] = A_tag_tag_mat[tag] / A_tag_tag_mat['sum']
    for word in words_uniq:
        B_tag_word_mat[word] = B_tag_word_mat[word] / B_tag_word_mat['sum']
    A_tag_tag_mat.drop(columns='sum', inplace=True)
    B_tag_word_mat.drop(columns='sum', inplace=True)
    print('finish get_transfer_mat')
    return pi_series, A_tag_tag_mat, B_tag_word_mat


def check_sen(words, B_tag_word_mat):
    dics = B_tag_word_mat.columns.tolist()

    for word in words:
        if (word not in dics):
            return word
    return ''


def sentence_to_words(sen):
    words = str(sen).split(" ")
    rtn_words = []
    for word in words:
        word = word.strip()
        if (len(word) > 0):
            if(len(word)>1 and word.endswith(',')):
                rtn_words.append(word[0:len(word)-1])
                rtn_words.append(',')
            else:
                rtn_words.append(word)
    return rtn_words


def log(value):
    if (value == 0):
        return np.log(0.000001)
    else:
        return np.log(value)


def viterbi(words, pi_series, A_tag_tag_mat, B_tag_word_mat):
    tags = pi_series.index

    T = list(range(len(words)))
    # data表示最大概率的前一个tag的名称和概率
    dp = pd.DataFrame(data=None, columns=tags, index=T)

    for tag in tags:
        prob = log(pi_series[tag]) + log(B_tag_word_mat.loc[tag, words[0]])
        dp.loc[0, tag] = ("na", prob)

    for i in range(1, dp.shape[0]):
        word = words[i]
        for tag in tags:
            max_prob_tag = None
            max_prob = -np.inf
            for pre_tag in tags:
                prob = dp.loc[i - 1, pre_tag][1] + log(A_tag_tag_mat.loc[pre_tag, tag]) +  log(B_tag_word_mat.loc[tag, word])
                if (prob > max_prob):
                    max_prob_tag = pre_tag
                    max_prob = prob
            dp.loc[i, tag] = (max_prob_tag, max_prob)

    print(dp.head())

    result = []

    # 先求出最后一个节点的结果
    max_prob_tag = None
    max_prob = -np.inf
    final_index = dp.shape[0] - 1
    pre_tag=None
    for tag in tags:
        pre_tag_tmp=dp.loc[final_index, tag][0]
        prob = float(dp.loc[final_index, tag][1])
        if (max_prob_tag is None or prob > max_prob):
            max_prob_tag = tag
            max_prob = prob
            pre_tag=pre_tag_tmp

    result.append(max_prob_tag)
    result.append(pre_tag)

    # 依序查出之前的
    na = False
    index = final_index - 1

    while (na == False):

        pre_tag = dp.loc[index, pre_tag][0]

        if (pre_tag == 'na'):
            na = True
        else:
            result.append(pre_tag)
            index -= 1
    result.reverse()
    return result

def print_verb_exp(words,result):
    dic=get_explain_dic()
    for i in range(len(words)):
        print("word:",words[i])
        print("result:", result[i])
        print("explain:", dic[result[i]])
        print("-----------------------------")


if __name__ == '__main__':

    # 取得训练集
    df = get_data(data_size=1000)
    # 训练 pi A B
    pi_series, A_tag_tag_mat, B_tag_word_mat = get_transfer_mat(df)
    # 测试句子
    sen = "Social Security number , passport number and details about the services provided for the payment"
    sen = "trying to keep pace with rival Time magazine"
    words = sentence_to_words(sen)
    #校验字典是否包含要测试的word
    result = check_sen(words, B_tag_word_mat)

    #维特比算法
    if (result == ''):
        print("viterbi start")
        result=viterbi(words, pi_series, A_tag_tag_mat, B_tag_word_mat)
        #打印注释
        print_verb_exp(words, result)
    else:
        print("字典里不包含{}".format(result))

