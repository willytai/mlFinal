# -*- coding: utf-8 -*-
# use the Manderine Chinese version jieba

import jieba
import sys

if len(sys.argv) != 3:
    print ('Usage: python3 segmentation.py <targetfile> <outputfile>')
    sys.exit()

def main():

    targetfile = sys.argv[1]
    outputfile = sys.argv[2]

    # jieba custom setting.
    # load user defined dictionary
    jieba.load_userdict('jieba_dict/user_dict.txt')

    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open(outputfile, 'w', encoding='utf-8')
    print ('')
    with open(targetfile, 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            #output.write('卍 ')
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            #output.write('乂')
            output.write('\n')

            print ('\rSplitting line %d' % (texts_num+1), end='', flush=True)

    output.close()
    print ('')

if __name__ == '__main__':
    main()
