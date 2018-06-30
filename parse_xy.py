import numpy as np
import sys, re

def main():
    targetfile = sys.argv[1]
    outputfile1 = sys.argv[2]
    outputfile2 = sys.argv[3]

    output1 = open(outputfile1, 'w', encoding='utf-8')
    output2 = open(outputfile2, 'w', encoding='utf-8')

    pre_line = ''
    with open(targetfile, 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            if pre_line != '':
                output2.write(line)
                output1.write(pre_line)
            pre_line = line
        print ('\rSplitting X & Y %d' % (texts_num+1), end='', flush=True)
    print('')





if __name__=='__main__':
    main()
