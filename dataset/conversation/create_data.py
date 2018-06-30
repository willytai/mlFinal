import re, sys
encoder_in = open('encoder.input', 'w+')
decoder_in = open('decoder.input', 'w+')
decoder_out = open('decoder.output', 'w+')
maxlen = 0

with open('conv.txt', 'r') as f:
    for line in f:
        line = re.sub("\n", "", line)
        line = line.split(' ++++ ')
        seq1 = line[0].split()
        seq2 = line[1].split()
        if maxlen < len(seq1):
            maxlen = len(seq1)
        if maxlen < len(seq2) + 1:
            maxlen = len(seq2) + 1
        encoder_in.write('{}\n'.format(" ".join(seq1)))
        decoder_in.write('<s> {}\n'.format(" ".join(seq2)))
        decoder_out.write('{} </s>\n'.format(" ".join(seq2)))

encoder_in.close()
decoder_in.close()
decoder_out.close()



print ('max length', maxlen)
