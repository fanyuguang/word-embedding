#!/bin/sh

make

time ./word2vec -train train_data.txt -output train_data_embedding.txt -cbow 1 -size 300 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15 -min-count 1
# ./distance train_data_embedding.txt
