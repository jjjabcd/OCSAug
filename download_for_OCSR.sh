#!/bin/bash

(
mkdir -p data
cd data_for_ocsr/

gdown https://drive.google.com/uc?id=1LZzzJ53yEUAMZ_rmW-byY9-w6D-A46oq
unzip csv.zip
rm csv.zip

gdown https://drive.google.com/uc?id=1pfxuhMuXv5KIgW2blSG5kUNwyne5DtmW
unzip image.zip
rm image.zip

gdown https://drive.google.com/uc?id=1Aetloltpf9FnXzYWt927RcQ7i5MOEdc5
unzip "real-world hand-drawn images.zip"
rm "real-world hand-drawn images.zip"
)


