#!/bin/bash

(
mkdir -p ckpt
cd ckpt/
gdown https://drive.google.com/uc?id=1onWVW-He48yEOXTWokxKMeG9QrJYSLl6
cd ..
mkdir -p data
cd data/

gdown https://drive.google.com/uc?id=1LZzzJ53yEUAMZ_rmW-byY9-w6D-A46oq
unzip csv.zip
rm csv.zip

gdown https://drive.google.com/uc?id=1pfxuhMuXv5KIgW2blSG5kUNwyne5DtmW
unzip image.zip
rm image.zip

mkdir -p test_mask_sample
cd test_mask_sample/
gdown https://drive.google.com/uc?id=1GL2gBE69DYVp118vJLKfWUgWh4KNaYd9
unzip test_mask_sample.zip
rm test_mask_sample.zip
cd ..

mkdir -p test_image_sample
cd test_image_sample/
gdown https://drive.google.com/uc?id=1w35ilYMA5KBD9KLbFlY_ORo7x2fdXyGx
unzip test_image_sample.zip
rm test_image_sample.zip
)
