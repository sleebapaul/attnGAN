#!/bin/bash 

# Install the Python dependencies

pip2 install -r requirements.txt

# Remove all files from Encoder folder and download the pre-trained models.

rm DAMSMencoders/coco/*
wget https://www.dropbox.com/s/ijjjtlicanhyfps/image_encoder100.pth -P DAMSMencoders/coco/
wget https://www.dropbox.com/s/1eb994eacv4eeul/text_encoder100.pth -P DAMSMencoders/coco/
ls -l --block-size=M DAMSMencoders/coco

# Remove all files from models folder and download the pre-trained models.

rm models/*
wget https://www.dropbox.com/s/16631o3z4lc50xy/coco_AttnGAN2.pth -P models/
ls -l --block-size=M models/

# Remove all files from data folder and download the required data
rm -rf data/*
wget https://www.dropbox.com/s/pfb3ax1w8yfn40x/coco.zip -P data/
unzip data/coco.zip -d data/
unzip data/coco/val2014-text.zip -d data/coco/
mv data/coco/val2014 data/coco/text
ls -l --block-size=M data/coco/

# Now you're good to go