# NBED
Code of paper "A new baseline for edge detection: Make Encoder-Decoder great again"
## Preparing the dataset
Download the dataset to any dir and point to the dir in the code  
-BSDS500 following the setting of "The Treasure Beneath Multiple Annotations: An Uncertainty-aware Edge Detector"  
-NYUDv2 following the setting of "Pixel Difference Networks for Efficient Edge Detection"  and random crop to 400*400
-BIPED following the setting of "Dense Extreme Inception Network for Edge Detection"  
## Preparing the pretrained weights
Down it from https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384_in21ft1k.pth
and put it into the dir ./model
## Training NBED
'''
python main.py --batch_size 4 --stepsize 3-4 --gpu 1 --savedir 0305-bsds --encoder Dul-M36 --decoder unetp --head default --note 'training on BSDS500' --dataset BSDS --maxepoch 6
'''
## Eval NBED
Following the previous methods. such as RCF and PiDiNet
