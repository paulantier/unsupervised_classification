# unsupervised_classification
Trying to get a good unsupervised image classification method with clustering from different types of autoencoder outputs (Vision transformer or simple CNN autoencoders). I also tried using PCA projection to visualize the class clustering but it's not ideal.

First try of clustering on a simple autoencoder on CIFAR10 wasnt very efficient so i want to try with a masked autoencoder to see if it gets better at extracting class-relevant elements from images.

Inspired by Masked Autoencoders Are Scalable Vision Learners
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick

(and i realized later than this idea is quite similar to this paper : Denoising Masked AutoEncoders Help Robust Classification
Quanlin Wu, Hang Ye, Yuntian Gu, Huishuai Zhang, Liwei Wang, Di He)

Add contrastive loss after encoding, idea from "ViT-AE++: Improving Vision Transformer Autoencoder for Self-supervised Medical Image Representations
Chinmay Prabhakar, Hongwei Bran Li, Jiancheng Yang, Suprosana Shit, Benedikt Wiestler, Bjoern Menze"
