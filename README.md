# Unsupervisedly Learning Sentence Representations

The project aims to explore better ways to learn phrase and sentence representations.
In Machine Vision, people have settled on an underlying neural network structure to “read” images and frequently use pre-training this “reading” layer, the same is not true for NLP, where we only really use word representations.

Particular questions this research is going to address include:
- Can we apply word representation methods to phrases and sentences?
- Can we do so in a computationally efficient manner?
- Will linguistic patterns arise?  How do we observe them?

## Skipthought vectors

As a starting point, this repository contains a Tensorflow 1.0 implementation of the Skipthought paper ([Kiros et al.](https://chara.cs.illinois.edu/sites/fa16-cs591txt/pdf/Kiros-2015-NIPS.pdf))

The structure of the model is as follows:
- An encoder is used to find a vector representation of a sentence;
- Then decoders are used to predict the preceding and the following sentence
- Training this model also yields word representations

![Figure taken from Kiros et al. \label{kiros}](https://cdn-images-1.medium.com/max/1000/1*MQXaRQ3BsTHpn0cfOXcbag.png)

The Skipthought vector model finds good sentence representations, such that similar sentences are close to each other in vector space. However by relying on two decoders, the method is computationally expensive. If the goal is just to find sentence embeddings, then other methods such as sequential autoencoders might be more efficient alternatives. My project is going to explore these alternatives.
