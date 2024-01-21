# Generating Facial Expressions with Bidirectional Conditional GAN
Applying Bidirectional Conditional GAN model on the FER2013 dataset which consists of 35340 examples, 48x48 pixel grayscale images of faces, classified to six emotions:  
Emotion 0 - Angry: 4953  
Emotion 1 - Neutral: 6198  
Emotion 2 - Fear: 5121  
Emotion 3 - Happy: 8989  
Emotion 4 - Sad: 6077  
Emotion 5 - Surprised: 4002

## People
[Dalia Sherman](https://github.com/Dalia-Sher)  
[Shir Mamia](https://github.com/ShirMamia)

## Data
[FER2013](https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge)

## Table of contents 
* [Final Model - BiCoGAN](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-GAN/blob/main/12.Final_Model_BiCoGAN.ipynb)


## Report
[Generating Realistic Facial Expressions with BiCoGAN.pdf](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-GAN/blob/main/Generating%20Realistic%20Facial%20Expressions%20with%20BiCoGAN.pdf)


## Papers
* [Generating Realistic Facial Expressions through Conditional Cycle-Consistent Generative Adversarial Networks (CCycleGAN)](https://openreview.net/pdf?id=HJg6j3-oeB)
* [Double Encoder Conditional GAN for Facial Expression Synthesis](https://ieeexplore.ieee.org/document/8483579)
* [ADVERSARIAL FEATURE LEARNING](https://openreview.net/pdf?id=BJtNZAFgg)
* [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
* [Bidirectional Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1711.07461.pdf)


## References
* Reading the data: 
  * https://www.kaggle.com/abhijeetambekar/deep-learning-face-emotion
* GAN: 
  * https://www.tensorflow.org/tutorials/generative/dcgan 
* Conditional GAN: 
  * https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
  * https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
* Bidirectional GAN: 
  * https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
* Wasserstein GAN: 
  * https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/  
  * https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
* Wasserstein GAN GP: 
  * https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
* Data Augmentation:
  * https://towardsdatascience.com/data-augmentation-techniques-in-python-f216ef5eed69
