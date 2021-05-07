# Generating Facial Expressions through Bidirectional Conditional GAN
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
* [1. GAN Version 1](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/01.GAN_Version_1.ipynb)  
* [2. GAN Version 2](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/02.GAN_Version_2.ipynb)  
* [3. Conditional GAN Version 1](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/03.Conditional_GAN_Version_1.ipynb)  
* [4. Conditional GAN Version 2](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/04.Conditional_GAN_Version_2.ipynb)  
* [5. Bidirectional GAN](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/05.Bidirectional_GAN.ipynb)  
* [6. Bidirectional Conditional GAN](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/06.Bidirectional_Conditional_GAN.ipynb)  
* [7. Wasserstein GAN Version 1](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/07.Wasserstein_GAN_Version_1.ipynb)  
* [8. Wasserstein GAN Version 2](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/08.Wasserstein_GAN_Version_2.ipynb)  
* [9. Wasserstein GAN GP](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/09.Wasserstein_GAN_GP.ipynb)  
* [10. Bidirectional Conditional WGAN](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-WGAN/blob/main/10.Bidirectional_Conditional_WGAN.ipynb)  
* [11. Data Augmentation](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-GAN/blob/main/11.Data_Augmentation.ipynb)  
* [12. Final Model - BiCoGAN](https://github.com/Dalia-Sher/Generating-Facial-Expressions-Bidirectional-Conditional-GAN/blob/main/12.Final_Model_BiCoGAN.ipynb)

## Report

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
