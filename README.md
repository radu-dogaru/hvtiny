# hvtiny
**A MCU-oriented TinyML CNN model**

A versatile CNN model giving acceptable accuracies while fiting inside MCU-oriented platforms. 
Code in support for paper:

R. Dogaru, I. Dogaru, R.C. Tecaru, "A Comparison of Several MCU-Oriented TinyML Models for Skin Lesions Classification", accepted to EHB2025 conference, Iasi - Romania, Nov. 2025 https://www.ehbconference.ro/

In the above paper the model was succesfully aplied to clasify skin-lesions while fiting into ESP-32 CAM platforms. Herein one can run it for datasets such as CIFAR10 obtaining around 80% validation accuracy with MAC (multiply and accumulates) less than 6000k 

By tuning the profile list and the k multiplier one can optimize the model in order to achieve best accuracy for a given size (as available from a MCU platform) 

Run directly in Google Colab: 
<a href="https://colab.research.google.com/github/radu-dogaru/hvtiny/blob/main/HVTINY_and_examples.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
