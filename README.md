# Classifying World Wonders using ANN Models (MLP, HF, DBN)

## Overview
This project aims to classify the wonders of the world images using different variations of the Multilayer Perceptron (MLP) network, Hopfield Network (HF), and Deep Belief Networks (DBN). 
The project is divided into several Jupyter notebooks, each handling a different part of the project.

## Workflow

### Pre-processing
In the `pre-processing.ipynb` notebook, we performed the necessary pre-processing steps on the images. Specifically:
* Defined utility functions for saving and loading data in pickle format, and for splitting data into training, validation, and test sets.
* Created An image data generator and an iterator to load images from a directory and convert them into a format suitable for machine learning models.
* Performed undersampling to balance the dataset to ensure that all classes in the dataset have an equal number of instances, preventing any class bias during model training.
* Used Principal Component Analysis (PCA) for feature extraction.
* Split the processed dataset into training, validation, and test sets using the utility function defined earlier.

### Chosen Models (MLP, HF, DBN)
In the `mlp`, `world wonders`, and `DBN` notebooks, we experiment with different variations of the MLP, HF, and DBN neural network models respectively. 
The aim is to evaluate and compare their performance in classifying the wonders of the world images, where the DBN model returned the "best" result.

Side note: For a detailed explanation of the DBN model, please refer to the following paper by Ashraf: [A Tutorial, Literature Review, and Comparative Analysis of Restricted Boltzmann Machines and Deep Belief Networks](https://www.researchgate.net/publication/366657577_A_Tutorial_Literature_Review_and_Comparative_Analysis_of_Restricted_Boltzmann_Machines_and_Deep_Belief_Networks)
