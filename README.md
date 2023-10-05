# README for ProblemSet1.ipynb
https://colab.research.google.com/github/CourtneyBrookes/Problem1/blob/main/ProblemSet1.ipynb 
This Jupyter Notebook, titled `ProblemSet1.ipynb`, was generated using Colaboratory and contains Python code for working with the MNIST dataset and training a random walk model. Below, we provide an overview of the code, its purpose, and how to use it.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Loading MNIST Dataset](#loading-mnist-dataset)
- [Random Walk Model](#random-walk-model)
- [Training the Model](#training-the-model)
- [Conclusion](#conclusion)

## Introduction<a name="introduction"></a>
This notebook demonstrates the following tasks:
1. Loading the MNIST dataset and displaying a montage of sample images.
2. Running a random walk model on the MNIST dataset.
3. Training the random walk model to achieve at least 75% accuracy.

## Dependencies<a name="dependencies"></a>
Before running the code in this notebook, ensure that you have the following dependencies installed:
- Python
- NumPy
- Matplotlib
- PyTorch
- scikit-image (skimage)
- wandb (Weights and Biases, used for tracking experiments)

You can install the required packages using the following command:
```python
pip install numpy matplotlib torch scikit-image wandb
```

## Loading MNIST Dataset<a name="loading-mnist-dataset"></a>
The code begins by loading the MNIST dataset using the PyTorch `datasets` module. You can uncomment and modify the code to load other datasets such as KMNIST or Fashion MNIST if needed. The dataset is loaded and preprocessed, and a montage of sample images is displayed using the `montage_plot` function.

## Random Walk Model<a name="random-walk-model"></a>
The code then demonstrates a simple random walk model. It reshapes a single image from the dataset and performs matrix multiplication with a randomly generated weight matrix. The result is stored in the variable `y`. The model's predictions are also displayed.

## Training the Model<a name="training-the-model"></a>
Next, the notebook defines a function `GPU` to move data to a GPU (if available) and a function `GPU_data` to create GPU tensors without requiring gradients. The random walk model is trained using a stochastic gradient descent (SGD) approach. The code iterates through multiple steps, updating the model weights and calculating accuracy. The best-performing model and its corresponding accuracy are printed.

## Conclusion<a name="conclusion"></a>
This notebook provides a hands-on introduction to loading and working with image data, running a simple random walk model, and training the model to achieve a specific accuracy threshold. You can modify and extend this code as needed for your own experiments or analysis.
