# Image Classification with Convolutional Neural Networks

## Overview
This project focuses on classifying microorganisms using deep learning models. The goal is to train and evaluate two different models: one without residual connections and one with residual connections. The project is implemented in a Jupyter Notebook environment.

## Organization 
- The 'cv3.ipynb' Jupyter Notebook file contains the code for training and evaluating the models, as well as plotting the results.File contains the definitions of the two models: 'Without_Residual' and 'WithResidual'and includes utility functions for training, evaluation, and plotting.
## Running the Code
1. Open Jupyter Notebook: `jupyter notebook`
2. Open the 'cv3.ipynb' file.
3. Run the code cells in the notebook to execute the project.

## Functions and Usage
- `train(model, loader, criterion, optimizer)`: Trains the model using the given data loader, criterion, and optimizer.
- `evaluate(model, loader, criterion)`: Evaluates the model using the given data loader and criterion.
- `confusion_mat(model, loader)`: Computes the confusion matrix for the model's predictions using the given data loader.
- `plot_metrics(metric, learning_rate, batch_size, num_epochs, ylabel, xlabel, title)`: Plots the specified metric (e.g., loss or accuracy) over epochs for different learning rates and batch sizes.

### Classes
- `Without_Residual`: This class defines the model without residual connections. It consists of several convolutional layers followed by fully connected layers.
- `WithResidual`: This class defines the model with residual connections. It includes residual connections between convolutional layers to improve feature propagation.
- `DataPreparation`: This class handles the preparation of the microorganism dataset by splitting it into training, validation, and test sets.
- `CustomDataset`: This class is a custom implementation of the `torch.utils.data.Dataset` class. It loads the microorganism dataset, applies data transformations, and provides access to the images and their corresponding labels.

## Results
- The code generates graphs showing the loss and accuracy change over epochs for different learning rates and batch sizes.
- The best model is selected based on validation accuracy and evaluated on the test set.
- Dropout is integrated into the best models to prevent overfitting.
- The confusion matrix is plotted to analyze the model's predictions.
