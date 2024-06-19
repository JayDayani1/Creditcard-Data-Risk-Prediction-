# Creditcard-Data-Risk-Prediction-
This repository contains the Jupyter Notebook for predicting the risk of credit card transactions. The project demonstrates the use of machine learning techniques to identify potentially fraudulent transactions based on transaction data.

## Project Description

### Objective

The objective of this project is to build and evaluate a machine learning model to predict the risk of credit card transactions. The key tasks include:

1. **Data Preprocessing**: Cleaning and preparing the data for analysis.
2. **Feature Engineering**: Creating new features and selecting relevant ones for the model.
3. **Model Building**: Constructing and training a machine learning model.
4. **Evaluation**: Assessing the performance of the model using appropriate metrics.

## File Description

- **Creditcard Data Risk Prediction.ipynb**: This is the main Jupyter Notebook file containing all the code, visualizations, and explanations for the project tasks.

## Requirements

To run the notebook, you will need the following Python packages:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

**Overall Observations:**
The dataset was preprocessed by standardizing the features using StandardScaler and handling class imbalance with RandomUnderSampler. It was then split into training and validation sets. A neural network model with three linear layers and appropriate activation functions was defined in PyTorch. The model was trained using the Adam optimizer and binary cross-entropy loss function. Analysis of the training process revealed that for EPOCH 200, the model gradually improved on the training data, but the validation loss plateaued, indicating potential overfitting. The training accuracy reached a high value of 0.9910, suggesting overfitting, while the validation accuracy plateaued around 0.9394. For EPOCH 100, the training loss was higher, indicating incomplete convergence, but the validation loss was lower, indicating better generalization. The training accuracy for EPOCH 100 was 0.9605, while the validation accuracy was 0.9192, slightly lower than EPOCH 200.

**Reasoning:**
At 100 epochs, the increase in training loss and the decrease in training accuracy suggest that the model is still learning and adapting well to the training data without overfitting. This indicates that the model is effectively capturing the patterns and relationships present in the training set.

By examining the validation loss and accuracy, we can assess the model's ability to generalize to unseen data. Overfitting occurs when a model becomes too specialized in the training data, resulting in poor performance on new or unseen data. In the case of 200 epochs, the plateauing of the validation loss and the slight decrease in validation accuracy suggest that the model might be overfitting. The performance gap between training and validation metrics indicates that the model is not generalizing well to unseen data.

Comparing the results between 200 and 100 epochs, it becomes apparent that 100 epochs strike a better balance between training and validation performance. Although the training loss is slightly higher at 100 epochs, it is likely that the model has not yet converged fully, and further training may yield better results. The lower validation loss at 100 epochs indicates improved generalization performance compared to 200 epochs. Additionally, the training accuracy for 100 epochs is lower but still reasonably high, indicating that the model is capturing the essential patterns in the training data while avoiding overfitting. The validation accuracy at 100 epochs, although slightly lower than that of 200 epochs, suggests that the model is performing well on unseen data and has a better generalization ability.

**Conclusion:**
After evaluating the output, it can be inferred that training the model for 100 epochs strikes a better balance between training and validation performance compared to training for 200 epochs. The results at 100 epochs demonstrate lower validation loss and comparable validation accuracy, indicating improved generalization capability. Therefore, it can be estimated that the model achieves optimal performance at approximately 100 epochs.
