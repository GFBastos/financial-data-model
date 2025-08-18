Fraud Detection with Neural Networks

  This project implements a fraud detection system using the Kaggle Transactions Fraud Dataset. The goal is to classify whether a transaction is fraudulent or not by training a deep neural network in TensorFlow/Keras.
  Dataset
  The dataset contains transactional and user information related to fraud detection. Before training, the data was cleaned and preprocessed. Preprocessing steps included handling class imbalance with class_weight, encoding categorical variables, and scaling numerical features.

Model

  The core model is defined in the build_model() function. It is a configurable feed-forward neural network with the following properties:
  
  Between 3 and 6 hidden layers
  
  32 to 256 neurons per layer
  
  Batch Normalization and ReLU activation after each hidden layer
  
  He normal initialization
  
  L2 regularization (except when using AdamW optimizer)
  
  Sigmoid activation in the output layer for binary classification
  
  Hyperparameters such as the number of layers, neurons per layer, learning rate, and optimizer were tuned using Keras Tuner. Supported optimizers included SGD, AdamW, Momentum, Nesterov, and RMSProp.
  
  The model was compiled with binary crossentropy loss and evaluated with accuracy, precision, recall, and AUC (PR and ROC) metrics.

Results

The best configuration found by the tuner used four hidden layers, thirty-two neurons per layer, a learning rate of about 0.009, and the Nesterov optimizer. The trained model was saved as best_model.keras.

On the validation set, this model achieved:

AUC-PR: 0.687

AUC-ROC: 0.933

Accuracy: ~87–88%

Precision: ~0.41

Recall: ~0.87–0.91

The model prioritizes recall, catching most fraudulent cases at the expense of more false positives. This is often desirable in fraud detection scenarios, where missing fraudulent activity is riskier than flagging extra cases.

Future Work

Potential improvements include:

Threshold tuning to balance precision and recall

Experimenting with focal loss or other imbalance-aware objectives

Feature engineering with transaction and user-level attributes
