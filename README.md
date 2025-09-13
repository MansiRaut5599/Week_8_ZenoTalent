# Week_8_ZenoTalent
It's a Deep Learning Assignment which includes 3 questions:
Question 1: Feedforward Neural Network (NN)
Task: Implement a feedforward neural network for handwritten digit classification.
● Dataset: MNIST (download from tensorflow.keras.datasets or Kaggle)
● Instructions:
Load and preprocess the dataset (normalize pixel values).
Build a neural network with:
Input layer
2 hidden layers (ReLU activation)
Output layer (Softmax for 10 classes)
Compile the model with appropriate loss (categorical_crossentropy) and optimizer (adam).
Train the model and evaluate accuracy.
Plot training vs validation loss and accuracy
Question 2: Convolutional Neural Network (CNN)
Task: Implement a CNN for image classification.
● Dataset: CIFAR-10 or MNIST
● Instructions:
Preprocess the dataset (normalize and reshape as needed).
Build a CNN with:
Convolutional layers + ReLU
MaxPooling layers
Dropout layers
Fully connected (Dense) layers with Softmax output
Compile, train, and evaluate the model.
Visualize feature maps from at least one convolutional layer.
Question 3: Recurrent Neural Network (RNN/LSTM)
Task: Implement an RNN/LSTM model for text classification (sentiment analysis).
● Dataset: IMDB Movie Reviews (tensorflow.keras.datasets) - available in kaggle
● Instructions:
● Tokenize and pad sequences.
● Build an RNN or LSTM model with:
● Embedding layer
● LSTM layer
● Dense output layer (Sigmoid for binary classification)
● Compile, train, and evaluate accuracy.
● Plot training vs validation loss and accuracy.

