# Autoencoder
# 1
Design an autoencoder using the dataset train_1.csv.
Create separate encoder and decoder models and connect them into a single model.
Learn autoencoder and loss function plotting from the learning history provided by the fit function.
Generate latent features as an output from the encoder.
Generate output images based on latent features.

# 2 
Load the MNIST image dataset (with the help of Keras functions).
Reduce the training dataset to 50 images per category and the test dataset to 10 images per category.
Shuffle both datasets. Normalize input data to the range (<0,1).
Create the autoencoder model for the reduced dataset with at least two hidden layers in both the encoder and decoder. At least one layer in the encoder will be a Conv2D layer with 30 filters. The dimension of latent space will be 4. Ensure that the output values of the model will be in the range <0,1).
Train the model with Adam optimizer with learning rate = 0.004 and mse as loss and matrics function. Use 30 epochs, and batch size = 32. Use the test set as validation data during training.
Split the trained model into an encoder and decoder.
Use the decoder model to generate four new images based on random values in latent space.
Display the generated images.
