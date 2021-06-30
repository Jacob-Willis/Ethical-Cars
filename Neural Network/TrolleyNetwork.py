import numpy as np
from numpy.lib.function_base import append
import NeuralNetwork as NN
import TrolleyData as TD

# nnfs.init()

# # Create dataset
# X, y, X_test, y_test = NN.create_data_mnist('fashion_mnist_images')

# # Shuffle the training dataset
# keys = np.array(range(X.shape[0]))
# np.random.shuffle(keys)
# X = X[keys]
# y = y[keys]

# # Scale and reshape samples
# X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
#              127.5) / 127.5

# data = np.array(data, dtype=np.float32)

X_Data = np.array(TD.X_Data, dtype=np.float32)
y_Data = np.array(TD.y_data, dtype=np.float32)

X = np.array(TD.X_Data, dtype=np.float32)
X_test = np.array(TD.X_Data, dtype=np.float32)
y = np.array(TD.y_data, dtype=np.float32)
y_test = np.array(TD.y_data, dtype=np.float32)

# Instantiate the model
model = NN.Model()


# Add layers
model.add(NN.Layer_Dense(4, 16, weight_regularizer_l1=5e-4, weight_regularizer_l2=5e-4,
                 bias_regularizer_l1=5e-4, bias_regularizer_l2=5e-4))
model.add(NN.Activation_ReLU())
model.add(NN.Layer_Dense(16, 1, weight_regularizer_l1=5e-4, weight_regularizer_l2=5e-4,
                 bias_regularizer_l1=5e-4, bias_regularizer_l2=5e-4))
model.add(NN.Activation_Sigmoid())

# Set loss, optimiser and accuracy objects
model.set(
    loss=NN.Loss_BinaryCrossentropy(),
    optimiser=NN.optimiser_Adam(decay=5e-5),
    accuracy=NN.Accuracy_Categorical(binary=True)
)

# finalise the model
model.finalise()

X_Split_Data = np.split(X_Data, 8)
y_Split_Data = np.split(y_Data, 8)

for i in range(7):
  if (i < 6):
    # Train the model
    X = X_Split_Data[i]
    X_test = X_Split_Data[i+1]
    y = y_Split_Data[i]
    y_test = y_Split_Data[i+1]

    model.train(X, y, validation_data=(X_test, y_test),
                epochs=750, batch_size=16, print_every=96)
  else:
    # Train the model
    X = X_Split_Data[6]
    X_test = X_Split_Data[0]
    y = y_Split_Data[6]
    y_test = y_Split_Data[0]

    model.train(X, y, validation_data=(X_test, y_test),
                epochs=750, batch_size=16, print_every=96)
      

X_test = X_Split_Data[7]
y_test = y_Split_Data[7]

# Retreive and print parameters
parameters = model.get_parameters()
model.save_parameters('trolley_network.parms')
model.save('trolley_network.model')

print("Eval test data")
model.evaluate(X_test, y_test)

# Predict on the first 5 samples from validation dataset
# and print the result
confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)

print(predictions)

# print first 5 labels
print(y_test[:5])

# for prediction in predictions:
#     print(NN.trolley_problem_choice[prediction[0]])



# NEW MODEL WITH WEIGHTS AND BIASES FROM PREVIOUS OLD MODEL

# Instantiate the model
# modelNew = NN.Model()

# # Add layers
# modelNew.add(NN.Layer_Dense(X.shape[1], 128))
# modelNew.add(NN.Activation_ReLU())
# modelNew.add(NN.Layer_Dense(128,128))
# modelNew.add(NN.Activation_ReLU())
# modelNew.add(NN.Layer_Dense(128,10))
# modelNew.add(NN.Activation_Softmax())

# # Set loss and accuracy objects
# # We do not set optimiser object this time - there's no need to do it
# # as we won't train the model
# modelNew.set(
#     loss=NN.Loss_CategoricalCrossentropy(),
#     accuracy=NN.Accuracy_Categorical()
# )

# # Finalise the model
# modelNew.finalise()

# # Set model with parameters instead of training it
# # modelNew.set_parameters(parameters)
# modelNew.load_parameters('fashion_mnist.parms')

# # Evaluate the model
# print("Eval of the new model")
# modelNew.evaluate(X_test, y_test)
# modelNew.save('NEW_fashion_mnist.model')

# print("Now evaluating the loaded model!")

# # Load the model
# modelLoadNew = NN.Model.load('fashion_mnist.model')

# # Evaluate the model
# modelLoadNew.evaluate(X_test, y_test)

# # Predict on the first 5 samples from validation dataset
# # and print the result
# confidences = modelLoadNew.predict(X_test[:5])
# predictions = model.output_layer_activation.predictions(confidences)
# print(predictions)

# # print first 5 labels
# print(y_test[:5])

# for prediction in predictions:
#     print(NN.fashion_mnist_labels[prediction])
# %%
