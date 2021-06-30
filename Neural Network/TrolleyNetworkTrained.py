import numpy as np
from gooey import Gooey
import NeuralNetwork as NN
import TrolleyData as TD

@Gooey
def main():
    X_Data = np.array(TD.X_Data, dtype=np.float32)
    y_Data = np.array(TD.y_data, dtype=np.float32)

    X_test = X_Data
    y_test = y_Data

    # Load the model
    model = NN.Model.load('trolley_network.model')

    # Evaluate the model
    model.evaluate(X_test, y_test)

    # Predict on the first 5 samples from validation dataset
    # and print the result
    confidences = model.predict(X_test[:5])
    predictions = model.output_layer_activation.predictions(confidences)
    # print(predictions)

    # print first 5 labels
    # print(y_test[:5])

    print("")
    print("Let's test your input!\n")

    # choice: [straight: avg age, straight: count, change: avg age, change: count]
    # your choice: [0] 0 = straight, 1 = change direction

    straightCount = input("How many people going straight ahead: ")
    straightAve = 0
    print("What are their ages?")
    for i in range(0, int(straightCount)):
        # temp = (input(str(i) + ": "))
        straightAve += int(input(str(i) + ": "))
        

    straightAve /= int(straightCount)

    otherCount = input("How many people in the other direction: ")
    otherAve = 0
    print("What are their ages?")
    for i in range(0, int(otherCount)):
        otherAve += int(input(str(i) + ": "))

    otherAve /= int(otherCount)

    x_input = np.array([[straightAve, straightCount, otherAve, otherCount]])

    # x_input = np.array([
    #       [70, 1, 5, 10],
    #       [5, 10, 70, 1],
    #       [22, 1, 70, 1],
    #       [36, 5, 22, 1],
    #       [ 3, 3, 22, 1],
    #       [22, 1,  3, 3],
    #       [ 1, 1,  1, 1],
    #       [10,100, 1, 1],
    #       [100,1,102, 4],
    #       [1  ,1,2  , 2],
    #       [2  ,2,1  , 1],
    #       [80 ,1,6  , 1]
    #                    ])

    confidences = model.predict(x_input)
    predictions = model.output_layer_activation.predictions(confidences)
    # print(predictions)

    print("\n")
    for prediction in predictions:
        print("Avg age going straight: " + str(x_input[0][0]) + ", Number of people going straight: " + str(x_input[0][1]))
        print("Avg age in other direction: " + str(x_input[0][2]) + ", Number of people in other direction: " + str(x_input[0][3]))
        # print("Avg age going straight: " + str(x_input[prediction[0]][0]) + ", Number of people going straight: " + str(x_input[prediction[0]][1]))
        # print("Avg age in other direction: " + str(x_input[prediction[0]][2]) + ", Number of people in other direction: " + str(x_input[prediction[0]][3]))
        print("I think the car should: " + NN.trolley_problem_choice[prediction[0]].lower())

main()