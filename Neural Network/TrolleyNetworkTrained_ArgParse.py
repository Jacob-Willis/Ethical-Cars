import numpy as np
import argparse
from gooey import Gooey
import NeuralNetwork as NN
import TrolleyData as TD

@Gooey
def main():
    parser = argparse.ArgumentParser(description="to determine if the AI will continue straight or change direction")
    parser.add_argument("Straight_count", type=int, help="Average age of people going straight")
    parser.add_argument("Straight_average", type=int, help="Number of people going straight")
    parser.add_argument("Other_count", type=int, help="Average age of people going in other direction")
    parser.add_argument("Other_average", type=int, help="Number of people going in other direction")
    parser.add_argument("-v", "--verbose", action="count", help="increase output verbosity")
    args = parser.parse_args()

    if args.verbose >= 2:
        print("Running '{}'".format(__file__))
    if args.verbose >= 1:
        print("Straight count - {}, Straight average - {}".format(args.Straight_count, args.Straight_average))
        print("Other count - {}, Other average - {}".format(args.Other_count, args.Other_average))


    X_Data = np.array(TD.X_Data, dtype=np.float32)
    y_Data = np.array(TD.y_data, dtype=np.float32)

    X_test = X_Data
    y_test = y_Data

    # Load the model
    model = NN.Model.load('trolley_network.model')

    # Evaluate the model
    # model.evaluate(X_test, y_test)

    # Predict on the first 5 samples from validation dataset
    # and print the result
    # confidences = model.predict(X_test[:5])
    # predictions = model.output_layer_activation.predictions(confidences)
    # print(predictions)

    print("\nLet's test your input!\n")

    # choice: [straight: avg age, straight: count, change: avg age, change: count]
    # your choice: [0] 0 = straight, 1 = change direction

    x_input = np.array([[args.Straight_average, args.Straight_count, args.Other_average, args.Other_count]])

    confidences = model.predict(x_input)
    predictions = model.output_layer_activation.predictions(confidences)
    # print(predictions)

    for prediction in predictions:
        print("Avg age going straight: " + str(x_input[0][0]) + ", Number of people going straight: " + str(x_input[0][1]))
        print("Avg age in other direction: " + str(x_input[0][2]) + ", Number of people in other direction: " + str(x_input[0][3]))
        # print("Avg age going straight: " + str(x_input[prediction[0]][0]) + ", Number of people going straight: " + str(x_input[prediction[0]][1]))
        # print("Avg age in other direction: " + str(x_input[prediction[0]][2]) + ", Number of people in other direction: " + str(x_input[prediction[0]][3]))
        print("\nI think the car should: " + NN.trolley_problem_choice[prediction[0]].lower())

main()