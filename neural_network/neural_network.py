'''
Neural Network Machine Learning 
Implemented for Function Prediction
'''
import numpy as np
import matplotlib
import sys

matplotlib.use("agg")
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, differential_evolution
from matplotlib import pyplot as plt
from itertools import product


def target_function(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


class TrainData:
    def __init__(self, sample_size, split_ratio, random_state=5) -> None:
        np.random.seed(random_state)
        X_train = np.random.uniform(-2 * np.pi, 2 * np.pi, sample_size)
        y_train = target_function(X_train)

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=split_ratio, random_state=random_state
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class MLP:
    def __init__(self, hidden_sizes, output_size=1, input_size=1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.weights_input_hidden = np.random.rand(input_size, hidden_sizes[0])
        self.bias_hidden = np.random.rand(hidden_sizes[0])

        self.weights_hidden_hidden = [
            np.random.rand(hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(len(hidden_sizes) - 1)
        ]
        self.biases_hidden = [np.random.rand(size) for size in hidden_sizes[1:]]

        self.weights_hidden_output = np.random.rand(hidden_sizes[-1], output_size)
        self.bias_output = np.random.rand(output_size)

    def tanh_activation(self, x):
        return np.tanh(x)

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))
    

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.tanh_activation(hidden_input)

        for weights, bias in zip(self.weights_hidden_hidden, self.biases_hidden):
            hidden_input = np.dot(hidden_output, weights) + bias
            hidden_output = self.tanh_activation(hidden_input)

        output = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output

        return output

    def get_params(self):
        all_params = [self.weights_input_hidden.flatten(), self.bias_hidden.flatten()]

        for weights, biases in zip(self.weights_hidden_hidden, self.biases_hidden):
            all_params.append(weights.flatten())
            all_params.append(biases.flatten())

        all_params.append(self.weights_hidden_output.flatten())
        all_params.append(self.bias_output.flatten())

        return np.concatenate(all_params)

    def set_params(self, params):
        input_hidden_size = self.weights_input_hidden.size
        hidden_hidden_sizes = [weights.size for weights in self.weights_hidden_hidden]
        hidden_output_size = self.weights_hidden_output.size

        current_index = 0

        self.weights_input_hidden = params[
            current_index : current_index + input_hidden_size
        ].reshape(self.weights_input_hidden.shape)
        current_index += input_hidden_size
        self.bias_hidden = params[
            current_index : current_index + self.bias_hidden.size
        ].reshape(self.bias_hidden.shape)
        current_index += self.bias_hidden.size

        for i in range(len(hidden_hidden_sizes)):
            self.weights_hidden_hidden[i] = params[
                current_index : current_index + hidden_hidden_sizes[i]
            ].reshape(self.weights_hidden_hidden[i].shape)
            current_index += hidden_hidden_sizes[i]
            self.biases_hidden[i] = params[
                current_index : current_index + self.biases_hidden[i].size
            ].reshape(self.biases_hidden[i].shape)
            current_index += self.biases_hidden[i].size

        self.weights_hidden_output = params[
            current_index : current_index + hidden_output_size
        ].reshape(self.weights_hidden_output.shape)
        current_index += hidden_output_size
        self.bias_output = params[
            current_index : current_index + self.bias_output.size
        ].reshape(self.bias_output.shape)


def loss_function(params, X, y, mlp: MLP):
    mlp.set_params(params)
    predictions = mlp.forward(X)
    return mean_squared_error(y, predictions)


def evaluate_gradient_mlp(hidden_sizes, data: TrainData):
    mlp = MLP(hidden_sizes)
    result_gradient = minimize(
        loss_function,
        mlp.get_params(),
        args=(data.X_train, data.y_train, mlp),
        method="L-BFGS-B",
    )
    mlp.set_params(result_gradient.x)

    predictions_test = mlp.forward(data.X_test)
    mse_test = mean_squared_error(data.y_test, predictions_test)

    return mse_test, predictions_test, result_gradient.x


def main(args):

    if(len(args) < 1):
        layers = 3
    else:
        layers = int(args[1])

    neuron_amounts = range(3, 6)
    configurations = list(product(neuron_amounts, repeat=layers))

    best_mse = float("inf")

    data = TrainData(1000, 0.2, 42)

    for config in configurations:
        mse, eval, params = evaluate_gradient_mlp(list(config), data)
        print(f"Configuration: {config}, Mean Squared Error: {mse:.2f}")

        if mse < best_mse:
            best_params = params
            best_mse = mse
            best_eval = eval
            best_configuration = config

    evo_mlp = MLP(list(best_configuration))
    evolutionary_output = differential_evolution(
        loss_function,
        [(-50, 50)] * len(evo_mlp.get_params()),
        args=(data.X_train, data.y_train, evo_mlp),
        strategy='best2exp'
    )

    evo_mlp.set_params(evolutionary_output.x)
    predictions_evolutionary = evo_mlp.forward(data.X_test)


    config_string = '-'.join(map(str, best_configuration))
    x_values = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y_target = target_function(x_values)

    plt.scatter(data.X_train, data.y_train, label="Training Data", alpha=0.2)
    plt.plot(x_values, y_target, label="Target Function", color="green", linewidth=2)
    plt.scatter(data.X_test, best_eval, label="MLP Gradient Predctions", color="yellow")
    plt.scatter(data.X_test, predictions_evolutionary, label="MLP Evolutionary Predictions", color="red")
    plt.title(f"Target Function vs MLP Predictions \n MLP Configuration: {config_string}")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.savefig(f'{config_string}.pdf')
    plt.close()

    mse_evo = mean_squared_error(data.y_test, predictions_evolutionary)
    print(f"Best Configuration: {best_configuration}")
    print(f"Gradient Mean Squared Error: {best_mse:.2f}")
    print(f"Evolutionary Mean Squeared Error: {mse_evo:.2f}")
    print(f'Best Gradient Weights: {best_params}')
    print(f'Best Evolutionary Weights: {evolutionary_output.x}')


if __name__ == "__main__":
    main(sys.argv)
