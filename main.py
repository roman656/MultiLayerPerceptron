import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


def activation_function(weighted_sum):
    return 1 / (1 + np.exp(-weighted_sum))


def get_weighted_sum(inputs, weights):
    return inputs @ weights


def get_output_neuron_error(real_output, expected_output):
    return (expected_output - real_output) * real_output * (1 - real_output)


def get_hidden_neuron_error(hidden_real_output, output_weights, output_neuron_error):
    error = hidden_real_output * (1 - hidden_real_output) * (output_weights @ output_neuron_error)
    return error.reshape(error.size, 1)


def get_updated_output_weights(weights, rate, output_neuron_error, hidden_outputs):
    delta = rate * output_neuron_error * hidden_outputs
    return weights + delta.reshape([delta.size, 1])


def get_updated_hidden_weights(weights, rate, hidden_neuron_error, inputs):
    return weights + rate * (hidden_neuron_error @ inputs.reshape([1, inputs.size])).T


def get_total_energy_error(real_output, expected_output):
    return 0.5 * (expected_output - real_output)**2


def draw_results(rms_error_energy_values, epochs_amount, experiments_amount):
    x = np.linspace(start=1, stop=epochs_amount, num=epochs_amount)

    figure = plt.figure()

    for index in range(experiments_amount):
        plt.plot(x, rms_error_energy_values[index], label=f'Эксперимент №{index + 1}')

    plt.xlabel("Эпохи")
    plt.ylabel("Значения среднеквадратичной ошибки")
    plt.grid(True)
    plt.xticks(range(1, epochs_amount + 1, epochs_amount // 10))
    plt.legend(loc='upper right')

    figure.tight_layout()
    figure.savefig(fname='result.png')


def get_learning_data(flower_classes_used, data_url):
    data_frame = pd.read_csv(data_url, names=['Длина чашелистика', 'Ширина чашелистика',
                                              'Длина лепестка', 'Ширина лепестка', 'Метка класса'])
    data_frame = data_frame.loc[(data_frame['Метка класса'] == flower_classes_used[0]) |
                                (data_frame['Метка класса'] == flower_classes_used[1])]
    data_frame['Метка класса'] = np.where(data_frame['Метка класса'] == flower_classes_used[0], 0, 1)

    outputs = data_frame['Метка класса'].to_numpy()
    outputs.shape = [outputs.size, 1]
    inputs = data_frame.iloc[:, :-1].to_numpy()

    return inputs, outputs


if __name__ == '__main__':
    epochs_amount = 300
    learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    training_inputs, expected_outputs = get_learning_data(flower_classes_used=['Iris-setosa', 'Iris-virginica'],
                                                          data_url=DATA_URL)
    iterations_amount = len(expected_outputs)
    input_neurons_amount = training_inputs.shape[1]
    hidden_neurons_amount = 5
    output_neurons_amount = 1
    rms_error_energy_values = np.zeros([len(learning_rates), epochs_amount])
    hidden_weights = np.random.uniform(-1, 1, [input_neurons_amount, hidden_neurons_amount])
    output_weights = np.random.uniform(-1, 1, [hidden_neurons_amount, output_neurons_amount])

    experiment_index = 0

    for learning_rate in learning_rates:
        print(f'\nЭксперимент №{experiment_index + 1} с коэффициентом обучения: {learning_rate}')
        current_hidden_weights = hidden_weights
        current_output_weights = output_weights

        for epoch in range(epochs_amount):
            total_energy_error = 0

            for iteration in range(iterations_amount):
                hidden_outputs = activation_function(get_weighted_sum(training_inputs[iteration],
                                                                      current_hidden_weights))
                output = activation_function(get_weighted_sum(hidden_outputs,
                                                              current_output_weights))

                output_neuron_error = get_output_neuron_error(output, expected_outputs[iteration])
                hidden_neuron_error = get_hidden_neuron_error(hidden_outputs, current_output_weights,
                                                              output_neuron_error)

                current_output_weights = get_updated_output_weights(current_output_weights, learning_rate,
                                                                    output_neuron_error, hidden_outputs)
                current_hidden_weights = get_updated_hidden_weights(current_hidden_weights, learning_rate,
                                                                    hidden_neuron_error, training_inputs[iteration])

                total_energy_error += get_total_energy_error(output, expected_outputs[iteration])

            rms_error_energy = total_energy_error / len(expected_outputs)
            rms_error_energy_values[experiment_index][epoch] = rms_error_energy
            print(f'Энергия среднеквадратичной ошибки на {epoch + 1} эпохе: {rms_error_energy}')

        print(f'Первоначальные веса скрытого слоя:\n{hidden_weights}')
        print(f'Итоговые веса скрытого слоя:\n{current_hidden_weights}')
        print(f'Первоначальные веса выходного слоя:\n{output_weights}')
        print(f'Итоговые веса выходного слоя:\n{current_output_weights}')

        experiment_index += 1

    draw_results(rms_error_energy_values, epochs_amount, len(learning_rates))
