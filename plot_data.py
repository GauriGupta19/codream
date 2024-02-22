import matplotlib.pyplot as plt
import json
import os
import re

def get_max_accuracy(json_file):
  with open(json_file) as f:
    data = json.load(f)
  # print(data)
  max_rounds = 400
  return max([x[2] for x in data[:max_rounds]])

# initialize results dict
scaling_results = {
    'FedAvg': {},
    'Co-Dream': {},
    'Isolated': {},
}
num_clients_setting = ['2', '4', '8', '12', '24']
for client_id in num_clients_setting:
  scaling_results['FedAvg'][client_id] = None
  scaling_results['Co-Dream'][client_id] = None
  scaling_results['Isolated'][client_id] = None


def load_results_fedavg():
    base_dir = './fl/'
    json_files = os.listdir(base_dir)
    for json_file in json_files:
        if json_file.endswith('.json'):
            match = re.search(r"(\d+)clients", json_file)
            if match:
                num_clients = int(match.group(1))
            else:
                print("No match")
            max_accuracy = get_max_accuracy(base_dir + json_file)
            scaling_results['FedAvg'][str(num_clients)] = [max_accuracy, None]

def load_results_codream():
    for client_id in num_clients_setting:
        scaling_results['Co-Dream'][client_id] = None
        base_dir = './codream_' + client_id + '_clients/'
        try:
            json_files = os.listdir(base_dir)
        except:
            continue
        max_accuracies = []
        for json_file in json_files:
            if json_file.endswith('.json'):
                match = re.search(r"(\d+)clients", json_file)
                if match:
                    num_clients = int(match.group(1))
                else:
                    print("No match")
                max_accuracy = get_max_accuracy(base_dir + json_file)
                max_accuracies.append(max_accuracy)
        if len(max_accuracies) == 0:
            continue
        # get mean and std
        max_accuracy = sum(max_accuracies) / len(max_accuracies)
        std = 0
        for accuracy in max_accuracies:
            std += (accuracy - max_accuracy) ** 2
        std = (std / len(max_accuracies)) ** 0.5
        scaling_results['Co-Dream'][str(num_clients)] = [max_accuracy, std]

def load_results_isolated():
    base_dir = './isolated/'
    json_files = os.listdir(base_dir)
    numSamplesToClients = {
        '1000': '24',
        '2000': '12',
        '3000': '8',
        '6000': '4',
        '12000': '2',
    }
    for json_file in json_files:
        if json_file.endswith('.json'):
            num_samples = re.search(r"(\d+)samples", json_file)
            if num_samples:
                num_samples = num_samples.group(1)
                print(num_samples)
            else:
                print("No match")
            max_accuracy = get_max_accuracy(base_dir + json_file)
            scaling_results['Isolated'][numSamplesToClients[num_samples]] = [max_accuracy, None]

load_results_fedavg()
load_results_codream()
load_results_isolated()

plt.figure()
algorithmToMarker = {
    'FedAvg': '^',
    'Co-Dream': 's',
    'Isolated': 'o',
}

for algorithm, data in scaling_results.items():
    markerAlgo = algorithmToMarker[algorithm]
    if data is not None:
        x_values = []
        y_values = []
        error_bars = []
        for key, result in data.items():
            if result is not None:
                mu, sigma = result
                x_values.append(int(key))
                y_values.append(mu)
                if sigma is not None:
                    error_bars.append(sigma)
        # plot if no error bars are available
        if len(error_bars) == 0:
            plt.plot(x_values, y_values, label=algorithm, marker=markerAlgo, ms=10)
        # plot if error bars are available
        if len(error_bars) > 0:
            plt.errorbar(x_values, y_values, yerr=error_bars, label=algorithm, marker=markerAlgo, ms=10)
# make x ticks as num_clients_setting
plt.xticks([2, 4, 8, 12, 24])
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.grid(alpha=0.3)
plt.xlabel('Number of clients', fontsize=14)
plt.ylabel('Test accuracy', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()