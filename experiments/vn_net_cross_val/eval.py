import pandas as pd
import matplotlib.pyplot as plt
import os

# Specify the path to your CSV file
dir_path = '/Users/shizhehe/dev/research/vector_neurons_mri/experiments/vn_net_cross_val'

csv_file_names = ['stat_fold_0.csv', 'stat_fold_1.csv', 'stat_fold_2.csv', 'stat_fold_3.csv', 'stat_fold_4.csv']


# Read and store validation loss data from each file
for mode in ['train', 'validation']:
    eval_losses = []
    for file_name in csv_file_names:
        csv_file_path = os.path.join(dir_path, file_name)
        df = pd.read_csv(csv_file_path)

        if mode == 'train':
            eval_data = df[df['info'].str.startswith('epoch')]
        else:
            eval_data = df[df['info'] == 'val']
        eval_loss = eval_data['all']
        
        eval_loss = eval_loss[:45]

        eval_losses.append(eval_loss)

    average_eval_loss = [sum(losses) / len(losses) for losses in zip(*eval_losses)]

    epochs = range(0, 45)

    plt.figure(figsize=(10, 6))

    for i, loss in enumerate(eval_losses):
        plt.plot(epochs, loss, label=f'{mode} Loss (Fold {i + 1})', linestyle='', marker='.')

    plt.plot(epochs, average_eval_loss, label=f'Average {mode} Loss', linestyle='-', marker='')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{mode} Loss Comparison and Average Across Cross-Validation Folds')
    plt.legend()
    plt.grid(True)
    plt.show()
