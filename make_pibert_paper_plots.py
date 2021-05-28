import numpy as np
import matplotlib.pyplot as plt

import argparse
import os


def collect_data(data_dir, task):
    if task == 'mnali':
        data_m = np.load(os.path.join(data_dir, 'pibert_predictions_mnli.npy'))
        data_mm = np.load(os.path.join(data_dir, 'pibert_predictions_mnli-mm.npy'))
        match_m = np.load(os.path.join(data_dir, 'pibert_match_data_mnli.npy'))
        match_mm = np.load(os.path.join(data_dir, 'pibert_match_data_mnli-mm.npy'))

        data = np.vstack([data_m, data_mm])
        match_data = np.vstack([match_m, match_mm])
    else:
        data = np.load(os.path.join(data_dir, f'pibert_predictions_{task}.npy'))
        match_data = np.load(os.path.join(data_dir, f'pibert_match_data_{task}.npy'))

    return data, match_data


def plot_sentence_samples(seq_data, data_dir, task):
    data = seq_data[np.argsort(seq_data[:, 0])]
    n = data.shape[0]
    sampled_data = data[1:n:int(np.ceil(n/20)), 1:]

    plt.figure(figsize=[4,2], constrained_layout=True)
    plt.plot(np.arange(1, data.shape[1]), sampled_data.transpose())
    plt.xlabel('Layer')
    plt.ylabel('Pruned sequence\nlength')
    plt.xticks(np.arange(2, data.shape[1], step=2))
    plt.savefig(os.path.join(data_dir, f'{task}_sampled_token_pruning_trajectories_short.pdf'))
    plt.close()


def plot_first_vs_last(seq_data, match_data, data_dir, task):
    final = [3, 6, 9]
    slopes = [1, .6, .2]
    
    ymax = np.max(seq_data[:, final[0]]) + 5

    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=[8,2])

    for j in range(3):
     x = seq_data[:, 0]
     y = seq_data[:, final[j]]
     heatmap = np.zeros((ymax, np.max(x) + 1))
     for i in range(seq_data.shape[0]):
         heatmap[y[i], x[i]] += 1

    cmax = heatmap.max()

    for j in range(3):
     x = seq_data[:, 0]
     y = seq_data[:, final[j]]
     # ymax = np.max(y) + 5
     heatmap = np.zeros((ymax, np.max(x) + 1))
     for i in range(seq_data.shape[0]):
         heatmap[y[i], x[i]] += 1

     heatmap[heatmap == 0] = np.nan

     ax = axs[j]
     ax.set_facecolor('#d0d0d0')
     ax.plot([0, ymax], [0, ymax], 'k--')
     ax.plot([0, ymax], [0, slopes[j]*ymax], 'k:')

     im = ax.imshow(heatmap, cmap='Reds', vmin=0, vmax=cmax)
     ax.invert_yaxis()
     if j ==2:
      fig.colorbar(im, ax=ax, shrink=0.7)

     ax.set_xlabel('Initial sequence length')
     ax.set_ylabel(f'Sequence length\nat Layer {final[j]}')

    plt.savefig(os.path.join(data_dir, f'{task}_heatmaps_short.pdf'))
    plt.close()


def plot_accuracy_histograms(seq_data, match_data, data_dir, task):
    match = (match_data[:, 0] == match_data[:, 1])
    not_match = (match_data[:, 0] != match_data[:, 1])

    initial_seq_lengths = seq_data[:, 0]

    final = [0, 3, 6, 9]
    mx = np.max(seq_data)
    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=[8,1.5])
    ymax = 0
    for i in range(4):
      final_seq_lengths = seq_data[:, final[i]]
      ax = axs[i]
      h = ax.hist([ final_seq_lengths[not_match], final_seq_lengths[match]], bins=np.arange(-.5,mx+1), alpha=1, color=['firebrick', 'lightgreen'], density=True, stacked=True)
      # ymax = max(0, h[0].max())
      #ax.hist(final_seq_lengths[not_match], range(mx+1), alpha=1, label='incorrect predictions', color='firebrick', density=True)
      ax.set_xlabel(f'Layer {final[i] + 1}')
      if i == 0:
        ax.set_ylabel('Density')
      ax.grid(True, linestyle=':', linewidth=.1)

    for i in range(4):
    #   axs[i].set_ylim([0, ymax+5])
      axs[i].set_xlim([0, mx])

    plt.savefig(os.path.join(data_dir, f'{task}_sequence_length_accuracy_short.pdf'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot some outputs of run_glue_pibert.')
    parser.add_argument('data_dir', help='output directory')
    parser.add_argument('task', help='task')
    args = parser.parse_args()

    data, match_data = collect_data(args.data_dir, args.task)
    plot_sentence_samples(data, args.data_dir, args.task)
    plot_first_vs_last(data, match_data, args.data_dir, args.task)
    plot_accuracy_histograms(data, match_data, args.data_dir, args.task)

if __name__ == '__main__':
    main()
