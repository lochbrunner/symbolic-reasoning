import matplotlib.pyplot as plt


def plot_ranks(ranks, filename='../out/ml/lstm-ranks.svg'):
    fig = plt.figure(figsize=(8, 6))
    # Transpose ranks
    ranks = list(map(list, zip(*ranks)))
    x = range(len(ranks[0]))
    for i, rank in enumerate(ranks[:5]):
        plt.plot(x, rank, label=f'top {i+1}')
    plt.legend()
    plt.savefig(filename)
