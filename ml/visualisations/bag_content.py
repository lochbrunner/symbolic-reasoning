import matplotlib.pyplot as plt


def plot_used_rules(bag, filename='../out/ml/lstm-rule-hist.svg'):
    plt.figure(figsize=(8, 6))
    rules = bag.meta.rules
    x = range(len(rules))
    sum_of_rules = sum([rule.fits for rule in rules])
    plt.barh(
        x, width=[rule.fits/sum_of_rules for rule in rules], align='center')
    plt.yticks(x, [f'$ {stat.rule.latex} $' for stat in rules])
    plt.tight_layout()
    plt.savefig(filename)
