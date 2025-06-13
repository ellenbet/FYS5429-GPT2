import seaborn as sns
import matplotlib.pyplot as plt


def set_plt_params(remove_grid=False):
    """Set parameters and use seaborn theme to plot."""
    sns.set_theme()
    if remove_grid:
        sns.set_style("whitegrid", {"axes.grid": False})
    params = {
        "font.family": "Serif",
        "font.serif": "Roman", 
        "text.usetex": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium", 
        "savefig.dpi": 300, 
        "axes.grid" : False
    }
    plt.rcParams.update(params)

def plot_eval(epochs_seen, tokens_seen, train_losses, val_losses, train_label, val_label, y_label, save_as):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label = train_label)
    ax1.plot(epochs_seen, val_losses, linestyle = "-.", label = val_label)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(y_label)
    ax1.legend(loc = "upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha = 0)
    fig.tight_layout()
    plt.savefig(save_as, bbox_inches = "tight")
    plt.show()
