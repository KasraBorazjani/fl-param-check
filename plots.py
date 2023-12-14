import matplotlib.pyplot as plt
import os

def draw_bar_plot(vars_to_draw, var_labels, color_list, plt_xlabel, plt_ylabel, plt_title, filename, result_dir):
    fig, ax = plt.subplots()
    bar_width = 0.5/len(vars_to_draw)
    cursor = -0.25
    for i, var_stat in enumerate(vars_to_draw):
        ax.bar(var_stat[0]+cursor, var_stat[1], bar_width, label=var_labels[i], color=color_list[i], alpha=0.7)
        cursor += bar_width
    ax.set_xlabel(plt_xlabel)
    ax.set_ylabel(plt_ylabel)
    ax.set_title(plt_title)
    ax.legend()
    plt.savefig(os.path.join(result_dir, filename))