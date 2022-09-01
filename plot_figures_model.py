'''
Created on Aug 9, 2018
@author: salinas
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import r2_score

def plot_retention_time(path):
    plt.clf()
    test_file = Path(path,"test_information_rt.tsv")
    train_file = Path(path,"train_information_rt.tsv")
    figure = Path(path,"graph_rt.png")

    test = pd.read_csv(test_file, sep='\t', skip_blank_lines=True).dropna()
    test = test.replace(r'\n', '', regex=True)
    train = pd.read_csv(train_file, sep='\t',  skip_blank_lines=True).dropna()
    train = train.replace(r'\n', '', regex=True)

    test["real_value"] = pd.to_numeric(test["real_value"])

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("RT: Training, validation and testing Results", fontsize=16)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    print("que pasa??")
    print(train)
    # Figure 1
    ax1.plot(train['train_loss'].astype(float))
    ax1.plot(train['val_loss'].astype(float))
    ax1.set_title('model train vs validation loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper right')
    ax1.grid(which='major', axis='both', linestyle='--')
    # Figure 2
    x = test['real_value'].tolist()
    y = test['predicted_value'].tolist()

    pc = round(pearsonr(x, y)[0], 5)
    r2= round(r2_score(x,y),5)
    ax2.set_title('Scatter plot real vs predictions (test dataset)')
    ax2.scatter(x, y, label='Scatter Plot')
    ax2.set_xlabel("Real Values")
    ax2.set_ylabel("Predicted Values")
    patch_pc = mpatches.Patch(color="white", label="PC: "+str(pc))
    patch_r2 = mpatches.Patch(color="white", label="R2: "+str(r2))
    ax2.legend(handles=[patch_pc,patch_r2])
    ax2.grid(which='major', axis='both', linestyle='--')
    plt.savefig(figure, dpi=300)

def plot_msms(path):
    plt.clf()
    sns.set(rc={'figure.figsize': (10, 5)})
    test_file = Path(path,"test_information_msms.tsv")
    figure = Path(path,"graph_msms.png")
    results=pd.read_csv(test_file, sep='\t')

    sns.set(style="whitegrid")

    box_plot = sns.boxplot(x="Dataset", y="PC", data=results)

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(lines[4 + cat * 6].get_ydata()[0], 4)
        print("===Values===")
        print(y)
        ax.text(
            cat,
            y,
            f'{y}',
            ha='center',
            va='center',
            fontweight='bold',
            size=10,
            color='white',
            bbox=dict(facecolor='#445A64'))

    box_plot.figure.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.title("PC MSMS spectra Real vs Predicted : Test and Validation")
    plt.savefig(figure, dpi=300)
