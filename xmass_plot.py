'''
Class to plot and save images for results of the training of retention time and MSMS predictions
'''
from pathlib import Path
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
matplotlib.use('Agg')

class XMassPlot:
    '''
    Class to plot results of the training of MSMS predictions and retention time.
    The folder that contains the results must be provided and should contain:
        - test_information_rt.tsv (Needed for retention time)
        - train_information_rt.tsv (Needed for retention time)
        - test_information_msms.tsv (Needed for msms)
    '''
    def __init__(self, folder_model_path: str, size_image= tuple, dpi_image : int =300 ):
        self.path=folder_model_path
        self.size_image =size_image
        self.dpi_image = dpi_image
        self._retention_time_test_file = Path(folder_model_path,"test_information_rt.tsv")
        self._retention_time_train_file = Path(folder_model_path,"train_information_rt.tsv")
        self._msms_test_file = Path(folder_model_path,"test_information_msms.tsv")

    def plot_retention_time(self, image_name="graph_rt.png"):
        '''
        Function to plot retention time: Training  curve and validation results
        '''
        plt.clf()
        figure = Path(self.path,image_name)
        test = pd.read_csv(self._retention_time_test_file, sep='\t',
                        skip_blank_lines=True).dropna()
        test = test.replace(r'\n', '', regex=True)
        train = pd.read_csv(self._retention_time_train_file, sep='\t',
                        skip_blank_lines=True).dropna()
        train = train.replace(r'\n', '', regex=True)
        test["real_value"] = pd.to_numeric(test["real_value"])
        fig = plt.figure(figsize=self.size_image) ###
        fig.suptitle("RT: Training, validation and testing Results", fontsize=16)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        # Figure - Left side (TRAINING CURVES)
        ax1.plot(train['train_loss'].astype(float))
        ax1.plot(train['val_loss'].astype(float))
        ax1.set_title('model train vs validation loss')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'val'], loc='upper right')
        ax1.grid(which='major', axis='both', linestyle='--')
        # Figure - Right side (SCATTER PLOT)
        real_values = test['real_value'].tolist()
        predicted_values = test['predicted_value'].tolist()
        person_correlation = round(pearsonr(real_values, predicted_values)[0], 5)
        r_squared= round(r2_score(real_values,predicted_values),5)
        ax2.set_title('Scatter plot real vs predictions (test dataset)')
        ax2.scatter(real_values, predicted_values, label='Scatter Plot')
        ax2.set_xlabel("Real Values")
        ax2.set_ylabel("Predicted Values")
        patch_pc = mpatches.Patch(color="white", label="PC: "+str(person_correlation))
        patch_r2 = mpatches.Patch(color="white", label="R2: "+str(r_squared))
        ax2.legend(handles=[patch_pc,patch_r2])
        ax2.grid(which='major', axis='both', linestyle='--')
        plt.savefig(figure, dpi=self.dpi_image )
        print("... Retention time image saved.")

    def plot_msms(self, image_name="graph_msms.png"):
        '''
        Function to plot msms: MSMS person correlation boxblots real vs predicted on test and
        validation set
        '''
        plt.clf()
        sns.set(rc={'figure.figsize': self.size_image })
        figure = Path(self.path,image_name)
        results=pd.read_csv(self._msms_test_file, sep='\t')
        sns.set(style="whitegrid")
        box_plot = sns.boxplot(x="Dataset", y="PC", data=results)
        ax = box_plot.axes
        lines = ax.get_lines()
        categories = ax.get_xticks()

        for cat in categories:
            person_corrlation_medians = round(lines[4 + cat * 6].get_ydata()[0], 4)
            ax.text(
                cat,
                person_corrlation_medians,
                f'{person_corrlation_medians}',
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
        plt.savefig(figure, dpi=self.dpi_image )
