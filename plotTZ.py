from sqlalchemy import create_engine
import pandas as pd
import windowfeatures as wf
import sqlalchemy
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path
from scipy.stats.stats import pearsonr
import seaborn as sns
def expSVRPrediction(lstpred):
    a= np.array(lstpred).clip(min=0)
    exp= (np.power(2, a)-1)/10000
    return  exp


database_username = 'root'
database_password = ''
database_ip       = '127.0.0.1'
database_name     = 'xmass'
database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                               format(database_username, database_password,
                                                      database_ip, database_name))

df = pd.read_sql('SELECT * FROM cid2_results', con=database_connection)

#sns.boxplot(data=df, x="Training_Size", y="PC")

import seaborn as sns



box_plot = sns.boxplot(x="Training_Size",y="PC",data=df)

medians = df.groupby(['Training_Size'])['PC'].median()
vertical_offset = df['PC'].median() * 0.05 # offset from median for display

for xtick in box_plot.get_xticks():
    box_plot.text(xtick,round(medians[xtick],4) + vertical_offset,round(medians[xtick],4),
            horizontalalignment='center',size='15',color='black',weight='semibold')