import retention_time_prediction as rtp

import pandas as pd

df=pd.read_csv(r"C:\Favio\xmass\training_folder\crosslinking_1\msms.txt", delimiter="\t")
idx1 = df.groupby(['Sequence'])['Score'].transform(max) == df['Score']
df_maxScoreRT = df[idx1][['Sequence', 'Retention time']]
p=df_maxScoreRT["Sequence"].to_list()
r=(df_maxScoreRT["Retention time"]/max(df_maxScoreRT["Retention time"])).to_list()
m=max(df_maxScoreRT["Retention time"])
import plot_figures_model as myplt
# Training data, 80% of the dataset will be used for training, 20% for testing, if you want to change this percentage
# just change RTP.test_size = 0.1, RT.process_data (For example 10% for testing)

peptides_to_train=["ABC1","ABC2","ABC3","ABC4","ABC5","ABC6","ABC7","ABC8","ABC9","ABC0"]
retention_times_to_train=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
# Initializing class, you can change different parameters like number of epochs, batch size etc, before training
RTP = rtp.RetentionTimePrediction(p,r,"models\\rttest")
# RTP.epochs <- Example changing number of epochs (the default is 40 ). You have to do it before training
RTP.train()
RTP.save("models\\rttest"+"\\model_rt.h5")
RTP.predict(["*S*LVSDFR|*DWDPK","*NLIGALLFDSGETAEATR|GY*EVK"])*m
#example of predictions
#peptides_to_predict =["ABC1","ABC2"]
RTP.predict_fromSavedModel1(["*S*LVSDFR|*DWDPK","*NLIGALLFDSGETAEATR|GY*EVK"],"models\\a2")*m


#myplt.plot_retention_time("C:\Favio\Xmass\models\\\experiment1_msms_rt_cid2")
#myplt.plot_msms("C:\Favio\Xmass\models\\\experiment1_msms_rt_cid2")

import os
path="C:\Favio\Xmass\training_folder\example2"
os.path.exists(path)
if os.path.exists(path) == False:
    print ( dict({'status': 'error',
                 'message': 'Path of the location folder does not exists. Please be sure the computer where is XMASS located has access to it.'}))

