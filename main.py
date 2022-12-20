'''
@Autor: Favio Salinas email: faviobol@hotmail.com
XMASS is a tool to train models for peptide intensities predictions and retention times
'''
from flask import Flask, request, send_file
from flask import render_template
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from keras.preprocessing.text import tokenizer_from_json
from tensorflow import keras
from keras.utils import pad_sequences
from scipy.stats.stats import pearsonr
from pathlib import Path
from xmass_plot import XMassPlot
import windowfeatures as wf
import numpy as np
import xgboost as xgb
import json
import os
import time
import pandas as pd
import retention_time_prediction as rtp
import os.path
import base64
from datetime import date

# Global Variables == models

DICT_MODELS_ACTION=dict()


# create the Flask app
app = Flask(__name__)

def load_models_in_memory():
    '''
    To speed up the predictions for the REST API. All the models are loaded at the beginning
    '''
    print("Loading models in memory......")
    lst_models = os.listdir("models")
    for m in lst_models:
        DICT_MODELS_ACTION[m]= {'MSMS': True, 'RT':False}
        if os.path.isfile(Path("models", m, "model_y.json")):
            xgb_model_loaded_y = xgb.XGBRegressor()
            xgb_model_loaded_b = xgb.XGBRegressor()
            xgb_model_loaded_y.load_model(Path("models",m,"model_y.json"))
            xgb_model_loaded_b.load_model(Path("models",m,"model_b.json"))
            DICT_MODELS_ACTION[m]['MSMS']= True
            DICT_MODELS_ACTION[m]['model_y']= xgb_model_loaded_y
            DICT_MODELS_ACTION[m]['model_b']= xgb_model_loaded_b
        if os.path.isfile(Path("models", m, "model_rt.h5")):
            DICT_MODELS_ACTION[m]['RT']= True
            DICT_MODELS_ACTION[m]['model_RT']= keras.models.load_model(Path("models", m, "model_rt.h5"),compile=False)

load_models_in_memory()

def exponential_prediction(list_predictions: list):
    '''
    For the training dataset a log function is applied, this is the reverse funcion (More information on XMass paper)
    Predictions that are negative will be changed to cero.
    '''
    a= np.array(list_predictions).clip(min=0)
    exp= (np.power(2, a)-1)#/10000
    return  exp
def exponential_prediction_array(array):
    '''
    For the training dataset a log function is applied, this is the reverse funcion (More information on XMass paper)
    Predictions that are negative will be changed to cero.
    '''
    a= array.clip(min=0)
    exp= (np.power(2, a)-1)
    return  exp

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/download')
def download_file():
    '''
    Downloads a template file that can be used as example of how to submit data to get predictions from text files
    '''
    filename = Path("download","xmass_template.txt")
    return send_file(filename, as_attachment=True)

@app.route('/load_list_models', methods=['POST','GET'])
def load_list_models():
    '''
    Returns a list of all the models that are contained in the folder models. If the folder is empty, it returns an empty list
    '''
    path_models= Path("models")
    lst_models = os.listdir(path_models)
    return json.dumps(sorted(lst_models))

@app.route('/get_image_rt', methods=['POST','GET'])
def get_image_rt():
    model_name = request.form['model']
    print("Get Retention Time image for model:"+model_name)
    encoded_string=""
    filename = Path("models",model_name,"graph_rt.png")
    if os.path.exists(filename):
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return json.dumps({'status': 'ok', 'image': encoded_string})
    else:
        return json.dumps({'status': 'error', 'message': 'Image does not exists','image': ''})

@app.route('/get_image_msms', methods=['POST','GET'])
def get_image_msms():
    model_name = request.form['model']
    print("Get MSMS image for model:"+model_name)
    encoded_string=""
    filename = Path("models",model_name,"graph_msms.png")
    if os.path.exists(filename):
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return json.dumps({'status': 'ok', 'image': encoded_string})
    else:
        return json.dumps({'status': 'error', 'message': 'Image does not exists','image': ''})

@app.route('/train_unique', methods=['POST'])
def train_unique():
    if request.method == 'POST':
        path = request.form['local_path']
        andromeda_score = request.form['andromeda_score']
        fragmentation = request.form['fragmentation']
        charge = request.form['charge']
        rest_api_name = request.form['rest_api_nam']
        path_api = Path("models",rest_api_name)
        flag_msms = request.form['msms']
        flag_rt = request.form['rt']
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, "models", rest_api_name)
        if rest_api_name.strip() =="":
            return dict({'status': 'error', 'message': 'Please provide a name for the REST API'})
        if os.path.exists(final_directory):
            return dict({'status': 'error', 'message': 'REST API: '+rest_api_name+', already exists'})
        try:
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)
        except OSError:
            message="Creation of the directory %s failed" % final_directory
            print ("ERROR:"+message)
            return dict({'status': 'error', 'message': message})
        print("Successfully created the directory %s " % final_directory)
        lstMSMSfiles = []
        appended_data = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "msms.txt":
                    lstMSMSfiles.append(os.path.realpath(os.path.join(root, file)))
        for m in lstMSMSfiles:
            msms = pd.read_csv(m,
                                error_bad_lines=False,
                                dtype={
                                   "Fragmentation": str,
                                    "Scan number": float,
                                    "Matches": str,
                                   "Intensities": str
                                   },
                                sep='\t',
                                lineterminator='\r')
            msmsFiltered = msms[
                    ["Raw file", "Scan number", "Sequence", "Modifications", "Retention time",
                     "Score", "Charge", "Fragmentation", "Mass analyzer", "Matches", "Intensities","Type"]]
            msmsFiltered = msmsFiltered[msmsFiltered['Score'] >= int(andromeda_score)]
            msmsFiltered = msmsFiltered[msmsFiltered['Charge'] >= int(charge)]
            msmsFiltered = msmsFiltered[msmsFiltered['Fragmentation'] == fragmentation]
            #msmsFiltered = msmsFiltered[msmsFiltered['Retention time'] <= 85]
            msmsFiltered = msmsFiltered[msmsFiltered['Modifications'] == "Unmodified"]
            appended_data.append(msmsFiltered)
        df_peptides_train = pd.concat(appended_data)
        # Filtering unique sequences by max Andromeda Score
        idx1= df_peptides_train.groupby(['Sequence'])['Score'].transform(max)==df_peptides_train['Score']
        df_maxScore = df_peptides_train[idx1][['Sequence','Matches','Intensities']]


        if flag_msms == "1":
            # %%
            count=0
            start_time = time.time()
            feature_matrix_x = []
            target_y = []
            target_b = []
            # Splitting data into training and testing
            train_msms, test_aux_msms = train_test_split(df_maxScore, test_size=0.2, random_state=42)
            test_msms, validation_msms = train_test_split(test_aux_msms, test_size=0.5, random_state=42)
            print("Shapes")
            print("Train" + str(len(train_msms)))
            print("Validation" + str(len(test_msms)))
            print("Test" + str(len(validation_msms)))

            for index, row in train_msms.iterrows():
                myTrainingMatrix = wf.createWindowData(row["Sequence"], row["Matches"], row["Intensities"])
                myTrainingMatrix.GenerateMatrix(24)
                d = myTrainingMatrix.GenerateDataset(removeZeros=False)
                for i in range(0, len(d['target_Y'])):
                    feature_matrix_x.append(d['featureMatrix'][i])
                    target_y.append(np.log2(1 + (1000 * d['target_Y'][i])))
                    target_b.append(np.log2(1 + (1000 * d['target_B'][i])))
            print("--- Creating feature Matrix %s seconds ---" % (time.time() - start_time))

            model_y = XGBRegressor(
                max_depth=12, learning_rate=0.1, n_estimators=1400,
                objective="reg:linear",
                nthread=8,
                gamma=0,
                min_child_weight=1, max_delta_step=0, subsample=0.8, colsample_bytree=0.8,
                colsample_bylevel=0.5, colsample_bynode=0.5, reg_alpha=0, reg_lambda=0.8,
                scale_pos_weight=1, base_score=0.5
            )

            model_y.fit(np.array(feature_matrix_x), target_y)

            try:
                print("Model to create")
                print(Path(final_directory,"model_y.json"))
                model_y.save_model(Path(final_directory,"model_y.json"))
            except OSError:
                print ("Error: Model was not created")
                print(Path(final_directory,"model_y.json"))

            model_b = XGBRegressor(
                max_depth=18, learning_rate=0.1, n_estimators=1300,
                objective="reg:linear",
                nthread=8,
                gamma=0,
                min_child_weight=1, max_delta_step=0, subsample=0.8, colsample_bytree=0.8,
                colsample_bylevel=0.6, colsample_bynode=0.7, reg_alpha=0, reg_lambda=0.8,
                scale_pos_weight=1, base_score=0.5
            )
            model_b.fit(np.array(feature_matrix_x), target_b)

            model_b.save_model(Path(final_directory,"model_b.json"))
            print ("Training MSMS - DONE")

            # PC test set
            pc_validation = []
            for index, row in validation_msms.iterrows():
                myTrainingMatrix = wf.createWindowData(row["Sequence"], row["Matches"], row["Intensities"])
                myTrainingMatrix.GenerateMatrix(24)
                d = myTrainingMatrix.GenerateDataset(removeZeros=False)
                y_real = []
                y_pred = []
                b_real = []
                b_pred = []
                for i in range(0, len(d['target_Y'])):
                    # log2(1 + (10000 * BIon)
                    y_real.append(d['target_Y'][i])
                    b_real.append(d['target_B'][i])
                    y_pred.append(model_y.predict(np.array(d['featureMatrix'][i]).reshape(1, -1))[0])
                    b_pred.append(model_b.predict(np.array(d['featureMatrix'][i]).reshape(1, -1))[0])
                pc_validation.append(pearsonr(y_real + b_real, exponential_prediction(y_pred + b_pred))[0])
            print(len(pc_validation))

            pc_test = []
            for index, row in test_msms.iterrows():
                myTrainingMatrix = wf.createWindowData(row["Sequence"], row["Matches"], row["Intensities"])
                myTrainingMatrix.GenerateMatrix(24)
                d = myTrainingMatrix.GenerateDataset(removeZeros=False)
                y_real = []
                y_pred = []
                b_real = []
                b_pred = []
                for i in range(0, len(d['target_Y'])):
                    # log2(1 + (10000 * BIon)
                    y_real.append(d['target_Y'][i])
                    b_real.append(d['target_B'][i])
                    y_pred.append(model_y.predict(np.array(d['featureMatrix'][i]).reshape(1, -1))[0])
                    b_pred.append(model_b.predict(np.array(d['featureMatrix'][i]).reshape(1, -1))[0])
                pc_test.append(pearsonr(y_real + b_real, exponential_prediction(y_pred + b_pred))[0])
            print(len(pc_validation))

            results = pd.DataFrame(
                {
                    'Dataset': ["Test"] * len(pc_validation) + ["Validation"] * len(pc_test),
                    'PC':  pc_test + pc_validation

                })

            results.to_csv(Path(final_directory,"test_information_msms.tsv"), sep='\t')
            xmass_plot= XMassPlot(folder_model_path=final_directory, size_image=(10,5))
            xmass_plot.plot_msms()
            print("Training Images - DONE")

        if flag_rt == "1":
            print("?"+final_directory)

            idx1 = df_peptides_train.groupby(['Sequence'])['Score'].transform(max) == df_peptides_train['Score']
            print (len(idx1))
            print(idx1)
            df_maxScoreRT = df_peptides_train[idx1][['Sequence', 'Matches', 'Retention time']]
            max_rt = max(df_maxScoreRT["Retention time"])
            df_maxScoreRT["Retention time aux"] = df_maxScoreRT["Retention time"]
            df_maxScoreRT["Retention time"] = df_maxScoreRT["Retention time"] / max_rt
            print (df_maxScoreRT)
            peptides = df_maxScoreRT["Sequence"].to_list()
            retention_times= df_maxScoreRT["Retention time"].to_list()
            # Initializing class, you can change different parameters like number of epochs, batch size etc, before training
            RTP = rtp.RetentionTimePrediction(peptides, retention_times,path_api)
            RTP.epochs = 40
            # RTP.epochs <- Example changing number of epochs (the default is 40 ). You have to do it before training
            RTP.train()
            RTP.model.save(Path(final_directory,"model_rt.h5"))
            a_file = open(Path(final_directory,"max_rt.json"), "w")
            d=dict()

            d['max_rt'] = str(max_rt)
            json.dump(d,a_file)
            a_file.close()
            # Saving evaluation RT
            # Saving testing value
            TestingFile = Path(final_directory,"test_information_rt.tsv")
            file_testing = open(TestingFile, 'w')
            file_testing.write("real_value\tpredicted_value\n")
            file_testing.close()
            file_testing = open(TestingFile, 'a')
            x = np.array(RTP.Y_test)  # *maxNum
            y = RTP.predict_feature_vector(RTP.X_test).flatten()  # *maxNum

            for idx, val in enumerate(x):
                file_testing.write(str(x[idx]*max_rt) + "\t" + str(y[idx]*max_rt) + "\n")
            file_testing.close()
            xmass_plot= XMassPlot(folder_model_path=final_directory, size_image=(10,5))
            xmass_plot.plot_retention_time()

        load_models_in_memory()
        return json.dumps({'status': 'ok', 'message': 'The model was succesfully created!'})


@app.route('/select_folder_training', methods=['POST'])
def select_folder_training():
    if request.method == 'POST':
        path = request.form['local_path']
        andromeda_score = request.form['andromeda_score']
        print(path)
        if (path.strip()) == "":
            return dict({'status': 'error',
                         'message': 'Path of the location folder is empty.'})
        if os.path.exists(path) ==False:
            return dict({'status':'error', 'message':'Path of the location folder does not exists. Please be sure the computer where is XMASS located has access to it.'})
        print ("Androme Score:")
        print (andromeda_score)
        if andromeda_score=="":
            andromeda_score= 0
        lstMSMSfiles = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "msms.txt":
                    lstMSMSfiles.append(os.path.realpath(os.path.join(root, file)))

        appended_data = []

        if len(lstMSMSfiles)==0:
            return dict({'status': 'error',
                         'message': 'No msms.txt files were not found in the folder path you specified.'})

        for m in lstMSMSfiles:
            msms = pd.read_csv(m,
                               error_bad_lines=False,
                               dtype={
                                   "Fragmentation": str,
                                   "Scan number": float,
                                   "Matches": str,
                                   "Intensities": str,
                                   "Masses": str

                               },
                               sep='\t',
                               lineterminator='\r')
            msmsFiltered = msms[
                ["Raw file", "Scan number", "Sequence", "Modifications",
                 "Score", "Charge", "Fragmentation", "Mass analyzer", "Matches", "Intensities",
                 "Masses",
                 "Type"]]


            msmsFiltered = msmsFiltered[msmsFiltered['Score'] >= int(andromeda_score)]
            msmsFiltered = msmsFiltered[msmsFiltered['Modifications'] == "Unmodified"]
            appended_data.append(msmsFiltered)
        
        appended_data = pd.concat(appended_data)
        appended_data_d = appended_data[["Fragmentation", "Charge","Sequence"]]
        appended_data_d= appended_data_d.groupby(["Fragmentation", "Charge"], as_index=False)['Sequence'].nunique().rename(columns={'Sequence':'NumberPeptides'})
        appended_data_d = appended_data_d.dropna()
        appended_data_d["Charge"] = appended_data_d['Charge'].astype(int)

        data = []
        for index, row in appended_data_d.iterrows():
            d = dict()
            d["id"] = index
            d["Fragmentation"] = row['Fragmentation']
            d["Charge"] = row['Charge']
            d["NumberPeptides"] = row['NumberPeptides']
            data.append(d)
        return dict({'status':'ok', 'message':'No errors found', 'data':data})

@app.route('/predict/<model_name>', methods=['POST','GET'])
def predict(model_name):
    start_time = time.time()
    peptides = request.get_json()
    list_matches=[]
    list_intensities=[]
    list_rt_times=[]
    global_list=[]
    
    msms_predictions= DICT_MODELS_ACTION[model_name]['MSMS']
    rt_predicitions= DICT_MODELS_ACTION[model_name]['RT']

    if msms_predictions:
        array_features=np.empty((0,549), int)
        for seq in peptides:
            myTrainingMatrix = wf.createWindowData(seq['peptide'], "y1", "1")
            myTrainingMatrix.GenerateMatrix(24)
            d = myTrainingMatrix.GenerateDataset(removeZeros=False)
            array_features=np.append(array_features, d['featureMatrix'], axis=0)   
        
        y_ions = (DICT_MODELS_ACTION[model_name]['model_y'].predict(np.array(array_features)))
        b_ions = (DICT_MODELS_ACTION[model_name]['model_b'].predict(np.array(array_features)))
        
        y_ions=exponential_prediction_array(y_ions).round(decimals=4, out=None)
        b_ions=exponential_prediction_array(b_ions).round(decimals=4, out=None)

        start=0
        for seq in peptides:
            end= start + len(seq['peptide'])-1

            ystr=["y"]*(len(seq['peptide'])-1)
            ynum=list(range((len(seq['peptide'])-1), 0, -1))
            ions_aux_y = [m + str(n) for m, n in zip(ystr, ynum)]

            bstr = ["b"] * (len(seq['peptide']) - 1)
            bnum = list(range(1, len(seq['peptide'])))
            ions_aux_b = [m + str(n) for m, n in zip(bstr, bnum)]

            ions=";".join(map(str, ions_aux_y+ions_aux_b))
            list_matches.append(ions)

            intensities=";".join(map(str, np.concatenate((y_ions[start:end],b_ions[start:end]), axis=None)))
            list_intensities.append(intensities)

            start=end

    lst_peptides=[seq['peptide'] for seq in peptides]
    if rt_predicitions:
        max_val_rt = 1
        with open(Path("models", model_name, "max_rt.json")) as json_file:
            data = json.load(json_file)
            max_val_rt = float(data["max_rt"])
        with open(Path("models", model_name, "tokenizer.json")) as f:
            data = json.load(f)
            loaded_tokenizer = tokenizer_from_json(data)
            feature_vector = loaded_tokenizer.texts_to_sequences(lst_peptides)
            feature_vector = pad_sequences(feature_vector, maxlen=50)
            rt_pred = DICT_MODELS_ACTION[model_name]['model_RT'].predict(feature_vector) * float(max_val_rt)
            list_rt_times=rt_pred

    if msms_predictions==True and rt_predicitions==True:
        for idx, seq in enumerate(peptides):
            d = dict()
            d["Peptide"] = seq['peptide']
            d["Intensities"] = list_intensities[idx]
            d["Matches"] = list_matches[idx]
            d["RT"] = str(list_rt_times[idx][0].round(decimals=4, out=None))
            global_list.append(d)

    if msms_predictions==True and rt_predicitions==False:
        for idx, seq in enumerate(peptides):
            d = dict()
            d["Peptide"] = seq
            d["Intensities"] = list_intensities[idx]
            d["Matches"] = list_matches[idx]
            global_list.append(d)
        
    if msms_predictions==False and rt_predicitions==True:
        for idx, seq in enumerate(peptides):
            d = dict()
            d["RT"] = str(list_rt_times[idx][0].round(decimals=4, out=None))
            global_list.append(d)

    end_time = time.time()
    print("Prediction took this long to run: {}".format(end_time - start_time))
    df_temp = pd.DataFrame(global_list)

    today = date.today()
    df_temp.to_csv(Path("temp_predictions",model_name+today.strftime("%d_%m_%Y")+".txt"), sep='\t' , index=False)
    print("Temporal file saved as: ", Path("temp_predictions",model_name+today.strftime("%d_%m_%Y")+".txt"))
    return json.dumps(global_list)

if __name__ == '__main__':
    #app.run(debug=True, port=5000) #-> Use this in case you want to debug
    app.run(host="0.0.0.0")
