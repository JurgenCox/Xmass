from flask import Flask, request, send_file
from flask import render_template
from sklearn.model_selection import train_test_split
import windowfeatures as wf
import numpy as np
import xgboost as xgb
import json
import os
import time
from xgboost import XGBRegressor
import pandas as pd
import retention_time_prediction as rtp
import os.path
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import plot_figures_model as myplt
import base64
from scipy.stats.stats import pearsonr
from pathlib import Path
from keras.backend import manual_variable_initialization
#manual_variable_initialization(True)

# create the Flask app
app = Flask(__name__)

# Global Variables == models

_RT_MODEL={"model_name":"", "keras_model":""}
_MSMS_MODEL={"model_name":"", "model_y":"", "model_b":""}

def expSVRPrediction(lstpred):
    a= np.array(lstpred).clip(min=0)
    exp= (np.power(2, a)-1)/10000
    return  exp

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/download')
def download_file():
    filename = Path("download","xmass_template.txt")
    return send_file(filename, as_attachment=True)

@app.route('/load_list_models', methods=['POST','GET'])
def load_list_models():
    path_models= Path("models")
    lst_models = os.listdir(path_models)
    return json.dumps(lst_models)

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
            appended_data.append(msmsFiltered)
        df_peptides_train = pd.concat(appended_data)
        # Filtering unique sequences by max Andromeda Score
        idx1= df_peptides_train.groupby(['Sequence'])['Score'].transform(max)==df_peptides_train['Score']
        df_maxScore = df_peptides_train[idx1][['Sequence','Matches','Intensities']]


        if flag_msms == "1":
            # %%
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

            ###

            model_y = XGBRegressor(
                max_depth=8, learning_rate=0.1, n_estimators=50,
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
                max_depth=8, learning_rate=0.1, n_estimators=50,
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
                pc_validation.append(pearsonr(y_real + b_real, expSVRPrediction(y_pred + b_pred))[0])
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
                pc_test.append(pearsonr(y_real + b_real, expSVRPrediction(y_pred + b_pred))[0])
            print(len(pc_validation))

            results = pd.DataFrame(
                {
                    'Dataset': ["Test"] * len(pc_validation) + ["Validation"] * len(pc_test),
                    'PC':  pc_test + pc_validation

                })

            results.to_csv(Path(final_directory,"test_information_msms.tsv"), sep='\t')
            myplt.plot_msms(final_directory)
            print("Training Images - DONE")

        if flag_rt == "1":
            print("?"+final_directory)

            # Training retention Time
            # Filtering unique sequences by max Andromeda Score
            print ("Antes:")
            print (df_peptides_train)
            print("Columns:")
            print(df_peptides_train.keys())
            print ("Despues")

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
            RTP.epochs = 20
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

            print("?"+final_directory)
            myplt.plot_retention_time(final_directory)
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
    res =[]
    peptides = request.get_json()
    print (peptides)
    if os.path.isfile(Path("models",model_name,"model_y.json")):
        #_MSMS_MODEL = {"model_name": "", "model_y": "", "model_b": ""}
        if (_MSMS_MODEL["model_name"] != model_name):
            xgb_model_loaded_y = xgb.XGBRegressor()
            xgb_model_loaded_y.load_model(Path("models",model_name,"model_y.json"))
            _MSMS_MODEL["model_y"] = xgb_model_loaded_y

            xgb_model_loaded_b = xgb.XGBRegressor()
            xgb_model_loaded_b.load_model(Path("models",model_name,"model_b.json"))
            _MSMS_MODEL["model_b"] = xgb_model_loaded_b

    if os.path.isfile(Path("models", model_name, "model_rt.h5")):
        #K.clear_session()
        #_RT_MODEL = {"model_name": "", "keras_model": ""}
        if (_RT_MODEL["model_name"]!=model_name):
            _RT_MODEL["keras_model"] = keras.models.load_model(Path("models", model_name, "model_rt.h5"),compile=False)
            _RT_MODEL["model_name"]= model_name

    ##########################
    for p in peptides:
        prediction=("","")
        d = dict()
        d["Peptide"] = p["peptide"]
        if os.path.isfile(Path("models", model_name,"model_y.json")):
            prediction = get_prediction(p["peptide"], _MSMS_MODEL["model_y"] , _MSMS_MODEL["model_b"] )
            d["Intensities"] = prediction[0]
            d["Matches"] = prediction[1]
        if os.path.isfile(Path("models", model_name, "model_rt.h5")):
            max_val_rt= 1
            with open(Path("models", model_name, "max_rt.json")) as json_file:
                data = json.load(json_file)
                max_val_rt=float(data["max_rt"])
                print("max value",str(max_val_rt))
            with open(Path("models",model_name,"tokenizer.json")) as f:
                data = json.load(f)
                loaded_tokenizer = tokenizer_from_json(data)
                feature_vector = loaded_tokenizer.texts_to_sequences([p["peptide"]])
                feature_vector = pad_sequences(feature_vector, maxlen=50)
                print("------")
                rt_pred=_RT_MODEL["keras_model"].predict(feature_vector)
            print("Prediction",str(rt_pred[0][0]))
            d["RT"]=str(rt_pred[0][0]*float(max_val_rt))
            res.append(d)
        else:
            d = dict()
            d["Peptide"] = p["peptide"]
            d["Intensities"] = prediction[0]
            d["Matches"] = prediction[1]
            res.append(d)
    print (res)
    return json.dumps(res)


def get_prediction(sequence, xgb_model_loaded_y, xgb_model_loaded_b):
        myTrainingMatrix = wf.createWindowData(sequence, "y1", "1")
        myTrainingMatrix.GenerateMatrix(24)
        d = myTrainingMatrix.GenerateDataset(removeZeros=False)
        y_ions = []
        b_ions = []
        for i in range(0, len(d['target_Y'])):
            y_ions.append(xgb_model_loaded_y.predict(np.array(d['featureMatrix'][i]).reshape(1, -1))[0])
            b_ions.append(xgb_model_loaded_b.predict(np.array(d['featureMatrix'][i]).reshape(1, -1))[0])

        ystr=["y"]*(len(sequence)-1)
        ynum=list(range((len(sequence)-1), 0, -1))
        ions_aux_y = [m + str(n) for m, n in zip(ystr, ynum)]


        bstr = ["b"] * (len(sequence) - 1)
        bnum = list(range(1, len(sequence)))
        ions_aux_b = [m + str(n) for m, n in zip(bstr, bnum)]


        ions=";".join(map(str, ions_aux_y+ions_aux_b))

        return ";".join(map(str, y_ions+b_ions)),ions

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
    #app.run(host="0.0.0.0")
