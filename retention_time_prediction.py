import pandas as pd
import numpy as np
import io
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from tensorflow import keras
from pathlib import Path
global TrainingFile
TrainingFile = ""

class record_losses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        file_training = open(TrainingFile, 'w')
        file_training.write("train_loss\tval_loss\n")
        file_training.close()

    def on_epoch_end(self, epoch, logs={}):
        file_training = open(TrainingFile, 'a')
        file_training.write(str(logs.get('loss')) + "\t" + str(logs.get('val_loss')) + "\n")
        file_training.close()


record_losses = record_losses()


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, metrics, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.metrics = metrics


class RetentionTimePrediction:
    def __init__(self, peptides, retention_times, path):
        self.peptides = peptides
        self.retention_times =retention_times
        self.test_size=0.2
        self.random_state = 42
        self.epochs= 10
        self.batch_size= 40
        self.X_train = np.array([])
        self.X_test = np.array([])
        self.X_validation = np.array([])
        self.Y_train =[]
        self.Y_test =[]
        self.Y_validation = []
        self.tokenizer =""
        self.process_data(path)
        self.model = keras.models.Sequential()

    def get_retentiontime_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(23, 60, input_length=50))
        print("changed to 23")
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(60, return_sequences=True), name='Layer_Bidirectional_1'))
        model.add(keras.layers.LSTM(60, dropout=0.4, recurrent_dropout=0.01, return_sequences=True, name='Layer_LSTM_1'))
        model.add(keras.layers.LSTM(60, dropout=0.4, recurrent_dropout=0.01, name='Layer_LSTM_2'))
        model.add(keras.layers.Dense(20, activation='tanh'))
        model.add(keras.layers.Dense(10, activation='tanh'))
        model.add(keras.layers.Dense(1, activation='linear', name="LastLayer"))
        model.compile(loss='mae', optimizer='adam')
        return model

    def process_data(self,path):

        ####
        global TrainingFile
        TrainingFile = Path(path, "train_information_rt.tsv")
        print(path)
        print(TrainingFile)
        #####

        df = pd.DataFrame(list(zip(self.peptides, self.retention_times)), columns=['sequence', 'retention_time'])
        print(df)
        df_training = df.append(df, ignore_index=True).drop_duplicates()
        self.tokenizer= Tokenizer(num_words=None, char_level=True)
        self.tokenizer.fit_on_texts(df_training['sequence'].values)
        X = self.tokenizer.texts_to_sequences(df_training['sequence'].values)
        X = pad_sequences(X, maxlen=50)
        tokenizer_json = self.tokenizer.to_json()
        with io.open(Path(path,"tokenizer.json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        self.X_train, self.X_test_aux, self.Y_train, self.Y_test_aux = train_test_split(X, df_training['retention_time'],
                                                                    test_size =self.test_size, random_state= self.random_state)

        self.X_test, self.X_validation, self.Y_test, self.Y_validation = train_test_split(self.X_test_aux, self.Y_test_aux, test_size=0.50,
                                                                      random_state= self.random_state)

        X_train_seq, X_test_aux_seq, Y_train_seq, Y_test_aux_seq = train_test_split(df_training['sequence'],
                                                                                        df_training['retention_time'],
                                                                                        test_size=self.test_size,
                                                                                        random_state=self.random_state)

        X_test_seq, X_validation_seq, Y_test_seq, Y_validation_seq = train_test_split(X_test_aux_seq,
                                                                                          self.Y_test_aux,
                                                                                          test_size=0.50,
                                                                                          random_state=self.random_state)

    def train(self):
        self.model = self.get_retentiontime_model()
        metrics_callback = MetricsCallback(validation_data=(self.X_validation, self.Y_validation),
                                           metrics=['mae'])
        self.model.compile(loss='mae', optimizer='adam')
        history = self.model.fit(self.X_train, self.Y_train, epochs=self.epochs,
                               batch_size=self.batch_size, verbose=2,
                            validation_data=(self.X_validation, self.Y_validation),
                            callbacks=[metrics_callback, record_losses])
    def predict(self, peptides):
        print (peptides)
        X = self.tokenizer.texts_to_sequences(peptides)
        X = pad_sequences(X, maxlen=50)
        return self.model.predict(X)

    def predict_feature_vector(self, peptides_fv):
        return self.model.predict(peptides_fv)

    def predict_fromSavedModel(self, peptides, path):
        predictions=[]
        with open(Path(path,"tokenizer.json")) as f:
            data = json.load(f)
            loaded_tokenizer = tokenizer_from_json(data)
            feature_matrix=loaded_tokenizer .texts_to_sequences(peptides)
            feature_matrix = pad_sequences(feature_matrix, maxlen=50)
            reconstructed_model = keras.models.load_model(Path(path,"model_rt.h5"))
            predictions=reconstructed_model.predict(feature_matrix)
        return(predictions)

    def predict_fromSavedModel1(self, peptides, path):
        predictions = []
        with open(Path(path, "tokenizer.json")) as f:
            data = json.load(f)
            loaded_tokenizer = tokenizer_from_json(data)
            feature_matrix = loaded_tokenizer.texts_to_sequences(peptides)
            feature_matrix = pad_sequences(feature_matrix, maxlen=50)
            print(feature_matrix)
            print(feature_matrix.shape)
            reconstructed_model = keras.models.load_model(Path(path, "model_rt.h5"),compile=False)
            predictions = reconstructed_model.predict(feature_matrix)
        return (predictions)

    def save(self, path):
        print("Saving Retention Time model:")
        self.model.save(path)



