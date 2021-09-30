import json
import os.path
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from pathlib import Path

model_name="rttest"
peptides=["*S*LVSDFR|*DWDPK","*NLIGALLFDSGETAEATR|GY*EVK"]

_RT_MODEL={"model_name":"", "keras_model":""}
_RT_MODEL["keras_model"] = keras.models.load_model(Path("models", model_name, "model_rt.h5"),compile=False)
_RT_MODEL["model_name"]= model_name

res=[]
for p in peptides:
    prediction = ("", "")
    d = dict()
    d["Peptide"] = p

    if os.path.isfile(Path("models", model_name, "model_rt.h5")):
        max_val_rt = 1
        with open(Path("models", model_name, "max_rt.json")) as json_file:
            data = json.load(json_file)
            max_val_rt = float(data["max_rt"])
            print("max value", str(max_val_rt))
        with open(Path("models", model_name, "tokenizer.json")) as f:
            data = json.load(f)
            loaded_tokenizer = tokenizer_from_json(data)
            feature_vector = loaded_tokenizer.texts_to_sequences([p])
            print (p)
            feature_vector = pad_sequences(feature_vector, maxlen=50)
            print("------")
            print (feature_vector.shape)
            rt_pred = _RT_MODEL["keras_model"].predict(feature_vector)
            print (rt_pred)
        print("Prediction", str(rt_pred[0][0]))
        d["RT"] = str(rt_pred[0][0] * float(max_val_rt))
        res.append(d)
    else:
        d = dict()
        d["Peptide"] = p["peptide"]
        d["Intensities"] = prediction[0]
        d["Matches"] = prediction[1]
        res.append(d)