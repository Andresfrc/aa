import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random 
import pickle


model=tf.keras.models.load_model('chatbot.keras')
with open('tokenizer.pickle')as f:
    tokenizer=pickle.load(f)
with open('labels_encoders.pickle','rb') as f:
    labels_encoder=pickle.load(f)
with open('responses.json','r',encoding='utf-8')as f:
    responses=json.load(f)
with open('model_config.json','r')as f:
    config=json.load(f)
    max_length=config['max_length']    
    
def predecir(texto):
    sequence=tokenizer.texts_to_sequences([texto.lower()])
    padded=pad_sequences(sequence,maxlen=max_length)
    prediction=model.predict(padded,verbose=0)
    predicted_class=np.argmax(prediction[0]) 
    confidence=prediction[0][predicted_class]
    intent=labels_encoder.inverse_transform([predicted_class])[0]   
    respuesta=random.choice(responses[intent])
    return intent,confidence,respuesta

while True:
    texto=input("\n👤")
    if texto.lower()in ['salir','exit','quit']:
        break
    intent,conf,resp=predecir(texto)
    print(f"🤖:{resp}")







import json 
import torch
from transformers import pipeline

class ChatbotCarnet:
    def __init__(self,
                 dataset_path="dataset_carnet_updated.json",
                 modelo_encharcer_path="./modelo_finetuned"):
        pass

    
