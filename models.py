import tensorflow as tf
try :
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    """
    tf.config.set_logical_device_configuration(
       tf.config.list_physical_devices('GPU')[0],
       [tf.config.LogicalDeviceConfiguration(memory_limit=5 * 1024)])
    """
except RuntimeError as e:
    print(e)


from nltk.stem.porter import PorterStemmer
from unicodedata import normalize
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle
import joblib

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout,MaxPooling1D,Conv1D,Flatten,GlobalMaxPooling1D,GlobalAveragePooling1D,Reshape
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model as keras_load_model
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score
import re

class Comment:
    def __init__(self, id, note, dictionary):
        self.note = note
        self.dictionary = dictionary


# Charger les données du fichier texte
def get_dictionary_from_xml_file(file, lexicon):
    
    comments_list = []
    tree = ET.parse(file)
    root = tree.getroot()
    nbcom = len(root.findall("comment"))

    # remplissage du dictionnaire
    with tqdm(total=nbcom, desc=f"Lecture de {file}") as pbar:
        for comment in root.findall('.//comment'):
            text = comment.find("commentaire").text

            if(text is None):
                text = ""

            # creation d'un nouveau commentaire (note + dictionnaire)
            if("test" not in file):
                note = float(comment.find("note").text.replace(",","."))
            else:
                note = 0.0
            comment = normalization(text)
            new_comment = Comment(id, note, count_occurrences(comment))

            comments_list.append(new_comment)
            lexicon.update(word_tokenize(comment))            
            pbar.update(1)

    return comments_list


# retourne un lexique numerote (mot:num_mot)
def numerotation(lexicon):
    i = 1
    lexicon_num = {}
    for word in lexicon:
        lexicon_num[word] = i
        i += 1
    return lexicon_num

# retourne un dictionnaire contenant le nombre d'occurence pour chaque mot
# <mot1>:<nb_mot1>, ...
def count_occurrences(message):

    occurrences = {}
    
    # Tokenisation du message
    msg_words = word_tokenize(message)

    # Pour chaque mot du comment
    for mot in msg_words:
        # Compter le nombre d'occurrences du mot dans le message
        nb_occurrences = msg_words.count(mot)
        if nb_occurrences > 0:
            occurrences[mot] = nb_occurrences
    
    return occurrences

def count_unique_movie_ids():
    unique_movie_ids = set()

    tree = ET.parse("data_final/dev.xml")
    root = tree.getroot()

    for comment in root.findall("comment"):
        movie_id = comment.find("movie").text
        unique_movie_ids.add(movie_id)

    return len(unique_movie_ids)

# renvoie une version normalise du texte en argument
def normalization(text):

    text = text.lower()
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    translator = str.maketrans(string.punctuation,' ' * len(string.punctuation))
    text = text.translate(translator)
    
    words = word_tokenize(text, language='french')
    french_stopwords = set(stopwords.words('french'))
    words = [word for word in words if word not in french_stopwords]
    normalized_text = ' '.join(words)

    return normalized_text

# renvoie le lexique numerote associe au texte en argument
# enregistre le lexique dans le fichier unigram.txt
def get_numbered_lexicon(lexicon):

    # Assigner un numéro pour chaque mot du lexique
    lexicon_num = numerotation(lexicon)
    nb_word = len(lexicon_num)

    # Ecriture du lexique dans un fichier
    with tqdm(total=nb_word, desc=f"Calcul unigrammes") as pbar:
        with open("unigrams.txt", 'w', encoding='utf-8') as output_file:
            for cle, valeur in lexicon_num.items():
                ligne = f"{cle} {valeur}\n"
                output_file.write(ligne)
                pbar.update(1)

    return lexicon_num

# remplace les mots du dictionnaire par leurs equivalents numerique dans le lexique numerote
def replace_word_by_word_number(comments_list, lexicon):

    for comment in comments_list:
        comment.dictionary = {lexicon[key]: value for key, value in comment.dictionary.items()}
        comment.dictionary = dict(sorted(comment.dictionary.items()))

# sauvegarde des dictionnaires dans un fichier
def generate_svm_file(filename, comments_list):
    num = len(comments_list)

    with tqdm(total=num, desc=f"Generation du fichier {filename}") as pbar:
        with open(filename, 'w', encoding='utf-8') as file:
            for comment in comments_list:
                str_note = str(int(comment.note*2) -1)
                file.write(str_note + " ")

                for key, val in comment.dictionary.items():
                    file.write(str(key) + ":" + str(val) + " ")
                file.write('\n')
                pbar.update(1)

    file.close()

# sauvegarde des données dans des fichiers sérialisés
def save_data(data_train, data_val, data_test, lexicon):
    with open('ser/train.pkl', 'wb') as file:
        for item in data_train:
            pickle.dump(item, file)

    with open('ser/val.pkl', 'wb') as file:
        for item in data_val:
            pickle.dump(item, file)

    with open('ser/test.pkl', 'wb') as file:
        for item in data_test:
            pickle.dump(item, file)
    with open('ser/lexicon.pkl', 'wb') as file:
        pickle.dump(lexicon, file)

# récupération des données depuis les fichiers sérialisés
def restore_data():
    data_train = []
    data_val = []
    data_test = []
    lexicon = None

    with open('ser/train.pkl', 'rb') as file:
        while True:
            try:
                item = pickle.load(file)
                data_train.append(item)
            except EOFError:
                break

    with open('ser/val.pkl', 'rb') as file:
        while True:
            try:
                item = pickle.load(file)
                data_val.append(item)
            except EOFError:
                break

    with open('ser/test.pkl', 'rb') as file:
        while True:
            try:
                item = pickle.load(file)
                data_test.append(item)
            except EOFError:
                break

    with open('ser/lexicon.pkl', 'rb') as file:
        lexicon = pickle.load(file)
    
    return data_train, data_val, data_test, lexicon


def xml_to_dataframe(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    data = []
    for comment in root.findall('.//comment'):
        row = {}
        for element in comment:
            row[element.tag] = element.text
        data.append(row)

    df = pd.DataFrame(data)
    return df


def save_model(clf,filepath):
    joblib.dump(clf, filepath) 

def load_model(filepath):
    clf = joblib.load(filepath)
    return clf

lexicon = set()

'''
# Recuperation des donnes de validation et de train
data_train = get_dictionary_from_xml_file('data_final/train.xml', lexicon)
data_val = get_dictionary_from_xml_file('data_final/dev.xml', lexicon)
data_test = get_dictionary_from_xml_file('data_final/test.xml', lexicon)


# recuperation du lexique
lexicon = get_numbered_lexicon(lexicon)

# sauvegarde des données
save_data(data_train, data_val, data_test, lexicon)
'''

# restauration depuis les fichiers
data_train, data_val, data_test, lexicon = restore_data()

'''
# remplacement des mots par leur valeur numerique
replace_word_by_word_number(data_train, lexicon)
replace_word_by_word_number(data_val, lexicon)
replace_word_by_word_number(data_test, lexicon)


#generation des fichiers SVM
generate_svm_file('svm/train.svm', data_train)
generate_svm_file('svm/validation.svm', data_val)
generate_svm_file('svm/test.svm', data_test)
'''

'''
Traitement possible:

- supp ponctuation, caractère spéciaux(ex: \n, \t,...)
- emoji ? (emoji peuvent être utilise pour la classification)
- supp/remplacement liens
- langue étrangère ?
- normalisation du texte ?
- supp mot vide
- completion de donnés(via AlloCine) => equilibrer les classes ?

mot plus frequent par note

'''

def dense_classification(train_data,val_data,test_data):
        
    # Tokenizez les commentaires
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_data['commentaire'])
    tokenizer.fit_on_texts(val_data['commentaire'])
    tokenizer.fit_on_texts(test_data['commentaire'])
    
    # Convertissez les commentaires en séquences
    train_sequences = tokenizer.texts_to_sequences(train_data['commentaire'])
    val_sequences = tokenizer.texts_to_sequences(val_data['commentaire'])
    test_sequences = tokenizer.texts_to_sequences(test_data['commentaire'])
    
    # Remplissez les séquences pour qu'elles aient toutes la même longueur
    max_len = 100
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    val_sequences = pad_sequences(val_sequences, maxlen=max_len, padding='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')
    
    
    #Classification
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=300, input_length=max_len))
    #model.add(Flatten())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Utiliser 'softmax' pour la classification multiclasse
    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    
    train_data['note'] = train_data['note'].str.replace(',', '.')
    val_data['note'] = val_data['note'].str.replace(',', '.')
    
    y_train = train_data['note'].astype(float)
    y_val = val_data['note'].astype(float)
    
    y_train_encoded = ((y_train - 0.5) * 2).astype(int)
    y_val_encoded = ((y_val - 0.5) * 2).astype(int)
    
    print(y_train_encoded)
    
    
    checkpoint = ModelCheckpoint("dense_classification.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    model.fit(train_sequences, y_train_encoded, epochs=10, batch_size=256, validation_data=(val_sequences, y_val_encoded),callbacks=[checkpoint])
    

    best_model = keras_load_model("dense_classification.h5")
    predictions = best_model.predict(test_sequences)
    predicted_notes = predictions.argmax(axis=1)
    
    print(predicted_notes)
    
    with open('out.txt', 'w') as output_file:
        for review_id, note in zip(test_data["review_id"], predicted_notes):
            # Supprimer les espaces blancs au début et à la fin de la note
            note = str(((float(note) / 2 )+0.5)).replace(".",",")
            # Écrire le review_id suivi de la note dans le nouveau fichier
            output_file.write(f"{review_id} {note}\n")

def cnn_classification(train_data,val_data,test_data):
            
    # Tokenizez les commentaires
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_data['commentaire'])
    tokenizer.fit_on_texts(val_data['commentaire'])
    tokenizer.fit_on_texts(test_data['commentaire'])
    
    # Convertissez les commentaires en séquences
    train_sequences = tokenizer.texts_to_sequences(train_data['commentaire'])
    val_sequences = tokenizer.texts_to_sequences(val_data['commentaire'])
    test_sequences = tokenizer.texts_to_sequences(test_data['commentaire'])
    
    # Remplissez les séquences pour qu'elles aient toutes la même longueur
    max_len = 100
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    val_sequences = pad_sequences(val_sequences, maxlen=max_len, padding='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')
    
    
    #Classification
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Utiliser 'softmax' pour la classification multiclasse
    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    
    train_data['note'] = train_data['note'].str.replace(',', '.')
    val_data['note'] = val_data['note'].str.replace(',', '.')
    
    y_train = train_data['note'].astype(float)
    y_val = val_data['note'].astype(float)
    
    y_train_encoded = ((y_train - 0.5) * 2).astype(int)
    y_val_encoded = ((y_val - 0.5) * 2).astype(int)
    
    print(y_train_encoded)
    
    
    checkpoint = ModelCheckpoint("cnn_classification.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    model.fit(train_sequences, y_train_encoded, epochs=10, batch_size=256, validation_data=(val_sequences, y_val_encoded),callbacks=[checkpoint])
    

    best_model = keras_load_model("cnn_classification.h5")
    predictions = best_model.predict(test_sequences)
    predicted_notes = predictions.argmax(axis=1)
    
    print(predicted_notes)
    
    with open('out.txt', 'w') as output_file:
        for review_id, note in zip(test_data["review_id"], predicted_notes):
            # Supprimer les espaces blancs au début et à la fin de la note
            note = str(((float(note) / 2 )+0.5)).replace(".",",")
            # Écrire le review_id suivi de la note dans le nouveau fichier
            output_file.write(f"{review_id} {note}\n")

def logistic_regression(train_data,val_data,test_data):
    
    cv = CountVectorizer(binary=True)
    cv.fit(train_data["commentaire"])
    
    train_onehot = cv.transform(train_data["commentaire"])
    val_onehot = cv.transform(val_data["commentaire"])
    test_onehot = cv.transform(test_data["commentaire"])


    train_data['note'] = train_data['note'].str.replace(',', '.')
    val_data['note'] = val_data['note'].str.replace(',', '.')

    y_train = train_data['note']
    y_val = val_data['note']


    lr = LogisticRegression(C=0.01)
    lr.fit(train_onehot, y_train)

    save_model(lr, "logistic_regression.pkl") 
    best_model = load_model("logistic_regression.pkl")
    predictions = best_model.predict(test_onehot)
    predicted_notes = predictions

    print(predicted_notes)

    with open('out.txt', 'w') as output_file:
        for review_id, note in zip(test_data["review_id"], predicted_notes):
            # Supprimer les espaces blancs au début et à la fin de la note
            note = str((float(note))).replace(".",",")
            # Écrire le review_id suivi de la note dans le nouveau fichier
            output_file.write(f"{review_id} {note}\n")

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


STOPWORDS = set(stopwords.words('french'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

#removing stopwords and word processing

def word_replace(text):
    return text.replace('','')

stemmer = PorterStemmer()

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    '''
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])
    '''
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
def preprocess(text):
    text=text.lower()
    text=word_replace(text)
    text=remove_urls(text)
    text=remove_html(text)
    text=remove_stopwords(text)
    text=remove_punctuation(text)
    text=lemmatize_words(text)
    return text

def cnn_classification2(train_data,val_data,test_data):
    
    '''
    train_data["commentaire"] = train_data["commentaire"].apply(lambda text: preprocess(text))
    val_data["commentaire"] = val_data["commentaire"].apply(lambda text: preprocess(text))
    test_data["commentaire"] = test_data["commentaire"].apply(lambda text: preprocess(text))
    '''
    from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

    sample_size = 25000
    notes = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"]
    
    train_data['note'] = train_data['note'].str.replace(',', '.')
    val_data['note'] = val_data['note'].str.replace(',', '.')
    
    subsamples = []
    for note in notes:
        subset = train_data[train_data['note'] == note]
        subsample = subset.sample(n=sample_size, random_state=42)
        subsamples.append(subsample)
    
    train_sampled = pd.concat(subsamples)
    
    print(train_sampled["note"].isna().sum())
    '''
    SIA = SentimentIntensityAnalyzer()
    
    temp_train_sentiments = [] 
    print("train begin")
    for i, comment in enumerate(train_sampled["commentaire"]):
        print("train",i)
        score = SIA.polarity_scores(comment)
        if score['compound'] <= -0.05:
            train_sampled.at[i, 'sentiment'] = 0
        elif score['compound'] >= 0.05:
            train_sampled.at[i, 'sentiment'] = 1
        else:
            train_sampled.at[i, 'sentiment'] = 2
    
    print("val begin")
    for i, comment in enumerate(val_data["commentaire"]):
        #print("val",i)
        score = SIA.polarity_scores(comment)
        if score['compound'] <= -0.05:
            val_data.at[i, 'sentiment'] = 0
        elif score['compound'] >= 0.05:
            val_data.at[i, 'sentiment'] = 1
        else:
            val_data.at[i, 'sentiment'] = 2
    
    print("test begin")
    for i, comment in enumerate(test_data["commentaire"]):
        #print("test",i)
        score = SIA.polarity_scores(comment)
        if score['compound'] <= -0.05:
            test_data.at[i, 'sentiment'] = 0
        elif score['compound'] >= 0.05:
            test_data.at[i, 'sentiment'] = 1
        else:
            test_data.at[i, 'sentiment'] = 2

    train_sampled['commentaire'] = train_sampled.apply(lambda row: str(row['sentiment']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1)    
    val_data['commentaire'] = val_data.apply(lambda row: str(row['sentiment']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1)    
    test_data['commentaire'] = test_data.apply(lambda row: str(row['sentiment']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1) 
    '''
    
    train_sampled['commentaire'] = train_sampled.apply(lambda row: str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1)    
    val_data['commentaire'] = val_data.apply(lambda row: str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1)    
    test_data['commentaire'] = test_data.apply(lambda row: str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1) 
    
    # Tokenizez les commentaires
    max_words = 5000
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_sampled['commentaire'])
    tokenizer.fit_on_texts(val_data['commentaire'])
    tokenizer.fit_on_texts(test_data['commentaire'])
    
    # Convertissez les commentaires en séquences
    train_sequences = tokenizer.texts_to_sequences(train_sampled['commentaire'])
    val_sequences = tokenizer.texts_to_sequences(val_data['commentaire'])
    test_sequences = tokenizer.texts_to_sequences(test_data['commentaire'])
    
    # Remplissez les séquences pour qu'elles aient toutes la même longueur
    max_len = 512 #150?
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    val_sequences = pad_sequences(val_sequences, maxlen=max_len, padding='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')
    
    embedding_dim = 32
    vocab_size = len(tokenizer.word_index) + 1
    
    #Classification
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(10, activation='softmax'))

    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    
 
    print(train_sampled["note"].isna().sum())
    train_sampled['note'] = train_sampled['note'].fillna('0.5')
    val_data['note'] = val_data['note'].fillna('0.5')
    
    y_train = train_sampled['note'].astype(float)
    y_val = val_data['note'].astype(float)
    
    y_train_encoded = ((y_train - 0.5) * 2).astype(int)
    y_val_encoded = ((y_val - 0.5) * 2).astype(int)
    
    print(y_train_encoded)
    
    checkpoint = ModelCheckpoint("cnn_classification.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    with tf.device('/GPU:0'):
        model.fit(train_sequences, y_train_encoded, epochs=5, batch_size=32, validation_data=(val_sequences, y_val_encoded),callbacks=[checkpoint])
    

    best_model = keras_load_model("cnn_classification.h5")
    predictions = best_model.predict(test_sequences)
    predicted_notes = predictions.argmax(axis=1)
    
    print(predicted_notes)
    
    with open('out.txt', 'w') as output_file:
        for review_id, note in zip(test_data["review_id"], predicted_notes):
            # Supprimer les espaces blancs au début et à la fin de la note
            note = str(((float(note) / 2 )+0.5)).replace(".",",")
            # Écrire le review_id suivi de la note dans le nouveau fichier
            output_file.write(f"{review_id} {note}\n")


def encode_reviews(tokenizer, reviews, max_length):
    token_ids = np.zeros(shape=(len(reviews), max_length),
                         dtype=np.int32)
    for i, review in enumerate(reviews):
        encoded = tokenizer.encode(review, max_length=max_length,truncation=True)
        token_ids[i, 0:len(encoded)] = encoded
    attention_mask = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_mask": attention_mask}
    #return token_ids


def encode_reviews_with_sentiments(tokenizer, reviews, sentiments, max_length):
    token_ids = np.zeros(shape=(len(reviews), max_length), dtype=np.int32)
    for i, (review, sentiment) in enumerate(zip(reviews, sentiments)):
        # Encoder le commentaire textuel
        encoded = tokenizer.encode(review, max_length=max_length, truncation=True)
        token_ids[i, 0:len(encoded)] = encoded

        # Ajouter l'information de sentiment à la fin de la séquence
        token_ids[i, len(encoded):] = sentiment

    attention_mask = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_mask": attention_mask}


def bert(train_data, val_data, test_data):
		
    from transformers import AutoTokenizer,TFCamembertModel,AutoModelForSequenceClassification,BertForSequenceClassification, TFAutoModelForSequenceClassification,CamembertTokenizer
    from transformers import pipeline
    import torch.nn as nn
    from tensorflow.keras.optimizers import Adam
    import random
    
    from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
    
    #pretrained_model = TFCamembertModel.from_pretrained("jplu/tf-camembert-base")
    #print(pretrained_model)
    
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
      
    '''
    auto_tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
    model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=auto_tokenizer)
    
    for i, comment in enumerate(train_data["commentaire"]):
        print(i)
        result = nlp(comment)[0]['label']
        if result == "POSITIVE":
           train_data.at[i, 'sentiment'] = 1
        elif result == "NEGATIVE":
           train_data.at[i, 'sentiment'] = 0
    
    
    '''


    with tf.device('/CPU:0'):

        tokenizer_base = CamembertTokenizer.from_pretrained("camembert-base")
        auto_tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
        
        
        sample_size = 25000
        notes = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"]
        
        train_data['note'] = train_data['note'].str.replace(',', '.')
        val_data['note'] = val_data['note'].str.replace(',', '.')
        
        subsamples = []
        for note in notes:
            subset = train_data[train_data['note'] == note]
            subsample = subset.sample(n=sample_size, random_state=42)
            subsamples.append(subsample)
        
        train_sampled = pd.concat(subsamples)
        
        print("train_sampled note = ",train_sampled["note"].isna().sum())
        
        y_train = train_sampled['note'].astype(float)
        y_val = val_data['note'].astype(float)
        
        y_train_encoded = ((y_train - 0.5) * 2).astype(int)
        y_val_encoded = ((y_val - 0.5) * 2).astype(int)
        
        '''
        for index, row in train_data.iterrows():
            comment = normalization(row['commentaire'])
            train_data.at[index, 'commentaire'] = comment
        
        for index, row in val_data.iterrows():
            comment = normalization(row['commentaire'])
            val_data.at[index, 'commentaire'] = comment
            
        for index, row in test_data.iterrows():
            comment = normalization(row['commentaire'])
            test_data.at[index, 'commentaire'] = comment
        '''
        
        print(train_data.head())
        
        SIA = SentimentIntensityAnalyzer()
        print("train_sampled note = ",train_sampled["note"].isna().sum())
        
        print(len(train_sampled["commentaire"]))
        print(len(val_data["commentaire"]))
        print(len(test_data["commentaire"]))
        
        '''
        print("train begin")
        for i, comment in enumerate(train_sampled["commentaire"]):
            #print("train",i)
            score = SIA.polarity_scores(comment)
            if score['compound'] <= -0.05:
                train_sampled.at[i, 'sentiment'] = "Negative"
            elif score['compound'] >= 0.05:
                train_sampled.at[i, 'sentiment'] = "Positive"
            else:
                train_sampled.at[i, 'sentiment'] = "Neutral"
        
        print("val begin")
        for i, comment in enumerate(val_data["commentaire"]):
            #print("val",i)
            score = SIA.polarity_scores(comment)
            if score['compound'] <= -0.05:
                val_data.at[i, 'sentiment'] = "Negative"
            elif score['compound'] >= 0.05:
                val_data.at[i, 'sentiment'] = "Positive"
            else:
                val_data.at[i, 'sentiment'] = "Neutral"
        
        print("test begin")
        for i, comment in enumerate(test_data["commentaire"]):
            #print("test",i)
            score = SIA.polarity_scores(comment)
            if score['compound'] <= -0.05:
                test_data.at[i, 'sentiment'] = "Negative"
            elif score['compound'] >= 0.05:
                test_data.at[i, 'sentiment'] = "Positive"
            else:
                test_data.at[i, 'sentiment'] = "Neutral"
        
        
        print("train_sampled note = ",train_sampled["note"].isna().sum())
        
        train_sampled['commentaire'] = train_sampled.apply(lambda row: str(row['sentiment']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1)    
        val_data['commentaire'] = val_data.apply(lambda row: str(row['sentiment']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1)    
        test_data['commentaire'] = test_data.apply(lambda row: str(row['sentiment']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']) + ' ' + str(row['commentaire']),axis=1) 
        
        print("train_sampled note = ",train_sampled["note"].isna().sum())
        
        '''
        train_sampled['commentaire'] = train_sampled.apply(lambda row: str(row['commentaire']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']),axis=1)    
        val_data['commentaire'] = val_data.apply(lambda row: str(row['commentaire']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']),axis=1)    
        test_data['commentaire'] = test_data.apply(lambda row: str(row['commentaire']) + ' ' + str(row['genres']) + ' ' + str(row['nb_notes']) + ' ' + str(row['synopsis']),axis=1) 
        
        train_reviews = np.array(train_sampled["commentaire"])
        val_reviews = np.array(val_data["commentaire"])
        test_reviews = np.array(test_data["commentaire"])
        


        print(train_reviews.size)
        print(val_reviews.size)
        print(test_reviews.size)
    
        
        #train_sampled_indices = random.sample(range(len(train_reviews)), 150_000)
        #val_sampled_indices = random.sample(range(len(val_reviews)), 150_000)
        
    
        #train_reviews = train_reviews[train_sampled_indices]
        #y_train_encoded = y_train_encoded.iloc[train_sampled_indices]
        
        '''
        val_reviews = val_reviews[val_sampled_indices]
        y_val_encoded = y_val_encoded.iloc[val_sampled_indices]
        '''
        
        
        '''
        train_sentiments = np.array(temp_train_sentiments)
        val_sentiments = np.array(val_data["sentiment"])
        test_sentiments = np.array(test_data["sentiment"])
        '''
        MAX_SEQ_LEN = 150#200 = 20% non concerné
        
        encoded_train = encode_reviews(auto_tokenizer, train_reviews, MAX_SEQ_LEN)
        encoded_val = encode_reviews(auto_tokenizer, val_reviews, MAX_SEQ_LEN)
        encoded_test = encode_reviews(auto_tokenizer, test_reviews, MAX_SEQ_LEN)
        
        
        '''
        encoded_train = encode_reviews_with_sentiments(auto_tokenizer, train_reviews, train_sentiments, MAX_SEQ_LEN)
        encoded_val = encode_reviews_with_sentiments(auto_tokenizer, val_reviews, val_sentiments, MAX_SEQ_LEN)
        encoded_test = encode_reviews_with_sentiments(auto_tokenizer, test_reviews,test_sentiments, MAX_SEQ_LEN)
        '''
        
        '''
        reviews_len = [len(auto_tokenizer.encode(review, max_length=512))
                          for review in train_reviews]
        print("Average length: {:.1f}".format(np.mean(reviews_len)))
        print("Max length: {}".format(max(reviews_len)))
        
        short_reviews = sum(np.array(reviews_len) <= MAX_SEQ_LEN)
        long_reviews = sum(np.array(reviews_len) > MAX_SEQ_LEN)
        
        print("{} reviews with LEN > {} ({:.2f} % of total data)".format(
            long_reviews,
            MAX_SEQ_LEN,
            100 * long_reviews / len(reviews_len)
        ))
        
        count_0_5 = (train_data['note'] == "0.5").sum()
        count_1_0 = (train_data['note'] == "1.0").sum()
        count_1_5 = (train_data['note'] == "1.5").sum()
        count_2_0 = (train_data['note'] == "2.0").sum()
        count_2_5 = (train_data['note'] == "2.5").sum()
        count_3_0 = (train_data['note'] == "3.0").sum()
        count_3_5 = (train_data['note'] == "3.5").sum()
        count_4_0 = (train_data['note'] == "4.0").sum()
        count_4_5 = (train_data['note'] == "4.5").sum()
        count_5_0 = (train_data['note'] == "5.0").sum()
        print("Count 0.5:", count_0_5)
        print("Count 1.0:", count_1_0)
        print("Count 1.5:", count_1_5)
        print("Count 2.0:", count_2_0)
        print("Count 2.5:", count_2_5)
        print("Count 3.0:", count_3_0)
        print("Count 3.5:", count_3_5)
        print("Count 4.0:", count_4_0)
        print("Count 4.5:", count_4_5)
        print("Count 5.0:", count_5_0)
        '''
        
        #pretrained_model = TFCamembertModel.from_pretrained("jplu/tf-camembert-base")
        pretrained_model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
        pretrained_model.classifier.out_proj = tf.keras.layers.Dense(units=10, activation='softmax')
               
        #opt = Adam(learning_rate=0.001)
        opt = Adam(learning_rate=5e-6, epsilon=1e-08)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()    
        pretrained_model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
        pretrained_model.summary()
    
    #pretrained_model.fit(encoded_train,y_train_encoded, epochs=5, batch_size=32, validation_data=(encoded_val,y_val_encoded))
    
    
    with tf.device('/GPU:0'):
        pretrained_model.fit(encoded_train,y_train_encoded, epochs=5, batch_size=32, validation_data=(encoded_val,y_val_encoded))
       #pretrained_model.fit(encoded_val,y_val_encoded, epochs=5, batch_size=32)
    
    
    #save_model(pretrained_model, "bert_model/bert")
    #best_model = keras_load_model("bert_model/bert")
    #pretrained_model.save_model("bert_model/bert")
    #best_model = TFAutoModelForSequenceClassification.from_pretrained("bert_model/bert")
   
    best_model = pretrained_model
    predictions = best_model.predict(encoded_test)    
    
         
    predicted_notes = np.argmax(predictions.logits, axis=1)#tf.argmax(predictions.logits, axis=1).numpy()
    print(predicted_notes)
   
    with open('out.txt', 'w') as output_file:
        for review_id, note in zip(test_data["review_id"], predicted_notes):
       			# Supprimer les espaces blancs au début et à la fin de la note
                note = str(((float(note) / 2 )+0.5)).replace(".",",")
       			# Écrire le review_id suivi de la note dans le nouveau fichier
                output_file.write(f"{review_id} {note}\n")
    
	
with tf.device('/CPU:0'):
    train_data = xml_to_dataframe("data_final/train.xml")
    val_data = xml_to_dataframe("data_final/dev.xml")
    test_data = xml_to_dataframe("data_final/test.xml")
    
    columns_to_drop = ['movie', 'review_id', 'user_id']
    
    train_data.drop(columns_to_drop,axis=1)
    val_data.drop(columns_to_drop,axis=1)
    test_data.drop(columns_to_drop,axis=1)
    
    train_data = xml_to_dataframe("scrap/outputs/train_out.xml")
    val_data = xml_to_dataframe("scrap/outputs/val_out.xml")
    test_data = xml_to_dataframe("scrap/outputs/test_out.xml")
    
    train_data['commentaire'] = train_data['commentaire'].fillna('Aucun commentaire')
    val_data['commentaire'] = val_data['commentaire'].fillna('Aucun commentaire')
    test_data['commentaire'] = test_data['commentaire'].fillna('Aucun commentaire')

    train_data['synopsis'] = train_data['synopsis'].fillna('Aucun synopsis')
    val_data['synopsis'] = val_data['synopsis'].fillna('Aucun synopsis')
    test_data['synopsis'] = test_data['synopsis'].fillna('Aucun synopsis')
    
    train_data['genres'] = train_data['genres'].fillna('Aucun genres')
    val_data['genres'] = val_data['genres'].fillna('Aucun genres')
    test_data['genres'] = test_data['genres'].fillna('Aucun genres')
    
    train_data['nb_notes'] = train_data['nb_notes'].fillna('Aucun nb_notes')
    val_data['nb_notes'] = val_data['nb_notes'].fillna('Aucun nb_notes')
    test_data['nb_notes'] = test_data['nb_notes'].fillna('Aucun nb_notes')
    
    print(train_data.head())
    
#dense_classification(train_data, val_data, test_data)
#cnn_classification2(train_data, val_data, test_data)
#logistic_regression(train_data, val_data, test_data)
bert(train_data, val_data, test_data)

'''
Traitement possible:

- supp ponctuation, caractère spéciaux(ex: \n, \t,...)
- emoji ? (emoji peuvent être utilise pour la classification)
- supp/remplacement liens
- langue étrangère ?
- normalisation du texte ?
- supp mot vide
- completion de donnés(via AlloCine) => equilibrer les classes ?

mot plus frequent par note

'''

