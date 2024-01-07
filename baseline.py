from unicodedata import normalize
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle
import random

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
                note = 0
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

# réduction de la taille des donnée
def reduce_data(data):
    half_size = len(data) // 2
    reduced_data = random.sample(data, half_size)

    return reduced_data


print(normalization("https://m.facebook.com/La-7eme-critique-393816544123997"))

'''
lexicon = set()
# Recuperation des donnes de validation et de train
data_train = get_dictionary_from_xml_file('data_final/train.xml', lexicon)
data_val = get_dictionary_from_xml_file('data_final/dev.xml', lexicon)
data_test = get_dictionary_from_xml_file('data_final/test.xml', lexicon)

# recuperation du lexique
lexicon = get_numbered_lexicon(lexicon)

# sauvegarde des données
save_data(data_train, data_val, data_test, lexicon)


# restauration depuis les fichiers
data_train, data_val, data_test, lexicon = restore_data()


# remplacement des mots par leur valeur numerique
replace_word_by_word_number(data_train, lexicon)
replace_word_by_word_number(data_val, lexicon)
replace_word_by_word_number(data_test, lexicon)

# reduction des données
# data_train = reduce_data(data_train)

# generation des fichiers SVM
generate_svm_file('svm/train.svm', data_train)
generate_svm_file('svm/validation.svm', data_val)
generate_svm_file('svm/test.svm', data_test)


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

