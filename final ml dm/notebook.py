#!/usr/bin/env python
# coding: utf-8

# ### All what you need to apply Named Entity Recognition (NER) 

# **Named Entity Recognition:** is the task of identifying and categorizing key information (entities) in text. An entity can be any word or series of words that consistently refers to the same thing.

# Find the Aggregated dataset at: https://www.kaggle.com/naseralqaydeh/named-entity-recognition-ner-corpus

# # 1- Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# # 2- Importing Data 

# In[59]:


data_path = "train_data.csv"

data = pd.read_csv(data_path, sep=';',  header=None, skip_blank_lines=False)
# filling the first column that determines which sentence each word belongs to.
data.fillna(method = 'ffill', inplace = True)
data.rename(columns = {0:'data'}, inplace = True)
data.head()


# In[61]:


import pandas as pd

# file_path = 'drive/MyDrive/Colab Notebooks/compete data/train_gt.csv'
file_path = 'train_gt.csv'

label = pd.read_csv(file_path, header=None, names=['Labels'], skip_blank_lines=False)
data['label'] = label
data = data[['data', 'label']]

data['data'] = data['data'].fillna('')
data['label'] = data['label'].fillna('')

data


# In[ ]:


# final_data['data'] = final_data['data'].fillna('')
# final_data['label'] = final_data['label'].fillna('')


# In[ ]:





# In[ ]:





# ## Ready to create new data

# In[6]:


from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_data(gt_path, data_path):
	with open(data_path) as f:
		data = f.read().splitlines()

	with open(gt_path, "r") as f:
		labels = f.read().splitlines()

	df = pd.DataFrame({"text": data, "label": labels})
	df = df[df["text"] != ";;;"]
	df["text"] = df["text"].apply(lambda x: x.replace(";;;", ""))
	df = df[~(df["label"].str.strip()=="")]
	df = df[~df["label"].str.contains(";")]

	df["label"] = df["label"].str.strip()

	df["label"] = np.where(df["label"] == "O O", "O", df["label"])
	return df

train = load_data("train_gt.csv", "train_data.csv")
valid = load_data("valid_gt.csv", "valid_data.csv")
# label_mapping = {'O': 0, 'B-ORG': 1, 'B-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-ORG': 6, 'I-MISC': 7, 'I-LOC': 8}
# train['label'] = train['label'].map(label_mapping)
# valid['label'] = valid['label'].map(label_mapping)
# print(len(valid))
# print(train['label'].unique())
# print(valid['label'].unique())
ner_pos = LabelEncoder()
def rows_to_sentences_and_labels(df):
    sentences = []
    sentences_labels = []
    current_sentence = []
    current_labels = []

    for index, row in tqdm(df.iterrows(), total = len(df)):
        word, label = row['text'], row['label']
        current_sentence.append(word.strip())
        current_labels.append(label)
        if word.strip() == '.':
            sentences.append(current_sentence)
            sentences_labels.append(current_labels)
            current_sentence = []
            current_labels = []

    return sentences, sentences_labels

train_sentences, train_sentences_labels = rows_to_sentences_and_labels(train)
valid_sentences, valid_sentences_labels = rows_to_sentences_and_labels(valid)


# In[7]:


train_sentences[:5]


# In[8]:


train_sentences_labels[:5]


# In[ ]:





# In[23]:


import pandas as pd

# Create a DataFrame
ready_data = pd.DataFrame({'data': train_sentences, 'label': train_sentences_labels})

# Print or use the DataFrame as needed
ready_data['sentence'] = ready_data['data'].apply(lambda x: ' '.join(x))

# ready_data['label'] = ready_data['label'].apply(lambda x: (t = '"' + t + '"') for t in x)
df['label'] = df['label'].apply(lambda x: [str(i) for i in x])

ready_data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# ready_dist_path = "../input/named-entity-recognition-ner-corpus/ner.csv"
# ready_data = pd.read_csv(ready_dist_path)
# ready_data.head()


# # 3- Get to know our data

# In[ ]:


# def join_a_sentence(sentence_number):

#     """
#     Args.:
#           sentence_number: sentence number we want to join and return. 
          
#     Returns:
#           The joined sentence.
#     """
    
#     sentence_number = str(sentence_number)
#     the_sentence_words_list = list(data[data['Sentence #'] == 'Sentence: {}'.format(sentence_number)]['Word'])
    
#     return ' '.join(the_sentence_words_list)


# In[ ]:


# join_a_sentence(sentence_number = 1)


# In[ ]:


# join_a_sentence(sentence_number = 100)


# In[14]:


# Data Shape
data.shape


# In[ ]:


# Number of unique sentences
# len(np.unique(data['Sentence #']))


# In[ ]:


# print("Number of unique words in the dataset: {}".format(data.Word.nunique()))
# print("Number of unique tags in the dataset: {}".format(data.Tag.nunique()))


# In[62]:


tags = data.label.unique()
tags


# In[ ]:


# def num_words_tags (tags, data):
    
#     """This functions takes the tags we want to count and the datafram 
#     and return a dict where the key is the tag and the value is the frequency
#     of that tag"""
    
#     tags_count = {}
    
#     for tag in tags:
#         len_tag = len(data[data['Tag'] == tag])
#         tags_count[tag] = len_tag
    
#     return tags_count


# In[ ]:


# tags_count = num_words_tags(tags, data)
# tags_count


# In[ ]:


# plt.figure(figsize = (10, 6))
# plt.hist(data.Tag, log = True, label = 'Tags', color = 'olive', bins = 50)
# plt.xlabel('Tags', fontsize = 16)
# plt.ylabel('Count', fontsize = 16)
# plt.title("Tags Frequency", fontsize = 20)
# plt.grid(alpha=0.3)
# plt.legend()
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xticks(rotation=90)
# plt.show()


# **Code that I used to produce  ready_data** 

# In[ ]:


# def process_Data():
    
#     data_dict = {}
    
#     for sn in range(1, len(np.unique(data['Sentence #']))+1):
        
#         all_sen_data = []
        
#         se_data = data[data['Sentence #']  == 'Sentence: {}'.format(sn)]
#         sentence = ' '.join(list(se_data['Word']))
#         all_sen_data.append(sentence)
        
#         sen_pos = list(se_data['POS'])
#         all_sen_data.append(sen_pos)
        
#         sen_tags = list(se_data['Tag'])
#         all_sen_data.append(sen_tags)
        
#         data_dict['Sentence: {}'.format(sn)] = all_sen_data
        
#         if sn % 10000 == 0:
#             print("{} sentences are processed".format(sn))
        
#     return data_dict


# # 4- Data Preprocessing

# In[24]:


ready_data.head()


# In[25]:


X = list(ready_data['sentence'])
Y = list(ready_data['label'])


# In[29]:


# from ast import literal_eval
# Y_ready = []

# for sen_tags in Y:
#     Y_ready.append(literal_eval(sen_tags))

Y_ready = Y


# In[30]:


# import json

# Y_ready = []

# for sen_tags in Y:
#     Y_ready.append(json.loads(sen_tags))

# Now Y_ready contains lists instead of string representations of lists


# In[31]:


print("First three sentences: \n")
print(X[:3])


# In[32]:


print("First three Tags: \n")
print(Y_ready[:3])


# We need to tokenize the sentences by mapping each word to a unique identifier, then we need to pad them because NN need the input sentences to have the same lenght.

# In[33]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[34]:


print("Number of examples: {}".format(len(X)))


# - **Toknize sentences**

# In[126]:


# cutoff reviews after 110 words
maxlen = 256

# consider the top 36000 words in the dataset
max_words = 36000

# tokenize each sentence in the dataset
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)


# In[203]:


X[1]


# In[202]:


sequences[1]


# In[127]:


len(sequences)


# In[128]:


word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))
ind2word = dict([(value, key) for (key, value) in word_index.items()])


# In[129]:


# ind2word


# In[130]:


word2id = word_index


# In[131]:


# dict. that map each identifier to its word
id2word = {}
for key, value in word2id.items():
    id2word[value] = key


# In[132]:


# id2word


# - **Sentences padding**

# In[133]:


# pad the sequences so that all sequences are of the same size
X_preprocessed = pad_sequences(sequences, maxlen=maxlen, padding='post')


# In[134]:


# first example after tokenization and padding. 
X_preprocessed[0]


# In[135]:


# 22479 example after tokenization and padding. 
X_preprocessed[7373]


# - **Preprocess tags**

# we need to preprocess tags by assigning a unique identifier for each one of them. 
# 
# Since also tags for each example have different lenght we need to fine a way to slove this problem.  
# 
# - We can assign a new tag for the zeros that we used in padding 
# - We can use the O tag for them. 
# 
# I will try the second choice of using the O tag to pad the tag list. 

# In[136]:


# dict. that map each tag to its identifier
tags2id = {}
for i, tag in enumerate(tags):
    tags2id[tag] = i


# In[137]:


tags2id


# In[138]:


# dict. that map each identifier to its tag
id2tag = {}
for key, value in tags2id.items():
    id2tag[value] = key


# In[139]:


id2tag


# In[140]:


def preprocess_tags(tags2id, Y_ready):
    
    Y_preprocessed = []
    maxlen = 256
    # for each target 
    for y in Y_ready:
        
        # place holder to store the new preprocessed tag list
        Y_place_holder = []
        
        # for each tag in rhe tag list 
        for tag in y:
            # append the id of the tag in the place holder list
            Y_place_holder.append(tags2id[tag])
        
        # find the lenght of the new preprocessed tag list 
        len_new_tag_list = len(Y_place_holder)
        # find the differance in length between the len of tag list and padded sentences
        num_O_to_add = maxlen - len_new_tag_list
        
        # add 'O's to padd the tag lists
        padded_tags = Y_place_holder + ([tags2id['O']] * num_O_to_add)
        Y_preprocessed.append(padded_tags[:maxlen])
        
    return Y_preprocessed


# In[141]:


Y_preprocessed = preprocess_tags(tags2id, Y_ready)


# In[142]:


print(Y_preprocessed[7373])


# In[143]:


print(Y_preprocessed[0])


# In[144]:


# for x in Y_preprocessed:
#     if len(x) != 110:
#         print(len(x))


# In[145]:


print(Y_ready[0])


# ### By now we have the data ready for training our model 
# 
# 
# **We have X_preprocessed and Y_preprocessed that we will use to train our model**

# The las step is to **split** the data into: 
# 
# - Training dataset 
# - Valisdation dataset 
# - testing dataset 

# - **Data shuffling and splitting**

# In[146]:


print("The Lenght of training examples: {}".format(len(X_preprocessed)))
print("The Lenght of training targets: {}".format(len(Y_preprocessed)))


# In[147]:


X_preprocessed = np.asarray(X_preprocessed)
# Y_preprocessed = [np.array(inner_list) for inner_list in Y_preprocessed]
Y_preprocessed = np.asarray(Y_preprocessed)
# Y_preprocessed = np.asarray(Y_preprocessed_arrays)


# In[148]:


Y_preprocessed


# In[149]:


X_preprocessed


# In[150]:


# 70% of the datat will be used for training 
training_samples = 0.7
# 15% of the datat will be used for validation 
validation_samples = 0.15
# 15% of the datat will be used for testing 
testing_samples = 0.15


# In[151]:


indices = np.arange(len(Y_preprocessed))


# In[152]:


np.random.seed(seed=555)
np.random.shuffle(indices)


# In[153]:


indices


# In[154]:


X_preprocessed = X_preprocessed[indices]
indices = indices.astype(int)
Y_preprocessed = Y_preprocessed[indices]


# In[155]:


X_train = X_preprocessed[: int(0.7 * len(X_preprocessed))]
print("Number of training examples: {}".format(len(X_train)))


X_val = X_preprocessed[int(0.7 * len(X_preprocessed)) : int(0.7 * len(X_preprocessed)) + (int(0.15 * len(X_preprocessed)) + 1)]
print("Number of validation examples: {}".format(len(X_val)))


X_test = X_preprocessed[int(0.7 * len(X_preprocessed)) + (int(0.15 * len(X_preprocessed)) + 1) : ]
print("Number of testing examples: {}".format(len(X_test)))



Y_train = Y_preprocessed[: int(0.7 * len(X_preprocessed))]
Y_val = Y_preprocessed[int(0.7 * len(X_preprocessed)) : int(0.7 * len(X_preprocessed)) + (int(0.15 * len(X_preprocessed)) + 1)]
Y_test = Y_preprocessed[int(0.7 * len(X_preprocessed)) + (int(0.15 * len(X_preprocessed)) + 1) : ]

print("Total number of examples after shuffling and splitting: {}".format(len(X_train) + len(X_val) + len(X_test)))


# # 5- Model Training and Evaluation

# In[156]:


X_train[1000]


# In[157]:


Y_train[1000]


# In[158]:


id2word[729]


# ## Load dataset to the model using train_dataset = tf.data.Dataset 
# 

# In[172]:


X_train[:2]


# In[160]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))


# In[161]:


BATCH_SIZE = 132
SHUFFLE_BUFFER_SIZE = 132

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# In[162]:


embedding_dim = 300
maxlen = 256
max_words = 36000
num_tags = len(tags)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_words, embedding_dim, input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, activation='tanh', return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, activation='tanh', return_sequences=True)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags, activation='softmax'))
])


# In[163]:


model.summary()


# In[164]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[165]:


# train_dataset.shape


# In[166]:


history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=15)


# In[167]:


model.evaluate(test_dataset)


# In[168]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 4), dpi=80)

ax[0].plot(epochs, acc, label = "Training Accuracy", color='darkblue')
ax[0].plot(epochs, val_acc, label = "Validation Accuracy", color='darkgreen')
ax[0].grid(alpha=0.3)
ax[0].title.set_text('Training Vs Validation Accuracy')
ax[0].fill_between(epochs, acc, val_acc, color='crimson', alpha=0.3)
plt.setp(ax[0], xlabel='Epochs')
plt.setp(ax[0], ylabel='Accuracy')


ax[1].plot(epochs, loss, label = "Training Loss", color='darkblue')
ax[1].plot(epochs, val_loss, label = "Validation Loss", color='darkgreen')
ax[1].grid(alpha=0.3)
ax[1].title.set_text('Training Vs Validation Loss')
ax[1].fill_between(epochs,loss, val_loss, color='crimson', alpha=0.3)
plt.setp(ax[1], xlabel='Epochs')
plt.setp(ax[1], ylabel='Loss')

plt.show()


# In[194]:


def make_prediction(model, preprocessed_sentence, id2word, id2tag):
    
    #if preprocessed_sentence.shape() != (1, 110):
    preprocessed_sentence = preprocessed_sentence.reshape((1, 256))
     
    # return preprocessed sentence to its orginal form
    sentence = preprocessed_sentence[preprocessed_sentence > 0]
    word_list = []
    for word in list(sentence):
        word_list.append(id2word[word])
    orginal_sententce = ' '.join(word_list)
    
    len_orginal_sententce = len(word_list)
    
    # make prediction
    prediction = model.predict(preprocessed_sentence)
    prediction = np.argmax(prediction[0], axis=1)
    
    # return the prediction to its orginal form
    prediction = list(prediction)[ : len_orginal_sententce] 
    
    pred_tag_list = []
    for tag_id in prediction:
        pred_tag_list.append(id2tag[tag_id])
    
    return (orginal_sententce,  pred_tag_list)


# In[195]:


(orginal_sententce,  pred_tag_list) = make_prediction(model=model,
                                                    preprocessed_sentence=X_test[520],
                                                    id2word=id2word,
                                                    id2tag=id2tag)


# In[200]:


make_prediction(model=loaded_model, preprocessed_sentence=X_test[520], id2word=id2word, id2tag=id2tag)[1][0]


# In[196]:


X_test[520]


# In[197]:


print(orginal_sententce)


# In[198]:


print(pred_tag_list)


# In[ ]:





# # Load the model

# In[169]:


# Save the trained model
model.save('ner_nn_model.h5')

loaded_model = tf.keras.models.load_model('ner_nn_model.h5')


# In[171]:


loaded_model.summary()


# # Making prediction

# In[186]:


# from sklearn.preprocessing import LabelEncoder
# from tqdm import tqdm

# def load_data_test(data_path):
#     with open(data_path) as f:
#         data = f.read().splitlines()

# #   with open(gt_path, "r") as f:
# #       labels = f.read().splitlines()

# #   df = pd.DataFrame({"text": data, "label": labels})
#     df = pd.DataFrame({"text": data})
#     df = df[df["text"] != ";;;"]
#     df["text"] = df["text"].apply(lambda x: x.replace(";;;", ""))
# #   df = df[~(df["label"].str.strip()=="")]
# #   df = df[~df["label"].str.contains(";")]

# #   df["label"] = df["label"].str.strip()

# #   df["label"] = np.where(df["label"] == "O O", "O", df["label"])
#     return df

# test = load_data_test("train_data.csv")
# # valid = load_data("valid_gt.csv", "valid_data.csv")
# # label_mapping = {'O': 0, 'B-ORG': 1, 'B-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-ORG': 6, 'I-MISC': 7, 'I-LOC': 8}
# # train['label'] = train['label'].map(label_mapping)
# # valid['label'] = valid['label'].map(label_mapping)
# # print(len(valid))
# # print(train['label'].unique())
# # print(valid['label'].unique())
# ner_pos = LabelEncoder()
# def rows_to_sentences_test(df):
#     sentences = []
# #     sentences_labels = []
#     current_sentence = []
# #     current_labels = []

#     for index, row in tqdm(df.iterrows(), total = len(df)):
#         word = row['text']
#         current_sentence.append(word.strip())
# #         current_labels.append(label)
#         if word.strip() == '.':
#             sentences.append(current_sentence)
# #             sentences_labels.append(current_labels)
#             current_sentence = []
# #             current_labels = []

#     return sentences

# test_sentences = rows_to_sentences_test(test)


# In[187]:


# test_sentences


# In[188]:


from pandas import read_csv

# data = read_csv('drive/MyDrive/Colab Notebooks/compete data/train_data.csv', sep=';',  header=None, skip_blank_lines=False)
test_data = read_csv('test_data.csv', sep=';',  header=None, skip_blank_lines=False)
# data = read_csv('drive/MyDrive/Colab Notebooks/compete data/train_data.csv', sep=';')

len(test_data)

test_data.rename(columns = {0:'data'}, inplace = True)

test_data['data'] = test_data['data'].fillna('')

test_d = test_data["data"]

# X_test_d_features = tokenizer.transform(test_d)


# In[192]:


test_data


# In[207]:


test_data['data'][0]


# In[208]:


tokenizer.texts_to_sequences([test_data['data'][0]])


# In[209]:


pad_sequences(tokenizer.texts_to_sequences([test_data['data'][0]]), maxlen=maxlen, padding='post')


# In[ ]:


# Print or use the DataFrame as needed
test_data['label'] = test_data['data'].apply(lambda x: make_prediction(model=loaded_model, preprocessed_sentence=pad_sequences(tokenizer.texts_to_sequences([test_data['data'][0]]), maxlen=maxlen, padding='post'), id2word=id2word, id2tag=id2tag)[1][0])


# In[ ]:


test_data


# In[ ]:


# y_test_d_predictions = loaded_model.predict(X_test_d_features)


# In[ ]:


# import numpy as np
# import csv

# csv_file_path = 'output_nnn.csv'

# with open(csv_file_path, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     for item in y_test_d_predictions:
#         csvwriter.writerow([item])

# print(f"Array has been written to {csv_file_path}")


# In[ ]:


# import zipfile

# csv_file_path = 'output_nnn.csv'
# zip_file_path = 'output_nnn.csv.zip'

# # Create a Zip file and add the CSV file to it
# with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#     zipf.write(csv_file_path, arcname='output_nnn.csv')

# print(f"{csv_file_path} has been zipped to {zip_file_path}")


# # 6- Thank you

# **Thank you for reading, I hope you enjoyed and benefited from it.**
# 
# **If you have any questions or notes please leave it in the comment section.**
# 
# **If you like this notebook please press upvote and thanks again.**
