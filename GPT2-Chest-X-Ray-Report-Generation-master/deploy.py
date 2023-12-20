#!/usr/bin/env python
# coding: utf-8

# In[43]:


import tensorflow as tf
from CNN_encoder import CNN_Encoder
import os
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from caption_evaluation import get_evalutation_scores
from utility import get_enqueuer
import numpy as np
from PIL import Image
import json
import time
from gpt.gpt2_model import TFGPT2LMHeadModel
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize

import re
import pickle


# In[44]:


from keras_preprocessing.image import img_to_array,load_img


# In[45]:


from generator import AugmentedImageSequence


# In[46]:


FLAGS = argHandler()
FLAGS.setDefaults()


# In[47]:


FLAGS.help()


# In[48]:


tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                         FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)


# In[49]:


step_n=1


# In[50]:


def get_sen(FLAGS, encoder, decoder, tokenizer_wrapper, images):
    global step_n
    visual_features, tags_embeddings = encoder(images)
    dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
    # dec_input = tf.tile(dec_input,[images.shape[0],1])
    num_beams = FLAGS.beam_width

    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])

    tokens = decoder.generate(dec_input, max_length=FLAGS.max_sequence_length, num_beams=num_beams, min_length=3,
                              eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(), no_repeat_ngram_size=0,
                              visual_features=visual_features,
                              tags_embedding=tags_embeddings, do_sample=False, early_stopping=True)
    sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
    sentence = tokenizer_wrapper.filter_special_words(sentence)
    step_n += 1
    return sentence


# In[51]:


def get_sentence(enqueuer, steps, FLAGS, encoder, decoder, tokenizer_wrapper, name='Test set', verbose=True,
                      write_json=True, write_images=False, test_mode=False):
    
    if not enqueuer.is_running():
        enqueuer.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)
    csv_dict = {"image_path": [], "real": [], "prediction": []}
    generator = enqueuer.get()
    for batch in tqdm(list(range(steps))):
        images, target, img_path = next(generator)
        predicted_sentence = get_sen(FLAGS, encoder, decoder, tokenizer_wrapper,images)
        target_sentence = tokenizer_wrapper.GPT2_decode(target[0])
        print("Target Sentence: ")
        print(target_sentence)
        print("PREDICTED SENTENCE: \n")
        print(predicted_sentence)
        return target_sentence,predicted_sentence


# In[58]:


df=pd.read_csv("IU-XRay/all_data.csv")
df


# In[108]:


im_path="CXR998_IM-2479-1001.png"


# In[110]:


dftemp=df[df["Image Index"]==im_path]
dftemp


# In[ ]:


#i=3
#df.iloc[i]


# In[111]:


i=1
csv_filename = 'output{}.csv'.format(i)
dftemp.to_csv(csv_filename, index=False)
i+=1


# In[82]:


deploy_enqueuer, deploy_steps = get_enqueuer("output{}.csv".format(i), 1, FLAGS, tokenizer_wrapper)
deploy_enqueuer.start(workers=1, max_queue_size=8)

encoder = CNN_Encoder('pretrained_visual_model', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers,
                          FLAGS.encoder_layers, FLAGS.tags_threshold, num_tags=len(FLAGS.tags))

decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)

#optimizer = tf.keras.optimizers.Adam()
optimizer=tf.keras.optimizers.legacy.Adam()

ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)


# In[83]:


print(ckpt_manager.latest_checkpoint)


# In[84]:


if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))


# In[85]:


target,pred = get_sentence(deploy_enqueuer, deploy_steps, FLAGS, encoder, decoder, tokenizer_wrapper, write_images=True, test_mode=True)


# In[86]:


pattern = re.compile(r'"([^"]*)"')
match = pattern.search(pred)
if match:
    pred = match.group(1)
    print(pred)
else:
    print("No sentence found.")


# In[87]:


print(target)


# In[88]:


pickle_file_path="medical_word_embeddings\saved_embeddings.pickle"


# In[89]:


with open(pickle_file_path,"rb") as file:
    word_embeddings=pickle.load(file)


# In[90]:


def get_sentence_embeddings(sentence, word_embeddings):
    words = sentence.lower().split()  # Assuming the words are lowercase in your word_embedding dictionary

    # Filter out words that are not in the word_embedding dictionary
    valid_words = [word for word in words if word in word_embeddings]

    if not valid_words:
        # If no valid words are found in the word_embedding dictionary, return a zero vector
        return np.zeros(len(word_embeddings[next(iter(word_embeddings))]))

    # Calculate the mean of word embeddings for the valid words
    embedding_matrix = [word_embeddings[word] for word in valid_words]
    sentence_embedding = np.mean(embedding_matrix, axis=0)

    return sentence_embedding


# In[91]:


from sklearn.metrics.pairwise import cosine_similarity


# In[92]:


def calculate_cosine_similarity(ground_truth, predicted, word_embeddings):
    ground_truth_embedding = get_sentence_embeddings(ground_truth, word_embeddings)
    predicted_embedding = get_sentence_embeddings(predicted, word_embeddings)

    # Reshape the vectors to 2D arrays
    ground_truth_embedding = ground_truth_embedding.reshape(1, -1)
    predicted_embedding = predicted_embedding.reshape(1, -1)

    similarity_score = cosine_similarity(ground_truth_embedding, predicted_embedding)
    return similarity_score[0][0]


# In[93]:


similarity = calculate_cosine_similarity(target, pred,word_embeddings)


# In[94]:


print("Cosine Similarity: {}".format(similarity))


# In[95]:


similarity


# In[96]:


from flask import Flask, render_template, request


# In[97]:


app=Flask("__name__")


# In[98]:


@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")


# In[99]:


@app.route("/",methods=["POST"])
def predict():

    classification = target


    return render_template('index.html', prediction=classification)


# In[100]:


if __name__=="__main__":
    app.run(port=3000,debug=True)

