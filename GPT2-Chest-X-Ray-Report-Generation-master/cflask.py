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
from keras_preprocessing.image import img_to_array,load_img
from generator import AugmentedImageSequence
from flask import Flask, render_template, request


FLAGS = argHandler()
FLAGS.setDefaults()

tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                         FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)


step_n=1
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
    
df=pd.read_csv("IU-XRay/all_data.csv")

im_path="CXR1_1_IM-0001-4001.png"


dftemp=df[df["Image Index"]==im_path]
dftemp

csv_filename = 'output.csv'
dftemp.to_csv(csv_filename, index=False)



deploy_enqueuer, deploy_steps = get_enqueuer("output.csv", 1, FLAGS, tokenizer_wrapper)
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

print(ckpt_manager.latest_checkpoint)


if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))

target,pred = get_sentence(deploy_enqueuer, deploy_steps, FLAGS, encoder, decoder, tokenizer_wrapper, write_images=True, test_mode=True)

from flask import Flask, render_template, request

app=Flask("__name__")

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():

    classification = pred
    return render_template('index.html', prediction=classification)

if __name__=="__main__":
    app.run(port=3000,debug=True)