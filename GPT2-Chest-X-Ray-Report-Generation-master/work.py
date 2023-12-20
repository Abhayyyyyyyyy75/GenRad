from flask import Flask, render_template, request
import werkzeug  # For secure filename
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
from keras_preprocessing.image import img_to_array,load_img
from generator import AugmentedImageSequence
import re
import pickle
FLAGS = argHandler()
FLAGS.setDefaults()
# ... [Other imports and initializations] ...
# Initialize encoder, decoder, optimizer here (outside of the Flask route)




app = Flask("__name__")

@app.route("/", methods=["GET", "POST"])  # Handle both GET and POST
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if 'imagefile' not in request.files:
        return 'No file part', 400
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return 'No selected file', 400
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    image_path = os.path.join("./oimages/", filename)
    imagefile.save(image_path)
    df=pd.read_csv("IU-XRay/all_data.csv")
    tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                            FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)
    im_path=imagefile.filename
    dftemp=df[df["Image Index"]==im_path]
    csv_filename = 'output.csv'
    dftemp.to_csv(csv_filename, index=False)
    encoder = CNN_Encoder('pretrained_visual_model', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers,
                        FLAGS.encoder_layers, FLAGS.tags_threshold, num_tags=len(FLAGS.tags))
    decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)
    optimizer = tf.keras.optimizers.legacy.Adam()
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))
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
        
    deploy_enqueuer, deploy_steps = get_enqueuer("output.csv", 1, FLAGS, tokenizer_wrapper)
    deploy_enqueuer.start(workers=1, max_queue_size=8)    
        
    target,pred = get_sentence(deploy_enqueuer, deploy_steps, FLAGS, encoder, decoder, tokenizer_wrapper, write_images=True, test_mode=True)

    # ... [Rest of your code to process the image and generate a prediction] ...

    return render_template('index.html', prediction=pred)

if __name__ == "main":
    app.run(port=3000, debug=True)
# This is the mistakes out lined by GPT
# Flask Route Methods: Ensure that each Flask route is set up to handle the appropriate HTTP methods. For the "/predict" route, it seems you are handling a POST request, which is correct for form submission. However, the home route "/" should typically handle both GET and POST requests if you're returning the form and handling its submission on the same route.

# Repeated Initialization: You're initializing encoder, decoder, and optimizer twice - once before starting the Flask app and once inside the predict function. This is redundant and can lead to unnecessary resource usage. It's better to initialize these components once and reuse them.

# File Path Handling: Ensure that the directory where you're saving the uploaded files ("./oimages/") exists. Also, when reading the image file, use the complete path (image_path), not just the filename (im_path).

# Security Consideration: Directly using the filename from the user's upload (imagefile.filename) without any sanitation is a security risk. Filenames should be secured before usage.