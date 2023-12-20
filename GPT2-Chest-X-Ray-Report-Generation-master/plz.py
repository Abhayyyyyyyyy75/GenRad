
#this mf is working
#use this
#make changes here
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
from flask import Flask, jsonify, render_template, request

import os
import base64
from io import BytesIO
FLAGS = argHandler()
FLAGS.setDefaults()



app=Flask("__name__")


UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/predict', methods=['POST'])


@app.route("/predict",methods=["POST"])
def predict():
    try:
        print("hello im app start")
        input_data = request.get_json()
        image_data = input_data['data']
        actual_name = input_data['name']
        print(image_data)
        filename = input_data.get('name',actual_name )
        
        l1 = filename
        actual_image = l1[:-4]

        # Ensure the image data is a bytes-like object
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to image
        image = Image.open(BytesIO(image_bytes))

        # Save the image to the upload folder with the original filename
        filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print("the image name is: ", filename)
        image.save(filename)




        tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                            FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)
        
        def get_sen(FLAGS, encoder, decoder, tokenizer_wrapper, images):
            step_n=1
            # global step_n
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
        

        im_path=l1

        
        # df=pd.read_csv("IU-XRay/all_data.csv")
        # imagefile=request.files["imagefile"]
        # image_path="./oimages/"+imagefile.filename
        # imagefile.save(image_path)

        # im_path=imagefile.filename

        
        dftemp=df[df["Image Index"]==im_path]
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
        pattern = re.compile(r'"([^"]*)"')
        match = pattern.search(pred)
        if match:
            pred = match.group(1)
            print("predicted value is :",pred)
        else:
            print("No sentence found.")
        classification = pred
        class2=target
        



        print('prediction',classification)
        print('target', target)
        print("done \\fms")
        response_data = {
            'prediction': classification,
            'target': target
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__=="__main__":
    print("server started ")
    app.run(port=8000,debug=True)
