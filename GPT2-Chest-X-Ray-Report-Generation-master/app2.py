import os
from flask import Flask, render_template, request, jsonify
from CNN_encoder import CNN_Encoder
from gpt.gpt2_model import TFGPT2LMHeadModel
from generator import AugmentedImageSequence
from caption_evaluation import get_evaluation_scores
from tokenizer_wrapper import TokenizerWrapper
from utility import get_enqueuer
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

FLAGS = argHandler()
FLAGS.setDefaults()

tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                     FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

def get_sen(FLAGS, encoder, decoder, tokenizer_wrapper, images):
    visual_features, tags_embeddings = encoder(images)
    dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
    num_beams = FLAGS.beam_width

    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])

    tokens = decoder.generate(dec_input, max_length=FLAGS.max_sequence_length, num_beams=num_beams, min_length=3,
                              eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(), no_repeat_ngram_size=0,
                              visual_features=visual_features,
                              tags_embedding=tags_embeddings, do_sample=False, early_stopping=True)
    sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
    sentence = tokenizer_wrapper.filter_special_words(sentence)
    return sentence

def get_sentence(enqueuer, steps, FLAGS, encoder, decoder, tokenizer_wrapper, name='Test set', verbose=True,
                 write_json=True, write_images=False, test_mode=False):
    if not enqueuer.is_running():
        enqueuer.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)
    csv_dict = {"image_path": [], "real": [], "prediction": []}
    generator = enqueuer.get()
    for batch in tqdm(list(range(steps))):
        images, target, img_path = next(generator)
        predicted_sentence = get_sen(FLAGS, encoder, decoder, tokenizer_wrapper, images)
        target_sentence = tokenizer_wrapper.GPT2_decode(target[0])
        print("Target Sentence: ")
        print(target_sentence)
        print("PREDICTED SENTENCE: \n")
        print(predicted_sentence)
        return target_sentence, predicted_sentence

df = pd.read_csv("IU-XRay/all_data.csv")

app = Flask("__name__")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path = "./oimages/" + imagefile.filename
    imagefile.save(image_path)

    im_path = imagefile.filename
    dftemp = df[df["Image Index"] == im_path]
    csv_filename = 'output.csv'
    dftemp.to_csv(csv_filename, index=False)
    deploy_enqueuer, deploy_steps = get_enqueuer("output.csv", 1, FLAGS, tokenizer_wrapper)
    deploy_enqueuer.start(workers=1, max_queue_size=8)

    encoder = CNN_Encoder('pretrained_visual_model', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers,
                          FLAGS.encoder_layers, FLAGS.tags_threshold, num_tags=len(FLAGS.tags))

    decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)

    optimizer = tf.keras.optimizers.legacy.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))

    target, pred = get_sentence(deploy_enqueuer, deploy_steps, FLAGS, encoder, decoder, tokenizer_wrapper,
                                write_images=True, test_mode=True)

    pred_sentence = pred
    target_sentence = target
    print(pred_sentence)

    # Save the input image path and predictions to a text file
    with open("output_log.txt", "a") as f:
        f.write(f"Image Path: {image_path}\n")
        f.write(f"Target Sentence: {target_sentence}\n")
        f.write(f"Predicted Sentence: {pred_sentence}\n\n")

    # Run the Jupyter Notebook
    os.system("jupyter nbconvert --to notebook --execute deploy.ipynb")

    return render_template('index.html', prediction=pred_sentence)

if __name__ == "__main__":
    app.run(port=3000, debug=True)
