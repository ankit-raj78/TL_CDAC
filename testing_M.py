import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
import Data_preprocessing



def load_params():
    with open('params1.p', mode='rb') as in_file:
        return pickle.load(in_file)

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = Data_preprocessing.load_preprocess()

load_path = load_params()



batch_size = 30

def word_to_seq(word, vocab_to_int):
    results = []
    for word in list(word):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results



transliterate_word = st.text_input("","")


#transliterate_word = input().lower()

transliterate_word = word_to_seq(transliterate_word, source_vocab_to_int)
loaded_graph = tf.Graph()

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
        
    loader = tf.train.import_meta_graph(load_path + '.meta')
    
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    transliterate_logits = sess.run(logits, {input_data: [transliterate_word]*batch_size,
                                         target_sequence_length: [len(transliterate_word)]*batch_size,
                                         keep_prob: 1.0})[0]
output = ""
for i in transliterate_logits:
    if target_int_to_vocab[i]!= '<EOS>':
        output = output + target_int_to_vocab[i]
st.write(output)

                 
        
#st.write('Input')
#st.write('  Word Ids:      {}'.format([i for i in transliterate_word]))
#st.write('  English Word: {}'.format([source_int_to_vocab[i] for i in transliterate_word]))

#st.write('\nPrediction')
#st.write('  Word Id:      {}'.format([i for i in transliterate_logits]))

