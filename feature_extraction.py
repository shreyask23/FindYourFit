import numpy as np
import pandas as pd
import tensorflow as tf
import os,sys,time
import matplotlib.pyplot as plt
from PIL import Image

import warnings as wrn
wrn.filterwarnings('ignore')

tf.compat.v1.disable_eager_execution()

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,GRU,CuDNNGRU,Embedding
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

dataDesc = pd.read_csv("/input/flickr8kimagescaptions/flickr8k/captions.txt")

ABSOLUTE_PATH = "/input/flickr8kimagescaptions/flickr8k/images/"

def loadDataset(resize=None):
    images = []
    for index,image in enumerate(np.unique(dataDesc["image"])):
        sys.stdout.write(f"\r Reading Image {index}")
        sys.stdout.flush()
        # Reading Image
        img = Image.open(ABSOLUTE_PATH + image)
        # If we specified resize, resize the image.
        if resize != None:
            img = img.resize(resize,resample=Image.LANCZOS)
        
        images.append(np.asarray(img))
            
    captions = []
    cnt = 0
    for image in np.unique(dataDesc["image"]):
        sys.stdout.write(f"\rReading Caption {cnt+1}")
        cnt += 1
        chck = []
        # Each image has captions more than one, so we'll store captions in a list of lists.
        for cap in dataDesc[dataDesc["image"] == image]["caption"].values:
            chck.append(cap)
        captions.append(chck)
        
    return np.asarray(images),captions

base_model = VGG16()
base_model.summary()

transfer_layer = base_model.get_layer("fc2")
image_model_transfer = Model(inputs=[base_model.input],outputs=[transfer_layer.output])

img_size = K.int_shape(image_model_transfer.input)[1:3]

transfer_values_size = K.int_shape(image_model_transfer.output)[1]

%%time
def processDataset():
    
    images,captions = loadDataset(resize=img_size)
    print("Reading images finished")
    vecs = image_model_transfer.predict(images)
    return vecs,captions

transfer_values,captions = processDataset()

mark_start = "ssss "
mark_end = " eeee"

def markCaptions(captions_listlist):
    return [[mark_start + caption + mark_end for caption in captions_list]
           for captions_list in captions_listlist]

marked_captions = markCaptions(captions)

flattened_captions = [caption for cap_list in marked_captions for caption in cap_list]

class TokenizerWrap(Tokenizer): 
    
    def __init__(self, texts, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)
        
        self.fit_on_texts(texts)

        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]
        
        text = " ".join(words)

        return text
    
    def captions_to_tokens(self, captions_listlist):
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        return tokens

num_words = 10000
tokenizer = TokenizerWrap(texts=flattened_captions,num_words=num_words)

tokens_train = tokenizer.captions_to_tokens(marked_captions)

token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]

# This function will return random captions for specified indices.
def get_random_captions_tokens(idx):
    results = [] 
    for i in idx:
        j = np.random.choice(len(np.asarray(tokens_train[i])))
        tokens = tokens_train[i][j]
        results.append(tokens)
    
    return results

num_images = 8091

def batch_generator(batch_size):
    while True:
        idx = np.random.randint(num_images,size=batch_size)
        
        t_values = np.asarray(list(map(transfer_values.__getitem__,idx)))
        tokens = get_random_captions_tokens(idx)

        num_tokens = [len(t) for t in tokens]
        max_tokens = np.max(num_tokens)

        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding="post",
                                      truncating="post"
                                     )

        decoder_input_data = tokens_padded[:,0:-1]
        decoder_output_data = tokens_padded[:,1:]

        x_data = {"decoder_input":decoder_input_data,
                  "transfer_values_input":t_values
                 }

        y_data = {"decoder_output":decoder_output_data}
        
        yield (x_data,y_data)

total_num_captions = len(dataDesc["caption"])

steps_per_epoch = int(total_num_captions / BATCH_SIZE)

state_size = 256
embedding_size = 100

transfer_values_input = Input(shape=(transfer_values_size,),
                              name="transfer_values_input"
                             )

decoder_transfer_map = Dense(state_size,
                             activation="tanh",
                             name="decoder_transfer_map"
                            )

decoder_input = Input(shape=(None,),name="decoder_input")

word2vec = {}
for line in open("../input/glove6b100dtxt/glove.6B.100d.txt",encoding="utf-8"):
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:],dtype="float32")
    word2vec[word] = vec

replaced_count = 0
embedding_matrix = np.random.uniform(-1,1,(num_words,embedding_size))
for word,i in tokenizer.word_index.items():
    vec = word2vec.get(word)
    if vec is not None:
        embedding_matrix[i] = vec
        replaced_count += 1

decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              weights=[embedding_matrix],
                              trainable=True,
                              name="decoder_embedding"
                             )

decoder_gru1 = CuDNNGRU(state_size,return_sequences=True,name="decoder_gru1")
decoder_gru2 = CuDNNGRU(state_size,return_sequences=True,name="decoder_gru2")
decoder_gru3 = CuDNNGRU(state_size,return_sequences=True,name="decoder_gru3")

decoder_dense = Dense(num_words,
                      activation="linear",
                      name="decoder_output"
                     )

def connectDecoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)

    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net,initial_state=initial_state)
    net = decoder_gru2(net,initial_state=initial_state)
    net = decoder_gru3(net,initial_state=initial_state)
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connectDecoder(transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input,decoder_input],outputs=[decoder_output])

def sparse_cross_entropy(y_true,y_pred):
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    
    loss_mean = tf.reduce_mean(loss)
    
    return loss_mean

optimizer = RMSprop(lr=1e-3)
decoder_target = tf.compat.v1.placeholder(dtype="int32",shape=(None,None))
decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target]
                     )

path_checkpoint = "model.keras"
checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                             save_weights_only=True
                            )

try:
    decoder_model.load_weights(path_checkpoint)
except Exception as E:
    print("Something went wrong when loading the checkpoint, training will start from scratch")
    print()
    print(E)

decoder_model.fit_generator(generator=batch_generator(BATCH_SIZE),
                            steps_per_epoch=steps_per_epoch,
                            epochs=10,
                            callbacks=[checkpoint]
                           )

def generate_caption(image_path,max_tokens=30):
    
    # Reading and extracting features using VGG16.
    image = np.asarray(Image.open(image_path).resize((224,224)))
    transfer_values = image_model_transfer.predict(np.expand_dims(image,axis=0))
    
    # We'll create our input text, it will have just spaces and model will fill them.
    decoder_input_data = np.zeros(shape=(1,max_tokens),dtype=np.int)
    
    token_int = token_start
    output_text = " "
    count_tokens = 0
    
    # While our model don't create finish token 
    while token_int != token_end and count_tokens < max_tokens:
        
        decoder_input_data[0,count_tokens] = token_int
        x_data = {"transfer_values_input":transfer_values,
                  "decoder_input":decoder_input_data
                 }
        
        # Model will predict the next word.
        decoder_output = decoder_model.predict(x_data)
        
        token_onehot = decoder_output[0,count_tokens,:]
        token_int = np.argmax(token_onehot)
        
        sampled_word = tokenizer.token_to_word(token_int)
        output_text = output_text + " " + sampled_word
        count_tokens += 1
        
    plt.imshow(image)
    plt.axis("off")
    print()
        
    print("Predicted Caption:")
    print(output_text.replace("ssss"," ").replace("eeee"," ").strip())
    print()

generate_caption("../input/clothing-dataset-full/images_original/00143901-a14c-4600-960f-7747b4a3a8cd.jpg")

