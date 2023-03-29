# Deep Learning with keras

In this workshop, we will try to build some feedforward models to do sentiment analysis, using keras, a deep learning library: https://keras.io/

You will need pandas, keras (2.3.1) and tensorflow (2.1.0; and their dependencies) to run this code (pip install pandas keras==2.3.1 tensorflow-cpu==2.1.0).

First let's prepare the data. We are using 1000 yelp reviews, nnotated with either positive or negative sentiments.


```python
import pandas as pd

corpus = "07-yelp-dataset.txt"
df = pd.read_csv(corpus, names=['sentence', 'label'], sep='\t')
print("Number of sentences =", len(df))
print("\nData:")
print(df.iloc[:3])
```

    Number of sentences = 1000
    
    Data:
                                        sentence  label
    0                   Wow... Loved this place.      1
    1                         Crust is not good.      0
    2  Not tasty and the texture was just nasty.      0


Next, let's create the train/dev/test partitions


```python
import random
import numpy as np

sentences = df['sentence'].values
labels = df['label'].values

#partition data into 80/10/10 for train/dev/test
sentences_train, y_train = sentences[:800], labels[:800]
sentences_dev, y_dev = sentences[800:900], labels[800:900]
sentences_test, y_test = sentences[900:1000], labels[900:1000]

#convert label list into arrays
y_train = np.array(y_train)
y_dev = np.array(y_dev)
y_test = np.array(y_test)

print(y_train[0], sentences_train[0])
print(y_dev[0], sentences_dev[0])
print(y_test[0], sentences_test[0])
```

    1 Wow... Loved this place.
    0 I'm super pissd.
    0 Spend your money elsewhere.


Let's tokenize the text. In this workshop, we'll use the ``tokenizer`` function provided by keras. Once the data is tokenized, we can then use ``texts_to_matrix`` to get the bag-of-words representation for each document.


```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(sentences_train)

x_train = tokenizer.texts_to_matrix(sentences_train, mode="count") #BOW representation
x_dev = tokenizer.texts_to_matrix(sentences_dev, mode="count") #BOW representation
x_test = tokenizer.texts_to_matrix(sentences_test, mode="count") #BOW representation

vocab_size = x_train.shape[1]
print("Vocab size =", vocab_size)
print(x_train[0])
```

    Using TensorFlow backend.


    Vocab size = 1811
    [0. 0. 0. ... 0. 0. 0.]


Before we build a neural network model, let's see how well logistic regression do with this dataset.


```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)

print("Accuracy:", score)
```

    Accuracy: 0.69


    /Users/laujh/.pyenv/versions/3.6.9/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


The logistic regression result is not too bad, and it will serve as a baseline for the deep learning models.

Now let's build a very simple feedforward network. Here the input layer is the BOW features, and we have one hidden layer (dimension = 10) and an output layer in the model.


```python
from keras.models import Sequential
from keras import layers

#model definition
model = Sequential(name="feedforward-bow-input")
model.add(layers.Dense(10, input_dim=vocab_size, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#since it's a binary classification problem, we use a binary cross entropy loss here
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

    Model: "feedforward-bow-input"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 10)                18120     
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 18,131
    Trainable params: 18,131
    Non-trainable params: 0
    _________________________________________________________________


Let's train the model. Notice that there are a few hyper-parameters here, e.g. hidden layer size, number of epochs and batch_size, and in practice these hyper-parameters should be tuned according to the development data to get an optimal model. In this workshop we'll use 20 epochs and a batch size of 10 (no further tuning). Once the model is trained, we'll compute the test accuracy performance.


```python
#training
model.fit(x_train, y_train, epochs=20, verbose=True, validation_data=(x_dev, y_dev), batch_size=10)

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("\nTesting Accuracy:  {:.4f}".format(accuracy))
```

    Train on 800 samples, validate on 100 samples
    Epoch 1/20
    800/800 [==============================] - 0s 399us/step - loss: 0.6876 - accuracy: 0.5475 - val_loss: 0.6790 - val_accuracy: 0.6300
    Epoch 2/20
    800/800 [==============================] - 0s 144us/step - loss: 0.6340 - accuracy: 0.7775 - val_loss: 0.6304 - val_accuracy: 0.7000
    Epoch 3/20
    800/800 [==============================] - 0s 129us/step - loss: 0.5328 - accuracy: 0.8425 - val_loss: 0.5619 - val_accuracy: 0.7700
    Epoch 4/20
    800/800 [==============================] - 0s 173us/step - loss: 0.4159 - accuracy: 0.9100 - val_loss: 0.5081 - val_accuracy: 0.7800
    Epoch 5/20
    800/800 [==============================] - 0s 254us/step - loss: 0.3230 - accuracy: 0.9500 - val_loss: 0.4694 - val_accuracy: 0.8100
    Epoch 6/20
    800/800 [==============================] - 0s 151us/step - loss: 0.2528 - accuracy: 0.9613 - val_loss: 0.4428 - val_accuracy: 0.8200
    Epoch 7/20
    800/800 [==============================] - 0s 207us/step - loss: 0.2003 - accuracy: 0.9750 - val_loss: 0.4162 - val_accuracy: 0.8300
    Epoch 8/20
    800/800 [==============================] - 0s 166us/step - loss: 0.1614 - accuracy: 0.9837 - val_loss: 0.4057 - val_accuracy: 0.8300
    Epoch 9/20
    800/800 [==============================] - 0s 142us/step - loss: 0.1315 - accuracy: 0.9887 - val_loss: 0.3980 - val_accuracy: 0.8100
    Epoch 10/20
    800/800 [==============================] - 0s 131us/step - loss: 0.1089 - accuracy: 0.9912 - val_loss: 0.3968 - val_accuracy: 0.8100
    Epoch 11/20
    800/800 [==============================] - 0s 145us/step - loss: 0.0913 - accuracy: 0.9937 - val_loss: 0.3848 - val_accuracy: 0.8100
    Epoch 12/20
    800/800 [==============================] - 0s 150us/step - loss: 0.0773 - accuracy: 0.9950 - val_loss: 0.3913 - val_accuracy: 0.8000
    Epoch 13/20
    800/800 [==============================] - 0s 130us/step - loss: 0.0656 - accuracy: 0.9975 - val_loss: 0.3939 - val_accuracy: 0.8000
    Epoch 14/20
    800/800 [==============================] - 0s 142us/step - loss: 0.0565 - accuracy: 0.9987 - val_loss: 0.3814 - val_accuracy: 0.8000
    Epoch 15/20
    800/800 [==============================] - 0s 134us/step - loss: 0.0494 - accuracy: 0.9987 - val_loss: 0.3860 - val_accuracy: 0.7900
    Epoch 16/20
    800/800 [==============================] - 0s 193us/step - loss: 0.0431 - accuracy: 0.9987 - val_loss: 0.4015 - val_accuracy: 0.8100
    Epoch 17/20
    800/800 [==============================] - 0s 189us/step - loss: 0.0376 - accuracy: 1.0000 - val_loss: 0.4027 - val_accuracy: 0.8100
    Epoch 18/20
    800/800 [==============================] - 0s 147us/step - loss: 0.0331 - accuracy: 1.0000 - val_loss: 0.4063 - val_accuracy: 0.8100
    Epoch 19/20
    800/800 [==============================] - 0s 150us/step - loss: 0.0294 - accuracy: 1.0000 - val_loss: 0.4202 - val_accuracy: 0.8100
    Epoch 20/20
    800/800 [==============================] - 0s 151us/step - loss: 0.0263 - accuracy: 1.0000 - val_loss: 0.4196 - val_accuracy: 0.8100
    
    Testing Accuracy:  0.7800


How does the performance compare to logistic regression? If you run it a few times you may find that it gives slightly different numbers, and that is due to random initialisation of the model parameters.

Even though we did not explicitly define any word embeddings in the model architecture, they are in our model: in the weights between the input and the hidden layer. The hidden layer can therefore be interpreted as a sum of word embeddings for each input document.

Let's fetch the word embeddings of some words, and look at their cosine similarity, and see if they make any sense.


```python
from numpy import dot
from numpy.linalg import norm

embeddings = model.get_layer(index=0).get_weights()[0] #word embeddings layer

emb_love = embeddings[tokenizer.word_index["love"]] #embeddings for 'love'
emb_like = embeddings[tokenizer.word_index["like"]]
emb_lukewarm = embeddings[tokenizer.word_index["lukewarm"]]
emb_bad = embeddings[tokenizer.word_index["bad"]]

print(emb_love)

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

print("love vs. like =", cos_sim(emb_love, emb_like))
print("love vs. lukewarm =", cos_sim(emb_love, emb_lukewarm))
print("love vs. bad =", cos_sim(emb_love, emb_bad))
print("lukewarm vs. bad =", cos_sim(emb_lukewarm, emb_bad))
```

    [-0.21396367  0.27749586  0.2945718  -0.13073313 -0.22764814 -0.2310485
      0.33856174 -0.20201477 -0.2462762   0.23290806]
    love vs. like = 0.8865069
    love vs. lukewarm = -0.966434
    love vs. bad = -0.97216696
    lukewarm vs. bad = 0.97636676


Not bad. You should find that for *love* and *like*, which are both positive sentiment words, produce high cosine similarity. Similar observations for *lukewarm* and *bad*. But when we compare opposite polarity words like *love* and *bad*, we get negative cosine similarity values.

Next, we are going to build another feed-forward model, but this time, instead of using BOW features as input, we want to use the word sequence as input (so order of words is preserved). It is usually not straightforward to do this for classical machine learning models, but with neural networks and embeddings, it's pretty straightforward.

Let's first tokenise the input documents into word sequences.


```python
#tokenise the input into word sequences

xseq_train = tokenizer.texts_to_sequences(sentences_train)
xseq_dev = tokenizer.texts_to_sequences(sentences_dev)
xseq_test = tokenizer.texts_to_sequences(sentences_test)

print(xseq_train[0])
```

    [354, 138, 9, 17]


Because documents have variable lengths, we need to first 'pad' them to make all documents have the same length. keras uses word index 0 to represent 'pad symbols'.


```python
from keras.preprocessing.sequence import pad_sequences

maxlen = 30
xseq_train = pad_sequences(xseq_train, padding='post', maxlen=maxlen)
xseq_dev = pad_sequences(xseq_dev, padding='post', maxlen=maxlen)
xseq_test = pad_sequences(xseq_test, padding='post', maxlen=maxlen)
print(xseq_train[0])
```

    [354 138   9  17   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0]


Now let's build our second model. This model first embeds each word in the input sequence into embeddings, and then concatenate the word embeddings together to represent input sequence. The ``Flatten`` function you see after the embedding layer is essentially doing the concatenation, by 'chaining' the list of word embeddings into a very long vector.

If our word embeddings has a dimension 10, and our documents always have 30 words (padded), then here the concatenated word embeddings have a dimension of 10 x 30 = 300. 

The concatenated word embeddings undergo a linear transformation with non-linear activations (``layers.Dense(10, activation='relu')``), producing a hidden representation with a dimension of 10. It is then passed to the output layer.


```python
embedding_dim = 10

#word order preserved with this architecture
model2 = Sequential(name="feedforward-sequence-input")
model2.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model2.add(layers.Flatten())
model2.add(layers.Dense(10, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model2.summary()
```

    Model: "feedforward-sequence-input"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 30, 10)            18110     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 300)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                3010      
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 21,131
    Trainable params: 21,131
    Non-trainable params: 0
    _________________________________________________________________


Now let's train the model and compute the test accuracy.


```python
model2.fit(xseq_train, y_train, epochs=20, verbose=True, validation_data=(xseq_dev, y_dev), batch_size=10)

loss, accuracy = model2.evaluate(xseq_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
```

    /Users/laujh/.pyenv/versions/3.6.9/lib/python3.6/site-packages/tensorflow_core/python/framework/indexed_slices.py:433: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "


    Train on 800 samples, validate on 100 samples
    Epoch 1/20
    800/800 [==============================] - 0s 321us/step - loss: 0.6925 - accuracy: 0.5462 - val_loss: 0.6948 - val_accuracy: 0.4400
    Epoch 2/20
    800/800 [==============================] - 0s 146us/step - loss: 0.6828 - accuracy: 0.7175 - val_loss: 0.6946 - val_accuracy: 0.4400
    Epoch 3/20
    800/800 [==============================] - 0s 126us/step - loss: 0.6560 - accuracy: 0.7675 - val_loss: 0.6851 - val_accuracy: 0.6200
    Epoch 4/20
    800/800 [==============================] - 0s 127us/step - loss: 0.5742 - accuracy: 0.8375 - val_loss: 0.6498 - val_accuracy: 0.6300
    Epoch 5/20
    800/800 [==============================] - 0s 130us/step - loss: 0.4661 - accuracy: 0.9262 - val_loss: 0.6240 - val_accuracy: 0.6800
    Epoch 6/20
    800/800 [==============================] - 0s 137us/step - loss: 0.3898 - accuracy: 0.9663 - val_loss: 0.6079 - val_accuracy: 0.7300
    Epoch 7/20
    800/800 [==============================] - 0s 137us/step - loss: 0.3401 - accuracy: 0.9887 - val_loss: 0.6138 - val_accuracy: 0.7100
    Epoch 8/20
    800/800 [==============================] - 0s 143us/step - loss: 0.3047 - accuracy: 0.9975 - val_loss: 0.6036 - val_accuracy: 0.7200
    Epoch 9/20
    800/800 [==============================] - 0s 136us/step - loss: 0.2775 - accuracy: 0.9987 - val_loss: 0.6233 - val_accuracy: 0.7000
    Epoch 10/20
    800/800 [==============================] - 0s 152us/step - loss: 0.2565 - accuracy: 1.0000 - val_loss: 0.6262 - val_accuracy: 0.7100
    Epoch 11/20
    800/800 [==============================] - 0s 133us/step - loss: 0.2380 - accuracy: 1.0000 - val_loss: 0.6243 - val_accuracy: 0.7200
    Epoch 12/20
    800/800 [==============================] - 0s 171us/step - loss: 0.2229 - accuracy: 1.0000 - val_loss: 0.6462 - val_accuracy: 0.7200
    Epoch 13/20
    800/800 [==============================] - 0s 159us/step - loss: 0.2091 - accuracy: 1.0000 - val_loss: 0.6473 - val_accuracy: 0.7200
    Epoch 14/20
    800/800 [==============================] - 0s 173us/step - loss: 0.1967 - accuracy: 1.0000 - val_loss: 0.6571 - val_accuracy: 0.7100
    Epoch 15/20
    800/800 [==============================] - 0s 150us/step - loss: 0.1855 - accuracy: 1.0000 - val_loss: 0.6629 - val_accuracy: 0.7000
    Epoch 16/20
    800/800 [==============================] - 0s 130us/step - loss: 0.1752 - accuracy: 1.0000 - val_loss: 0.6687 - val_accuracy: 0.7100
    Epoch 17/20
    800/800 [==============================] - 0s 144us/step - loss: 0.1658 - accuracy: 1.0000 - val_loss: 0.6733 - val_accuracy: 0.7100
    Epoch 18/20
    800/800 [==============================] - 0s 144us/step - loss: 0.1570 - accuracy: 1.0000 - val_loss: 0.6704 - val_accuracy: 0.7100
    Epoch 19/20
    800/800 [==============================] - 0s 160us/step - loss: 0.1490 - accuracy: 1.0000 - val_loss: 0.6753 - val_accuracy: 0.7100
    Epoch 20/20
    800/800 [==============================] - 0s 164us/step - loss: 0.1414 - accuracy: 1.0000 - val_loss: 0.6817 - val_accuracy: 0.7100
    Testing Accuracy:  0.8100


You may find that the performance isn't as good as the BOW model. In general, concatenating word embeddings isn't a good way to represent word sequence.

A better way is to build a recurrent model. But first, let's extract the word embeddings for the 4 words as before and look at their similarity.


```python
embeddings = model2.get_layer(index=0).get_weights()[0] #word embeddings

emb_love = embeddings[tokenizer.word_index["love"]]
emb_like = embeddings[tokenizer.word_index["like"]]
emb_lukewarm = embeddings[tokenizer.word_index["lukewarm"]]
emb_bad = embeddings[tokenizer.word_index["bad"]]

print("love vs. like =", cos_sim(emb_love, emb_like))
print("love vs. lukewarm =", cos_sim(emb_love, emb_lukewarm))
print("love vs. bad =", cos_sim(emb_love, emb_bad))
print("lukewarm vs. bad =", cos_sim(emb_lukewarm, emb_bad))
```

    love vs. like = -0.29512215
    love vs. lukewarm = -0.10457008
    love vs. bad = -0.63287705
    lukewarm vs. bad = 0.1924367


Now, let's try to build an LSTM model. After the embeddings layer, the LSTM layer will process the words one at a time, and compute the next state (dimension for the hidden state = 10 in this case). The output of the LSTM layer is the final state, produced after processing the last word, and that will be fed to the output layer.


```python
from keras.layers import LSTM

#word order preserved with this architecture
model3 = Sequential(name="lstm")
model3.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model3.add(LSTM(10))
model3.add(layers.Dense(1, activation='sigmoid'))
model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model3.summary()
```

    Model: "lstm"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 30, 10)            18110     
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 10)                840       
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 18,961
    Trainable params: 18,961
    Non-trainable params: 0
    _________________________________________________________________


Let's train the LSTM model and see the test performance.


```python
model3.fit(xseq_train, y_train, epochs=20, verbose=True, validation_data=(xseq_dev, y_dev), batch_size=10)

loss, accuracy = model3.evaluate(xseq_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
```

    /Users/laujh/.pyenv/versions/3.6.9/lib/python3.6/site-packages/tensorflow_core/python/framework/indexed_slices.py:433: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "


    Train on 800 samples, validate on 100 samples
    Epoch 1/20
    800/800 [==============================] - 2s 2ms/step - loss: 0.6893 - accuracy: 0.5650 - val_loss: 0.7060 - val_accuracy: 0.4400
    Epoch 2/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.6838 - accuracy: 0.5650 - val_loss: 0.7292 - val_accuracy: 0.4400
    Epoch 3/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.6749 - accuracy: 0.5675 - val_loss: 0.7452 - val_accuracy: 0.4400
    Epoch 4/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.4584 - accuracy: 0.7975 - val_loss: 0.5791 - val_accuracy: 0.7300
    Epoch 5/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.2807 - accuracy: 0.9175 - val_loss: 0.6209 - val_accuracy: 0.7600
    Epoch 6/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.1843 - accuracy: 0.9525 - val_loss: 0.4598 - val_accuracy: 0.7800
    Epoch 7/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.1304 - accuracy: 0.9700 - val_loss: 0.4873 - val_accuracy: 0.8000
    Epoch 8/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0824 - accuracy: 0.9837 - val_loss: 0.5476 - val_accuracy: 0.8000
    Epoch 9/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0537 - accuracy: 0.9925 - val_loss: 0.8906 - val_accuracy: 0.7500
    Epoch 10/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0501 - accuracy: 0.9912 - val_loss: 0.7342 - val_accuracy: 0.7900
    Epoch 11/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0317 - accuracy: 0.9962 - val_loss: 0.6869 - val_accuracy: 0.8300
    Epoch 12/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0282 - accuracy: 0.9962 - val_loss: 0.8365 - val_accuracy: 0.8000
    Epoch 13/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0263 - accuracy: 0.9962 - val_loss: 0.8690 - val_accuracy: 0.8000
    Epoch 14/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0253 - accuracy: 0.9962 - val_loss: 0.9037 - val_accuracy: 0.8000
    Epoch 15/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0246 - accuracy: 0.9962 - val_loss: 0.9340 - val_accuracy: 0.8000
    Epoch 16/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0241 - accuracy: 0.9962 - val_loss: 0.9678 - val_accuracy: 0.8000
    Epoch 17/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0234 - accuracy: 0.9962 - val_loss: 1.0190 - val_accuracy: 0.8000
    Epoch 18/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0342 - accuracy: 0.9950 - val_loss: 1.0346 - val_accuracy: 0.7700
    Epoch 19/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0314 - accuracy: 0.9925 - val_loss: 1.1278 - val_accuracy: 0.7700
    Epoch 20/20
    800/800 [==============================] - 1s 1ms/step - loss: 0.0442 - accuracy: 0.9900 - val_loss: 0.8666 - val_accuracy: 0.7900
    Testing Accuracy:  0.7700


You should notice that the training is quite a bit slower, and that's because now the model has to process the sequence one word at a time. But the results should be better!

And lastly, let's extract the embeddings and look at the their similarity.


```python
embeddings = model3.get_layer(index=0).get_weights()[0] #word embeddings

emb_love = embeddings[tokenizer.word_index["love"]]
emb_like = embeddings[tokenizer.word_index["like"]]
emb_lukewarm = embeddings[tokenizer.word_index["lukewarm"]]
emb_bad = embeddings[tokenizer.word_index["bad"]]

print("love vs. like =", cos_sim(emb_love, emb_like))
print("love vs. lukewarm =", cos_sim(emb_love, emb_lukewarm))
print("love vs. bad =", cos_sim(emb_love, emb_bad))
print("lukewarm vs. bad =", cos_sim(emb_lukewarm, emb_bad))
```

    love vs. like = 0.42321843
    love vs. lukewarm = -0.95175457
    love vs. bad = -0.98218477
    lukewarm vs. bad = 0.93053085



```python

```
