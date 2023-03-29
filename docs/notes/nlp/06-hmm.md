# Hidden Markov Models in python

Here we'll show how the Viterbi algorithm works for HMMs, assuming we have a trained model to start with. We will use the example in the JM3 book (Ch. 8.4.6).


```python
import numpy as np
```

Initialise the model parameters based on the example from the slides/book (values taken from figure). Notice that here we explicitly split the initial probabilities "pi" from the transition matrix "A".


```python
tags = NNP, MD, VB, JJ, NN, RB, DT = 0, 1, 2, 3, 4, 5, 6
tag_dict = {0: 'NNP',
           1: 'MD',
           2: 'VB',
           3: 'JJ',
           4: 'NN',
           5: 'RB',
           6: 'DT'}
words = Janet, will, back, the, bill = 0, 1, 2, 3, 4

A = np.array([
    [0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025],
    [0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041],
    [0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231],
    [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036],
    [0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068],
    [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
    [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]
    ])

pi = np.array([0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026])

B = np.array([
    [0.000032, 0, 0, 0.000048, 0],
    [0, 0.308431, 0, 0, 0],
    [0, 0.000028, 0.000672, 0, 0.000028],
    [0, 0, 0.000340, 0.000097, 0],
    [0, 0.000200, 0.000223, 0.000006, 0.002337],
    [0, 0, 0.010446, 0, 0],
    [0, 0, 0, 0.506099, 0]
    ])

```

Now we'll code the Viterbi algorithm. It keeps a store of two components, the best scores to reach a state at a give time, and the last step of the path to get there. Scores alpha are initialised to -inf to denote that we haven't set them yet. 


```python
alpha = np.zeros((len(tags), len(words))) # states x time steps
alpha[:,:] = float('-inf')
backpointers = np.zeros((len(tags), len(words)), 'int')
```

The base case for the recursion sets the starting state probs based on pi and generating the observation. (Note: we also change Numpy precision when printing for better viewing)


```python
# base case, time step 0
alpha[:, 0] = pi * B[:,Janet]
np.set_printoptions(precision=2)
print(alpha)
```

    [[8.85e-06     -inf     -inf     -inf     -inf]
     [0.00e+00     -inf     -inf     -inf     -inf]
     [0.00e+00     -inf     -inf     -inf     -inf]
     [0.00e+00     -inf     -inf     -inf     -inf]
     [0.00e+00     -inf     -inf     -inf     -inf]
     [0.00e+00     -inf     -inf     -inf     -inf]
     [0.00e+00     -inf     -inf     -inf     -inf]]


Now for the recursive step, where we maximise over incoming transitions reusing the best incoming score, computed above.


```python
# time step 1
for t1 in tags:
    for t0 in tags:
        score = alpha[t0, 0] * A[t0, t1] * B[t1, will]
        if score > alpha[t1, 1]:
            alpha[t1, 1] = score
            backpointers[t1, 1] = t0
print(alpha)
```

    [[8.85e-06 0.00e+00     -inf     -inf     -inf]
     [0.00e+00 3.00e-08     -inf     -inf     -inf]
     [0.00e+00 2.23e-13     -inf     -inf     -inf]
     [0.00e+00 0.00e+00     -inf     -inf     -inf]
     [0.00e+00 1.03e-10     -inf     -inf     -inf]
     [0.00e+00 0.00e+00     -inf     -inf     -inf]
     [0.00e+00 0.00e+00     -inf     -inf     -inf]]


Note that the running maximum for any incoming state (t0) is maintained in alpha[1,t1], and the winning state is stored in addition, as a backpointer. 

Repeat with the next observations. (We'd do this as a loop over positions in practice.)


```python
# time step 2
for t2 in tags:
    for t1 in tags:
        score = alpha[t1, 1] * A[t1, t2] * B[t2, back]
        if score > alpha[t2, 2]:
            alpha[t2, 2] = score
            backpointers[t2, 2] = t1
print(alpha)

# time step 3
for t3 in tags:
    for t2 in tags:
        score = alpha[t2, 2] * A[t2, t3] * B[t3, the]
        if score > alpha[t3, 3]:
            alpha[t3, 3] = score
            backpointers[t3, 3] = t2
print(alpha)

# time step 4
for t4 in tags:
    for t3 in tags:
        score = alpha[t3, 3] * A[t3, t4] * B[t4, bill]
        if score > alpha[t4, 4]:
            alpha[t4, 4] = score
            backpointers[t4, 4] = t3
print(alpha)
```

    [[8.85e-06 0.00e+00 0.00e+00     -inf     -inf]
     [0.00e+00 3.00e-08 0.00e+00     -inf     -inf]
     [0.00e+00 2.23e-13 1.61e-11     -inf     -inf]
     [0.00e+00 0.00e+00 5.11e-15     -inf     -inf]
     [0.00e+00 1.03e-10 5.36e-15     -inf     -inf]
     [0.00e+00 0.00e+00 5.33e-11     -inf     -inf]
     [0.00e+00 0.00e+00 0.00e+00     -inf     -inf]]
    [[8.85e-06 0.00e+00 0.00e+00 2.49e-17     -inf]
     [0.00e+00 3.00e-08 0.00e+00 0.00e+00     -inf]
     [0.00e+00 2.23e-13 1.61e-11 0.00e+00     -inf]
     [0.00e+00 0.00e+00 5.11e-15 5.23e-16     -inf]
     [0.00e+00 1.03e-10 5.36e-15 5.94e-18     -inf]
     [0.00e+00 0.00e+00 5.33e-11 0.00e+00     -inf]
     [0.00e+00 0.00e+00 0.00e+00 1.82e-12     -inf]]
    [[8.85e-06 0.00e+00 0.00e+00 2.49e-17 0.00e+00]
     [0.00e+00 3.00e-08 0.00e+00 0.00e+00 0.00e+00]
     [0.00e+00 2.23e-13 1.61e-11 0.00e+00 1.02e-20]
     [0.00e+00 0.00e+00 5.11e-15 5.23e-16 0.00e+00]
     [0.00e+00 1.03e-10 5.36e-15 5.94e-18 2.01e-15]
     [0.00e+00 0.00e+00 5.33e-11 0.00e+00 0.00e+00]
     [0.00e+00 0.00e+00 0.00e+00 1.82e-12 0.00e+00]]


Now read of the best final state:


```python
t4 = np.argmax(alpha[:, 4])
print(tag_dict[t4])
```

    NN


We need to work out the rest of the path which is the best way to reach the final state, t2. We can work this out by taking a step backwards looking at the best incoming edge, i.e., as stored in the backpointers.


```python
t3 = backpointers[t4, 4]
print(tag_dict[t3])
```

    DT


Repeat this until we reach the start of the sequence.


```python
t2 = backpointers[t3, 3]
print(tag_dict[t2])
t1 = backpointers[t2, 2]
print(tag_dict[t1])
t0 = backpointers[t1, 1]
print(tag_dict[t0])
```

    VB
    MD
    NNP


Phew. The best state sequence is t = [NNP MD VB DT NN]

## Formalising things

Now we can put this all into a function to handle arbitrary length inputs 


```python
def viterbi(params, words):
    pi, A, B = params
    N = len(words)
    T = pi.shape[0]
    
    alpha = np.zeros((T, N))
    alpha[:, :] = float('-inf')
    backpointers = np.zeros((T, N), 'int')
    
    # base case
    alpha[:, 0] = pi * B[:, words[0]]
    
    # recursive case
    for w in range(1, N):
        for t2 in range(T):
            for t1 in range(T):
                score = alpha[t1, w-1] * A[t1, t2] * B[t2, words[w]]
                if score > alpha[t2, w]:
                    alpha[t2, w] = score
                    backpointers[t2, w] = t1
    
    # now follow backpointers to resolve the state sequence
    output = []
    output.append(np.argmax(alpha[:, N-1]))
    for i in range(N-1, 0, -1):
        output.append(backpointers[output[-1], i])
    
    return list(reversed(output)), np.max(alpha[:, N-1])
```

Let's test the method on the same input, and a longer input observation sequence. Notice that we are using only 5 words as the vocabulary so we have to restrict tests to sentences containing only these words.


```python
output, score = viterbi((pi, A, B), [Janet, will, back, the, bill])
print([tag_dict[o] for o in output])
print(score)
```

    ['NNP', 'MD', 'VB', 'DT', 'NN']
    2.013570710221386e-15



```python
output, score = viterbi((pi, A, B), [Janet, will, back, the, Janet, back, bill])
print([tag_dict[o] for o in output])
print(score)
```

    ['NNP', 'MD', 'VB', 'DT', 'NNP', 'NN', 'NN']
    2.4671007551487516e-26


## Exhaustive method

Let's verify that we've done the above algorithm correctly by implementing exhaustive search, which forms the cross-product of states^M.


```python
from itertools import product

def exhaustive(params, words):
    pi, A, B = params
    N = len(words)
    T = pi.shape[0]
    
    # track the running best sequence and its score
    best = (None, float('-inf'))
    # loop over the cartesian product of |states|^M
    for ss in product(range(T), repeat=N):
        # score the state sequence
        score = pi[ss[0]] * B[ss[0], words[0]]
        for i in range(1, N):
            score *= A[ss[i-1], ss[i]] * B[ss[i], words[i]]
        # update the running best
        if score > best[1]:
            best = (ss, score)
            
    return best
```


```python
output, score = exhaustive((pi, A, B), [Janet, will, back, the, bill])
print([tag_dict[o] for o in tag_dict])
print(score)
```

    ['NNP', 'MD', 'VB', 'JJ', 'NN', 'RB', 'DT']
    2.0135707102213855e-15



```python
output, score = exhaustive((pi, A, B), [Janet, will, back, the, Janet, back, bill])
print([tag_dict[o] for o in tag_dict])
print(score)
```

    ['NNP', 'MD', 'VB', 'JJ', 'NN', 'RB', 'DT']
    2.4671007551487507e-26


Yay, it got the same results as before. Note that the exhaustive method is practical on anything beyond toy data due to the nasty cartesian product. But it is worth doing to verify the Viterbi code above is getting the right results. 

## Supervised training

Let's train the HMM parameters on the Penn Treebank, using the sample from NLTK. Note that this is a small fraction of the treebank, so we shouldn't expect great performance of our method trained only on this data.


```python
from nltk.corpus import treebank
```


```python
corpus = treebank.tagged_sents()
print(corpus)
```

    [[('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')], [('Mr.', 'NNP'), ('Vinken', 'NNP'), ('is', 'VBZ'), ('chairman', 'NN'), ('of', 'IN'), ('Elsevier', 'NNP'), ('N.V.', 'NNP'), (',', ','), ('the', 'DT'), ('Dutch', 'NNP'), ('publishing', 'VBG'), ('group', 'NN'), ('.', '.')], ...]


We have to first map words and tags to numbers for compatibility with the above methods.


```python
word_numbers = {}
tag_numbers = {}

num_corpus = []
for sent in corpus:
    num_sent = []
    for word, tag in sent:
        wi = word_numbers.setdefault(word.lower(), len(word_numbers))
        ti = tag_numbers.setdefault(tag, len(tag_numbers))
        num_sent.append((wi, ti))
    num_corpus.append(num_sent)
    
word_names = [None] * len(word_numbers)
for word, index in word_numbers.items():
    word_names[index] = word
tag_names = [None] * len(tag_numbers)
for tag, index in tag_numbers.items():
    tag_names[index] = tag
```

Now let's hold out the last few sentences for testing, so that they are unseen during training and give a more reasonable estimate of accuracy on fresh text.


```python
training = num_corpus[:-10] # reserve the last 10 sentences for testing
testing = num_corpus[-10:]
```

Next we compute relative frequency estimates based on the observed tag and word counts in the training set. Note that smoothing is important, here we add a small constant to all counts. 


```python
S = len(tag_numbers)
V = len(word_numbers)

# initalise
eps = 0.1
pi = eps * np.ones(S)
A = eps * np.ones((S, S))
B = eps * np.ones((S, V))

# count
for sent in training:
    last_tag = None
    for word, tag in sent:
        B[tag, word] += 1
        # bug fixed here 27/3/17; test was incorrect 
        if last_tag == None:
            pi[tag] += 1
        else:
            A[last_tag, tag] += 1
        last_tag = tag
        
# normalise
pi /= np.sum(pi)
for s in range(S):
    B[s,:] /= np.sum(B[s,:])
    A[s,:] /= np.sum(A[s,:])
```

Now we're ready to use our Viterbi method defined above


```python
predicted, score = viterbi((pi, A, B), list(map(lambda w_t: w_t[0], testing[0])))
```


```python
print('%20s\t%5s\t%5s' % ('TOKEN', 'TRUE', 'PRED'))
for (wi, ti), pi in zip(testing[0], predicted):
    print('%20s\t%5s\t%5s' % (word_names[wi], tag_names[ti], tag_names[pi]))
```

                   TOKEN	 TRUE	 PRED
                       a	   DT	   DT
                   white	  NNP	  NNP
                   house	  NNP	  NNP
               spokesman	   NN	   NN
                    said	  VBD	  VBD
                    last	   JJ	   JJ
                    week	   NN	   NN
                    that	   IN	   IN
                     the	   DT	   DT
               president	   NN	   NN
                      is	  VBZ	  VBZ
             considering	  VBG	  VBG
                     *-1	-NONE-	-NONE-
               declaring	  VBG	  VBG
                    that	   IN	   IN
                     the	   DT	   DT
            constitution	  NNP	  NNP
              implicitly	   RB	  NNP
                   gives	  VBZ	  VBZ
                     him	  PRP	  PRP
                     the	   DT	   DT
               authority	   NN	   NN
                     for	   IN	   IN
                       a	   DT	   DT
               line-item	   JJ	   JJ
                    veto	   NN	   NN
                     *-2	-NONE-	-NONE-
                      to	   TO	   TO
                 provoke	   VB	   VB
                       a	   DT	   DT
                    test	   NN	   NN
                    case	   NN	   NN
                       .	    .	    .


Hey, not bad, only one error. Can you explain why this one might have occurred?


```python

```
