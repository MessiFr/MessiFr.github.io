# *N*-gram Language Models

In this notebook, we'll be building bigram *n*-gram language models from scratch. The first part of building a language model is collecting counts from corpora. We'll do some preprocessing first, by lowercasing everything and add `<s>` (start) and `</s>` (end) symbols at the beginning and end of each sentence. For bigrams, we are using dictionaries of dictionaries with the strings as keys, which is a convenient though not particularly memory efficient way to represent things. We will use the unigram counts later for doing smoothing.


```python
from collections import defaultdict
from collections import Counter


def convert_sentence(sentence):
    return ["<s>"] + [w.lower() for w in sentence] + ["</s>"]

def get_counts(sentences):
    bigram_counts = defaultdict(Counter)
    unigram_counts = Counter()
    start_count = 0  # "<s>" counts: need these for bigram probs

    # collect initial unigram statistics
    for sentence in sentences:
        sentence = convert_sentence(sentence)
        for word in sentence[1:]: # from 1, so we don't generate the <s> token
            unigram_counts[word] += 1
        start_count += 1

    # collect bigram counts
    for sentence in sentences:
        sentence = convert_sentence(sentence)
        # generate a list of bigrams
        bigram_list = zip(sentence[:-1], sentence[1:])
        # iterate over bigrams
        for bigram in bigram_list:
            first, second = bigram
            bigram_counts[first][second] += 1
            
    token_count = float(sum(unigram_counts.values()))
    return unigram_counts, bigram_counts, start_count, token_count
```


```python
sentences = [['I', 'want', 'to', 'have', 'dinner'], ['hello', 'to', 'world']]

unigram_counts, bigram_counts, start_count, token_count = get_counts(sentences)
```


```python
unigram_counts
# bigram_counts
# start_count
# token_count
```




    Counter({'i': 1,
             'want': 1,
             'to': 2,
             'have': 1,
             'dinner': 1,
             '</s>': 2,
             'hello': 1,
             'world': 1})



Once we have counts, we can use them to generate sentences. Here we use [numpy.random.choice](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html), which allows us to randomly choose from a list based on a corresponding list of probabilities, which we calculate by normalizing the raw counts. We start with &lt;s&gt;, and generate the next word given the bigram counts which begin with &lt;s&gt;, and then use that word to generate the next word, etc. It stops when it generates an &lt;/s&gt;. We return a string with some fixes to make the sentence look a proper sentence.


```python
from numpy.random import choice 

def generate_sentence(bigram_counts):
    current_word = "<s>"
    sentence = [current_word]
    while current_word != "</s>":
        # get counts for previous word
        prev_word = current_word
        prev_word_counts = bigram_counts[prev_word]
        # obtain bigram probability distribution given the previous word
        bigram_probs = []
        total_counts = float(sum(prev_word_counts.values()))
        for word in prev_word_counts:
            bigram_probs.append(prev_word_counts[word] / total_counts)
        # sample the next word
        current_word = choice(list(prev_word_counts.keys()), p=bigram_probs)
        sentence.append(current_word)

    # get rid of start and end of sentence tokens
    sentence = " ".join(sentence[1:-1])
    return sentence
        
            
```


```python
generate_sentence()
```

Now let's try out our *n*-gram driven sentence generator on samples from two corpora: the Penn Treebank, and some out-of-copyright English literature from Project Gutenberg:


```python
import nltk
from nltk.corpus import gutenberg, treebank
nltk.download('gutenberg')
nltk.download('treebank')


gutenberg_unigrams, gutenberg_bigrams, gutenberg_start_count, gutenberg_token_count = get_counts(gutenberg.sents())
print("Gutenberg")
for i in range(1,6):
    print("Sentence %d" % i)
    print(generate_sentence(gutenberg_bigrams))
    
treebank_unigrams, treebank_bigrams, treebank_start_count, treebank_token_count = get_counts(treebank.sents())
print("Treebank")
for i in range(1,6):
    print("Sentence %d" % i)
    print(generate_sentence(treebank_bigrams))
```

    [nltk_data] Downloading package gutenberg to
    [nltk_data]     /Users/messifr/nltk_data...
    [nltk_data]   Package gutenberg is already up-to-date!
    [nltk_data] Downloading package treebank to
    [nltk_data]     /Users/messifr/nltk_data...
    [nltk_data]   Package treebank is already up-to-date!


    Gutenberg
    Sentence 1
    i will be unjust unto you know me not , but as a little space of the world behinde strooke him that if my father in the sand , let us .
    Sentence 2
    your grace of a couch , and is most noble brutus .
    Sentence 3
    that with thy hand and miss fairfax .-- on the blood , which the lord ; the happier effect increased the sun , ladies .
    Sentence 4
    amen , if they that made it was in the laughter , in his brow of old stand in working ?
    Sentence 5
    no monuments , go on the passovers for it is so abruptly by his chair before by the set down , was not to receive our body admitted no master and deadly wound it just before thy gravestone , he gathered an inheritance of being in thy name , for menstealers , that but live in question .
    Treebank
    Sentence 1
    not alike , of the s&p 500 widget , * by this abortion .
    Sentence 2
    `` they do all domestic affairs of we 've got *-1 hitting a student looking to slow *-1 to 1,500 spaces , haut-brion , factory *ich*-2 in the czech dam was rejected reports , an effective headcount-control program trading , but also , the funds , 1996 , *-1 publicly on our efforts * providing compliance and general manager says *t*-1 is n't enough rules of valhi rose modestly as a premium .
    Sentence 3
    the sale of hope and alcoholism .
    Sentence 4
    corestates financial adviser , drugs or retail sales are here , `` we had *-2 to be available pursuant to lose several so-called weak in most political consultants , which *t*-25 has been striving -- 271,124 -- for imports of a stanford achievement test procedures .
    Sentence 5
    the market 's stock price changes in legislation 0 income surged 7 % of the broader question the filing levels .


Generally, we can see some local coherence but most of these sentences are complete nonsense. Across the two corpora, the sentences are noticeably different, it's very obvious that the model from Project Gutenberg is trained on literature, whereas the Penn Treebank data is financial. For the latter, there are some strange tokens (those starting with \*) we should probably have filtered out.

Using language models to generate sentences is fun but not very useful. A more practical application is the ability to assign a probability to a sentence. In order to do that for anything but toy examples, however, we will need to smooth our models so it doesn't assign a zero probability to the sentence whenever it sees a bigram. Here, we'll test two fairly simple smoothing techniques, add-*k* smoothing and interpolated smoothing. In both cases, we will calculate the log probability, to avoid working with very small numbers. The functions below give the probability for a single word at index i in a sentence.

Notice that interpolation is implemented using 3 probabilities: the bigram, the unigram and a "zerogram" probability. The "zerogram" actually refers to the probability of any word appearing. We need this extra probability in order to account for out-of-vocabulary (OOVs) words, which result in zero probability for both bigrams and unigrams. Estimating the probability of OOVs is a general problem: here we use an heuristic that uses a uniform distribution over all words in the vocabulary (1 / |V|).


```python
import math

def get_log_prob_addk(prev_word, word, unigram_counts, bigram_counts, k):
    sm_bigram_counts = bigram_counts[prev_word][word] + k
    sm_unigram_counts = unigram_counts[prev_word] + k*len(unigram_counts)
    return math.log(sm_bigram_counts / sm_unigram_counts)

def get_log_prob_interp(prev_word, word, unigram_counts, bigram_counts, start_count, token_count, lambdas):
    bigram_lambda = lambdas[0]
    unigram_lambda = lambdas[1]
    zerogram_lambda = 1 - lambdas[0] - lambdas[1]
    
    # start by getting bigram probability
    sm_bigram_counts = bigram_counts[prev_word][word] * bigram_lambda
    if sm_bigram_counts == 0.0:
        interp_bigram_counts = 0
    else:
        if prev_word == "<s>":
            u_counts = start_count
        else:
            u_counts = unigram_counts[prev_word]
        interp_bigram_counts = sm_bigram_counts / float(u_counts)
        
    # unigram probability
    interp_unigram_counts = (unigram_counts[word] / token_count) * unigram_lambda
    
    # "zerogram" probability: this is to account for out-of-vocabulary words
    # this is just 1 / |V|
    vocab_size = len(unigram_counts)
    interp_zerogram_counts = (1 / float(vocab_size)) * zerogram_lambda
    
    return math.log(interp_bigram_counts + interp_unigram_counts + interp_zerogram_counts)

```

Extending this to calculate the probability of an entire sentence is trivial.


```python
def get_sent_log_prob_addk(sentence, unigram_counts, bigram_counts, start_count, token_count, k):
    sentence = convert_sentence(sentence)
    bigram_list = zip(sentence[:-1], sentence[1:])
    return sum([get_log_prob_addk(prev_word, 
                                  word, 
                                  unigram_counts, 
                                  bigram_counts, 
                                  k) for prev_word, word in bigram_list])

def get_sent_log_prob_interp(sentence, unigram_counts, bigram_counts, start_count, token_count, lambdas):
    sentence = convert_sentence(sentence)
    bigram_list = zip(sentence[:-1], sentence[1:])
    return sum([get_log_prob_interp(prev_word, 
                                    word, 
                                    unigram_counts, 
                                    bigram_counts,
                                    start_count,
                                    token_count, 
                                    lambdas) for prev_word, word in bigram_list])
    
sentence = "revenue increased last quarter .".split()
print(get_sent_log_prob_addk(sentence, gutenberg_unigrams, gutenberg_bigrams, gutenberg_start_count,
                             gutenberg_token_count, 0.05))
print(get_sent_log_prob_interp(sentence, 
                               gutenberg_unigrams, 
                               gutenberg_bigrams, 
                               gutenberg_start_count, 
                               gutenberg_token_count, 
                               (0.8, 0.19)))
print(get_sent_log_prob_addk(sentence, treebank_unigrams, treebank_bigrams, treebank_start_count,
                             treebank_token_count, 0.05))
print(get_sent_log_prob_interp(sentence, 
                               treebank_unigrams, 
                               treebank_bigrams, 
                               treebank_start_count, 
                               treebank_token_count, 
                               (0.8, 0.19)))
```

    -48.460406447015146
    -49.538820071127226
    -39.776378681452364
    -39.245480234555636


The output for our sample sentence looks reasonable, in particular using the Treebank model results in a noticeably higher probability, which is what we'd expect given the input sentence. The differences in probability between the different smoothing techniques is more modest (though keep in mind this is a logrithmic scale). Now, let's use perplexity to evaluate different smoothing techniques at the level of the corpus. For this, we'll use the Brown corpus again, dividing it up randomly into a training set and a test set based on an 80/20 split.


```python
from nltk.corpus import brown
from random import shuffle
nltk.download('brown')

sents = list(brown.sents())
shuffle(sents)
cutoff = int(0.8*len(sents))
training_set = sents[:cutoff]
test_set = [[word.lower() for word in sent] for sent in sents[cutoff:]]

brown_unigrams, brown_bigrams, brown_start_count, brown_token_count = get_counts(training_set)
```

    [nltk_data] Downloading package brown to /Users/messifr/nltk_data...
    [nltk_data]   Package brown is already up-to-date!



```python

```

Since our probabilities are in log space, we will calculate perplexity in log space as well, then take the exponential at the end

$$PP(W) = \sqrt[m]{\frac{1}{P(W)}}$$

$$\log{PP(W)} = -\frac{1}{m} \log{P(W)}$$


```python
def calculate_perplexity(sentences, unigram_counts, bigram_counts, start_count, 
                         token_count, smoothing_function, parameter):
    total_log_prob = 0
    test_token_count = 0
    for sentence in sentences:
        test_token_count += len(sentence) + 1 # have to consider the end token
        total_log_prob += smoothing_function(sentence,
                                             unigram_counts,
                                             bigram_counts,
                                             start_count,
                                             token_count,
                                             parameter)
    return math.exp(-total_log_prob / test_token_count)

                
```

Let's see how our two smoothing techniques do with a range of possible parameter values 


```python
print("add k")
for k in [0.0001,0.001,0.01, 0.05,0.2,1.0]:
    print(k)
    print(calculate_perplexity(test_set,
                               brown_unigrams,
                               brown_bigrams,
                               brown_start_count,
                               brown_token_count,
                               get_sent_log_prob_addk,
                               k))
print("interpolation")
for bigram_lambda in [0.98,0.95,0.75,0.5,0.25,0.001]:
    unigram_lambda = 0.99 - bigram_lambda
    lambdas = (bigram_lambda, unigram_lambda)
    print(lambdas) 
    print(calculate_perplexity(test_set, 
                               brown_unigrams, 
                               brown_bigrams,
                               brown_start_count,
                               brown_token_count, 
                               get_sent_log_prob_interp, 
                               lambdas))
```

    add k
    0.0001
    747.860084740397
    0.001
    608.1037030464857
    0.01
    715.7003595336809
    0.05
    1033.197366149906
    0.2
    1681.6595463271226
    1.0
    3486.8451342027374
    interpolation
    (0.98, 0.010000000000000009)
    762.7719951864839
    (0.95, 0.040000000000000036)
    585.2040019562442
    (0.75, 0.24)
    425.4470331471468
    (0.5, 0.49)
    422.80505963590605
    (0.25, 0.74)
    497.62931007291564
    (0.001, 0.989)
    980.3820682875372


Our results indicate that, with regards to perplexity, interpolation is generally better than add k. Very low (though not too low) k is preferred for add k, wheres our best lambdas is in the middle of the range, though apparently with a small preference for more weight on the bigram probability, which makes sense.

From the basis given here, you can try playing around with some of the other corpora in NLTK and see if you get similar results. You could also implement a trigram model, or another kind of smoothing, to see if you can get better perplexity scores.


```python

```
