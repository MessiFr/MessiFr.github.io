# Part of Speech Tagging

Several corpora with manual part of speech (POS) tagging are included in NLTK. For this exercise, we'll use a sample of the Penn Treebank corpus, a collection of Wall Street Journal articles. We can access the part-of-speech information for either the Penn Treebank or the Brown as follows. We use sentences here because that is the preferred representation for doing POS tagging.


```python
import nltk
from nltk.corpus import treebank, brown
nltk.download('treebank')

print(treebank.tagged_sents()[0])
print(brown.tagged_sents()[0])
```

    [nltk_data] Downloading package treebank to
    [nltk_data]     C:\Users\Jason\AppData\Roaming\nltk_data...
    [nltk_data]   Package treebank is already up-to-date!
    [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
    [('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL'), ('said', 'VBD'), ('Friday', 'NR'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'IN'), ("Atlanta's", 'NP$$'), ('recent', 'JJ'), ('primary', 'NN'), ('election', 'NN'), ('produced', 'VBD'), ('``', '``'), ('no', 'AT'), ('evidence', 'NN'), ("''", "''"), ('that', 'CS'), ('any', 'DTI'), ('irregularities', 'NNS'), ('took', 'VBD'), ('place', 'NN'), ('.', '.')]


In NLTK, word/tag pairs are stored as tuples, the transformation from the plain text "word/tag" representation to the python data types is done by the corpus reader.

The two corpora do not have the same tagset; the Brown was tagged with a more fine-grained tagset: for instance, instead of "DT" (determiner) as in the Penn Treebank, the word "the" is tagged as "AT" (article, which is a kind of determiner). We can convert them both to the Universal tagset.


```python
import nltk
nltk.download('universal_tagset')
print(treebank.tagged_sents(tagset="universal")[0])
print(brown.tagged_sents(tagset="universal")[0])
```

Now, let's create a basic unigram POS tagger. First, we need to collect POS distributions for each word. We'll do this (somewhat inefficiently) using a dictionary of dictionaries. Note that we are using the PTB tag set from here on.


```python
from collections import defaultdict

POS_dict = defaultdict(dict)
for word_pos_pair in treebank.tagged_words():
    word = word_pos_pair[0].lower()
    POS = word_pos_pair[1]
    POS_dict[word][POS] = POS_dict[word].get(POS,0) + 1
```

Let's look at some words which appear with multiple POS, and their POS counts:


```python
for word in list(POS_dict.keys())[:100]:
    if len(POS_dict[word]) > 1:
        print(word)
        print(POS_dict[word])
```

Common ambiguities that we see here are between nouns and verbs (<i>increase</i>, <i>refunding</i>, <i>reports</i>), and, among verbs, between past tense and past participles (<i>contributed</i>, <i>reported</i>, <i>climbed</i>).

To create an actual tagger, we just need to pick the most common tag for each


```python
tagger_dict = {}
for word in POS_dict:
    tagger_dict[word] = max(POS_dict[word],key=lambda x: POS_dict[word][x])

def tag(sentence):
    return [(word,tagger_dict.get(word,"NN")) for word in sentence]

example_sentence = """You better start swimming or sink like a stone , cause the times they are a - changing .""".split() 
print(tag(example_sentence))
```

Though we'd probably want some better handling of capitalized phrases (backing off to NNP, or using the statistics for the lower-case token), and there are a few other obvious errors, generally it's not too bad. 

NLTK has built-in support for n-gram taggers; Let's build unigram and bigram taggers, and test their performance. First we need to split our corpus into training and testing


```python
size = int(len(treebank.tagged_sents()) * 0.9)
train_sents = treebank.tagged_sents()[:size]
test_sents = treebank.tagged_sents()[size:]

```

Let's first compare a unigram and bigram tagger. All NLTK taggers have an evaluate method which prints out the accuracy on some test set.


```python
from nltk import UnigramTagger, BigramTagger

unigram_tagger = UnigramTagger(train_sents)
bigram_tagger = BigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))
print(unigram_tagger.tag(example_sentence))
print(bigram_tagger.evaluate(test_sents))
print(bigram_tagger.tag(example_sentence))
```

The unigram tagger does way better. The reason is sparsity, the bigram tagger doesn't have counts for many of the word/tag context pairs; what's worse, once it can't tag something, it fails catastrophically for the rest of the sentence tag, because it has no counts at all for missing tag contexts. We can fix this by adding backoffs, including the default tagger with just tags everything as NN


```python
from nltk import DefaultTagger

default_tagger = DefaultTagger("NN")
unigram_tagger = UnigramTagger(train_sents,backoff=default_tagger)
bigram_tagger = BigramTagger(train_sents,backoff=unigram_tagger)

print(bigram_tagger.evaluate(test_sents))
print(bigram_tagger.tag(example_sentence))
```

We see a 3% increase in performance from adding the bigram information on top of the unigram information.

NLTK has interfaces to the Brill tagger (nltk.tag.brill) and also pre-build, state-of-the-art sequential POS tagging models, for instance the Stanford POS tagger (StanfordPOSTagger), which is what you should use if you actually need high-quality POS tagging for some application; if you are working on a computer with the <b>Stanford CoreNLP</b> tools installed and NLTK set up to use them (this is the case for the lab computers where workshops are held), the below code should work. If not, see the documentation <a href="https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software">here</a> under "Stanford Tagger, NER, Tokenizer and Parser" to install them. 


```python
from nltk import StanfordPOSTagger
nltk.download('stanford-tagger')

stanford_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
print(stanford_tagger.tag(brown.sents()[1]))
```


```python

```
