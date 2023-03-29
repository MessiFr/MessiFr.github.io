# BPE

*We use Python 3.8 for all labs in this subject.*

Train BPE on a toy text example

bpe algorithm: https://web.stanford.edu/~jurafsky/slp3/2.pdf (2.4.3)


```python
import re, collections

text = "The aims for this subject is for students to develop an understanding of the main algorithms used in natural language processing, for use in a diverse range of applications including text classification, machine translation, and question answering. Topics to be covered include part-of-speech tagging, n-gram language modelling, syntactic parsing and deep learning. The programming language used is Python, see for more information on its use in the workshops, assignments and installation at home."
# text = 'low '*5 +'lower '*2+'newest '*6 +'widest '*3

#what is this function
def get_vocab(text):
    vocab = collections.defaultdict(int)
    for word in text.strip().split():
        #note: we use the special token </w> (instead of underscore in the lecture) to denote the end of a word
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    # matches unmerged bigrams
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        # substitute unmerged pairs in each word
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


vocab = get_vocab(text)
print("Vocab =", vocab)
print('==========')
print('Tokens Before BPE')
tokens = get_tokens(vocab)
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))
print('==========')

#about 100 merges we start to see common words
num_merges = 100
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    new_token = ''.join(best)
    vocab = merge_vocab(best, vocab)
    print('Iter: {}'.format(i))
    print('Best pair: {}'.format(best))
    # add new token to the vocab
    tokens[new_token] = pairs[best]
    # deduct frequency for tokens have been merged
    tokens[best[0]] -= pairs[best]
    tokens[best[1]] -= pairs[best]
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))
    print('==========')
```

    Vocab = defaultdict(<class 'int'>, {'T h e </w>': 2, 'a i m s </w>': 1, 'f o r </w>': 4, 't h i s </w>': 1, 's u b j e c t </w>': 1, 'i s </w>': 2, 's t u d e n t s </w>': 1, 't o </w>': 2, 'd e v e l o p </w>': 1, 'a n </w>': 1, 'u n d e r s t a n d i n g </w>': 1, 'o f </w>': 2, 't h e </w>': 2, 'm a i n </w>': 1, 'a l g o r i t h m s </w>': 1, 'u s e d </w>': 2, 'i n </w>': 3, 'n a t u r a l </w>': 1, 'l a n g u a g e </w>': 3, 'p r o c e s s i n g , </w>': 1, 'u s e </w>': 2, 'a </w>': 1, 'd i v e r s e </w>': 1, 'r a n g e </w>': 1, 'a p p l i c a t i o n s </w>': 1, 'i n c l u d i n g </w>': 1, 't e x t </w>': 1, 'c l a s s i f i c a t i o n , </w>': 1, 'm a c h i n e </w>': 1, 't r a n s l a t i o n , </w>': 1, 'a n d </w>': 3, 'q u e s t i o n </w>': 1, 'a n s w e r i n g . </w>': 1, 'T o p i c s </w>': 1, 'b e </w>': 1, 'c o v e r e d </w>': 1, 'i n c l u d e </w>': 1, 'p a r t - o f - s p e e c h </w>': 1, 't a g g i n g , </w>': 1, 'n - g r a m </w>': 1, 'm o d e l l i n g , </w>': 1, 's y n t a c t i c </w>': 1, 'p a r s i n g </w>': 1, 'd e e p </w>': 1, 'l e a r n i n g . </w>': 1, 'p r o g r a m m i n g </w>': 1, 'P y t h o n , </w>': 1, 's e e </w>': 1, 'm o r e </w>': 1, 'i n f o r m a t i o n </w>': 1, 'o n </w>': 1, 'i t s </w>': 1, 'w o r k s h o p s , </w>': 1, 'a s s i g n m e n t s </w>': 1, 'i n s t a l l a t i o n </w>': 1, 'a t </w>': 1, 'h o m e . </w>': 1})
    ==========
    Tokens Before BPE
    Tokens: defaultdict(<class 'int'>, {'T': 3, 'h': 11, 'e': 39, '</w>': 73, 'a': 38, 'i': 37, 'm': 12, 's': 34, 'f': 9, 'o': 29, 'r': 22, 't': 29, 'u': 14, 'b': 2, 'j': 1, 'c': 13, 'd': 15, 'n': 45, 'v': 3, 'l': 16, 'p': 11, 'g': 22, ',': 7, 'x': 1, 'q': 1, 'w': 2, '.': 3, '-': 3, 'y': 2, 'P': 1, 'k': 1})
    Number of tokens: 31
    ==========
    Iter: 0
    Best pair: ('i', 'n')
    Tokens: defaultdict(<class 'int'>, {'T': 3, 'h': 11, 'e': 39, '</w>': 73, 'a': 38, 'i': 19, 'm': 12, 's': 34, 'f': 9, 'o': 29, 'r': 22, 't': 29, 'u': 14, 'b': 2, 'j': 1, 'c': 13, 'd': 15, 'n': 27, 'v': 3, 'l': 16, 'p': 11, 'g': 22, ',': 7, 'x': 1, 'q': 1, 'w': 2, '.': 3, '-': 3, 'y': 2, 'P': 1, 'k': 1, 'in': 18})
    Number of tokens: 32
    ==========
    Iter: 1
    Best pair: ('e', '</w>')
    Tokens: defaultdict(<class 'int'>, {'T': 3, 'h': 11, 'e': 23, '</w>': 57, 'a': 38, 'i': 19, 'm': 12, 's': 34, 'f': 9, 'o': 29, 'r': 22, 't': 29, 'u': 14, 'b': 2, 'j': 1, 'c': 13, 'd': 15, 'n': 27, 'v': 3, 'l': 16, 'p': 11, 'g': 22, ',': 7, 'x': 1, 'q': 1, 'w': 2, '.': 3, '-': 3, 'y': 2, 'P': 1, 'k': 1, 'in': 18, 'e</w>': 16})
    Number of tokens: 33
    ==========
    Iter: 2
    Best pair: ('a', 'n')
    Tokens: defaultdict(<class 'int'>, {'T': 3, 'h': 11, 'e': 23, '</w>': 57, 'a': 27, 'i': 19, 'm': 12, 's': 34, 'f': 9, 'o': 29, 'r': 22, 't': 29, 'u': 14, 'b': 2, 'j': 1, 'c': 13, 'd': 15, 'n': 16, 'v': 3, 'l': 16, 'p': 11, 'g': 22, ',': 7, 'x': 1, 'q': 1, 'w': 2, '.': 3, '-': 3, 'y': 2, 'P': 1, 'k': 1, 'in': 18, 'e</w>': 16, 'an': 11})
    Number of tokens: 34
    ==========
    Iter: 3
    Best pair: ('s', '</w>')
    Tokens: defaultdict(<class 'int'>, {'T': 3, 'h': 11, 'e': 23, '</w>': 47, 'a': 27, 'i': 19, 'm': 12, 's': 24, 'f': 9, 'o': 29, 'r': 22, 't': 29, 'u': 14, 'b': 2, 'j': 1, 'c': 13, 'd': 15, 'n': 16, 'v': 3, 'l': 16, 'p': 11, 'g': 22, ',': 7, 'x': 1, 'q': 1, 'w': 2, '.': 3, '-': 3, 'y': 2, 'P': 1, 'k': 1, 'in': 18, 'e</w>': 16, 'an': 11, 's</w>': 10})
    Number of tokens: 35
    ==========
    Iter: 4
    ...
    ==========
    Iter: 99
    Best pair: ('nat', 'u')
    Tokens: defaultdict(<class 'int'>, {'T': 1, 'h': 4, 'e': 8, '</w>': 11, 'a': 3, 'i': 2, 'm': 4, 's': 7, 'f': 1, 'o': 3, 'r': 4, 't': 6, 'u': 1, 'b': 1, 'j': 0, 'c': 4, 'd': 2, 'n': 3, 'v': 0, 'l': 4, 'p': 4, 'g': 2, ',': 0, 'x': 1, 'q': 1, 'w': 2, '.': 0, '-': 3, 'y': 2, 'P': 1, 'k': 1, 'in': 3, 'e</w>': 7, 'an': 0, 's</w>': 3, 'ing': 0, 'or': 2, 'on': 2, 'at': 1, ',</w>': 4, 'd</w>': 0, 'ion': 1, 'for': 1, 'th': 1, 'de': 2, 'ation': 1, 'for</w>': 4, 'st': 2, 'ing</w>': 2, 'in</w>': 3, 'us': 0, 'ang': 1, 'ag': 1, 'ic': 2, 'is</w>': 2, 'nt': 1, 've': 0, 'op': 2, 'of': 1, 'al': 2, 'ed</w>': 1, 'lang': 0, 'langu': 0, 'languag': 0, 'language</w>': 3, 'ss': 1, 'ing,</w>': 3, 'cl': 1, 'and</w>': 3, '.</w>': 1, 'ar': 1, 'Th': 0, 'The</w>': 2, 'ms</w>': 0, 'ec': 1, 't</w>': 1, 'nts</w>': 1, 'to': 0, 'to</w>': 2, 'ding</w>': 1, 'of</w>': 2, 'the</w>': 2, 'ma': 1, 'used</w>': 2, 'pr': 0, 'pro': 2, 'use</w>': 2, 'ver': 2, 'ication': 2, 'incl': 0, 'inclu': 2, 'ass': 0, 'assi': 2, 'ans': 2, 'lation': 2, 'ing.</w>': 2, 'par': 2, 'gr': 0, 'gra': 0, 'gram': 2, 'me': 2, 'ai': 0, 'aims</w>': 1, 'this</w>': 1, 'su': 0, 'sub': 0, 'subj': 0, 'subjec': 0, 'subject</w>': 1, 'stu': 0, 'stude': 0, 'students</w>': 1, 'deve': 0, 'devel': 0, 'develop': 0, 'develop</w>': 1, 'an</w>': 1, 'un': 0, 'unde': 0, 'under': 0, 'underst': 0, 'understan': 0, 'understanding</w>': 1, 'main</w>': 1, 'alg': 0, 'algor': 0, 'algori': 0, 'algorith': 0, 'algorithms</w>': 1, 'nat': 0, 'natu': 1})
    Number of tokens: 131
    ==========


After training, used the BPE dictionaries to tokenise sentences


```python
def get_vocab_tokenization(vocab):
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return vocab_tokenization

def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)

def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token] * len(string)

    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token)
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        # if no match found in the string, go to next token
        if len(matched_positions) == 0:
            continue
        # collect end position of each sub-word (that hasn't been tokenized) in the string
        # on the left side of the matched token(s)
        untokenized_left_substring_end_positions = [matched_position[0] for matched_position in matched_positions]
        untokenized_left_substring_start_position = 0
        for untokenized_left_substring_end_position in untokenized_left_substring_end_positions:
            # slice for untokenized sub-word on the left side of the matched pattern
            untokenized_left_substring = string[untokenized_left_substring_start_position:untokenized_left_substring_end_position]
            # tokenize this sub-word with tokens remaining
            string_tokens += tokenize_word(string=untokenized_left_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            # since we have already tokenized the left-side subword and the "matched token",
            # we move to the next untokenized sub-word by moving the starting position forward
            untokenized_left_substring_start_position = untokenized_left_substring_end_position + len(token)
        # tokenize the remaining sub-word on the right
        untokenized_right_substring = string[untokenized_left_substring_start_position:]
        string_tokens += tokenize_word(string=untokenized_right_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    else:
        # return list of unknown token if no match is found for the string
        string_tokens = [unknown_token] * len(string)
    return string_tokens

def sort_tokens(tokens_frequencies):
    sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
    return sorted_tokens

#display the vocab
vocab_tokenization = get_vocab_tokenization(vocab)

#sort tokens by length and frequency
sorted_tokens = sort_tokens(tokens)
print("Tokens =", sorted_tokens, "\n")

sentence_1 = 'I like natural language processing!'
sentence_2 = 'I like natural languaaage processing!'
sentence_list = [sentence_1, sentence_2]

for sentence in sentence_list:
    
    print('==========')
    print("Sentence =", sentence)
    
    for word in sentence.split():
        word = word + "</w>"

        print('Tokenizing word: {}...'.format(word))
        if word in vocab_tokenization:
            print(vocab_tokenization[word])
        else:
            print(tokenize_word(string=word, sorted_tokens=sorted_tokens, unknown_token='</u>')) 

```

    Tokens = ['understanding</w>', 'algorithms</w>', 'language</w>', 'students</w>', 'understan', 'subject</w>', 'develop</w>', 'algorith', 'ication', 'languag', 'develop', 'underst', 'lation', 'subjec', 'algori', 'ing,</w>', 'used</w>', 'inclu', 'ing.</w>', 'ation', 'ding</w>', 'aims</w>', 'this</w>', 'main</w>', 'langu', 'stude', 'devel', 'under', 'algor', 'for</w>', 'and</w>', 'ing</w>', 'The</w>', 'the</w>', 'use</w>', 'assi', 'gram', 'nts</w>', 'natu', 'lang', 'incl', 'subj', 'deve', 'unde', 'in</w>', 'is</w>', 'to</w>', 'of</w>', 'pro', 'ver', 'ans', 'par', 'ion', 'for', 'ang', 'ed</w>', 'an</w>', 'ing', 'ms</w>', 'ass', 'gra', 'sub', 'stu', 'alg', 'nat', 'e</w>', ',</w>', 'in', 's</w>', 'or', 'on', 'de', 'st', 'ic', 'op', 'al', 'me', 'at', 'th', 'ag', 'nt', 'of', 'ss', 'cl', '.</w>', 'ar', 'ec', 't</w>', 'ma', 'an', 'd</w>', 'us', 've', 'Th', 'to', 'pr', 'gr', 'ai', 'su', 'un', '</w>', 'e', 's', 't', 'h', 'm', 'r', 'c', 'l', 'p', 'a', 'o', 'n', '-', 'i', 'd', 'g', 'w', 'y', 'T', 'f', 'u', 'b', 'x', 'q', 'P', 'k', 'j', 'v', ',', '.'] 
    
    ==========
    Sentence = I like natural language processing!
    Tokenizing word: I</w>...
    ['</u>', '</w>']
    Tokenizing word: like</w>...
    ['l', 'i', 'k', 'e</w>']
    Tokenizing word: natural</w>...
    ['natu', 'r', 'al', '</w>']
    Tokenizing word: language</w>...
    ['language</w>']
    Tokenizing word: processing!</w>...
    ['pro', 'c', 'e', 'ss', 'ing', '</u>', '</w>']
    ==========
    Sentence = I like natural languaaage processing!
    Tokenizing word: I</w>...
    ['</u>', '</w>']
    Tokenizing word: like</w>...
    ['l', 'i', 'k', 'e</w>']
    Tokenizing word: natural</w>...
    ['natu', 'r', 'al', '</w>']
    Tokenizing word: languaaage</w>...
    ['langu', 'a', 'a', 'ag', 'e</w>']
    Tokenizing word: processing!</w>...
    ['pro', 'c', 'e', 'ss', 'ing', '</u>', '</w>']



```python

```
