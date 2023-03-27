# Assignment 1: Preprocessing and Text Classification

Student Name: Yuheng Fan

Student ID: 987807

# General Info

<b>Due date</b>: Monday, 27 March 2023 1pm

<b>Submission method</b>: Canvas submission

<b>Submission materials</b>: completed copy of this iPython notebook

<b>Late submissions</b>: -10% per day (both week and weekend days counted)

<b>Marks</b>: 9% of mark for class (with 8% on correctness + 1% on quality and efficiency of your code)

<b>Materials</b>: See "Using Jupyter Notebook and Python" page on Canvas (under Modules>Resources) for information on the basic setup required for this class, including an iPython notebook viewer and the python packages NLTK, Numpy, Scipy, Matplotlib, Scikit-Learn, Gensim, Keras and Pytorch. We recommend installing all the data for NLTK, since you will need various parts of it to complete this assignment. You can also use any Python built-in packages, but do not use any other 3rd party packages (the packages listed above are all fine to use); if your iPython notebook doesn't run on the marker's machine, you will lose marks. <b> You should use Python 3.8</b>.  

To familiarize yourself with NLTK, here is a free online book:  Steven Bird, Ewan Klein, and Edward Loper (2009). <a href=https://www.nltk.org/book/>Natural Language Processing with Python</a>. O'Reilly Media Inc. You may also consult the <a href=https://www.nltk.org/api/nltk.html>NLTK API</a>.

<b>Evaluation</b>: Your iPython notebook should run end-to-end without any errors in a reasonable amount of time, and you must follow all instructions provided below, including specific implementation requirements and instructions for what needs to be printed (please avoid printing output we don't ask for). You should edit the sections below where requested, but leave the rest of the code as is. You should leave the output from running your code in the iPython notebook you submit, to assist with marking. The amount each question is worth is explicitly given. 

You will be marked not only on the correctness of your methods, but also the quality and efficency of your code: in particular, you should be careful to use Python built-in functions and operators when appropriate and pick descriptive variable names that adhere to <a href="https://www.python.org/dev/peps/pep-0008/">Python style requirements</a>. If you think it might be unclear what you are doing, you should comment your code to help the marker make sense of it.

<b>Updates</b>: Any major changes to the assignment will be announced via Canvas. Minor changes and clarifications will be announced on the discussion board; we recommend you check it regularly.

<b>Academic misconduct</b>: For most people, collaboration will form a natural part of the undertaking of this homework, and we encourge you to discuss it in general terms with other students. However, this ultimately is still an individual task, and so reuse of code or other instances of clear influence will be considered cheating. We will be checking submissions for originality and will invoke the University’s <a href="http://academichonesty.unimelb.edu.au/policy.html">Academic Misconduct policy</a> where inappropriate levels of collusion or plagiarism are deemed to have taken place. In regards to the use of artificial intelligence tools in the context of academic integrity, please see the university's statement <a href="https://academicintegrity.unimelb.edu.au/plagiarism-and-collusion/artificial-intelligence-tools-and-technologies">here</a>.

# Overview

In this homework, you'll be working with a collection of tweets. The task is to predict the geolocation (country) where the tweet comes from. This homework involves writing code to preprocess data and perform text classification.

# Preprocessing (4 marks)

**Instructions**: Download the data (as1-data.json) from Canvas and put it in the same directory as this iPython notebook. Run the code below to load the json data. This produces two objects, `x` and `y`, which contains a list of  tweets and corresponding country labels (it uses the standard [2 letter country code](https://www.iban.com/country-codes)) respectively. **No implementation is needed.**


```python
import json

x = []
y = []
data = json.load(open("as1-data.json"))
for k, v in data.items():
    x.append(k)
    y.append(v)
    
print("Number of tweets =", len(x))
print("Number of labels =", len(y))
print("\nSamples of data:")
for i in range(10):
    print("Country =", y[i], "\tTweet =", x[i])
    
assert(len(x) == 943)
assert(len(y) == 943)
```

    Number of tweets = 943
    Number of labels = 943
    
    Samples of data:
    Country = us 	Tweet = @Addictd2Success thx u for following
    Country = us 	Tweet = Let's just say, if I were to ever switch teams, Khalesi would be top of the list. #girlcrush
    Country = ph 	Tweet = Taemin jonghyun!!! Your birits make me go~ http://t.co/le8z3dntlA
    Country = id 	Tweet = depart.senior 👻 rapat perdana (with Nyayu, Anita, and 8 others at Ruang Aescullap FK Unsri Madang) — https://t.co/swRALlNkrQ
    Country = ph 	Tweet = Done with internship with this pretty little lady!  (@ Metropolitan Medical Center w/ 3 others) [pic]: http://t.co/1qH61R1t5r
    Country = gb 	Tweet = Wow just Boruc's clanger! Haha Sunday League stuff that, Giroud couldn't believe his luck! #clown
    Country = my 	Tweet = I'm at Sushi Zanmai (Petaling Jaya, Selangor) w/ 5 others http://t.co/bcNobykZ
    Country = us 	Tweet = Mega Fest!!!! Its going down🙏🙌  @BishopJakes
    Country = gb 	Tweet = @EllexxxPharrell wow love the pic babe xx
    Country = us 	Tweet = You have no clue how much you hurt me


### Question 1 (1.0 mark)

**Instructions**: Next we need to preprocess the collected tweets to create a bag-of-words representation (based on frequency). The preprocessing steps required here are: (1) tokenize each tweet into individual word tokens (using NLTK `TweetTokenizer`); (2) lowercase all words; (3) remove any word that does not contain any English letters in the alphabet (e.g. {_hello_, _#okay_, _abc123_} would be kept, but not {_123_, _!!_}) and (4) remove stopwords (based on NLTK `stopwords`). An empty tweet (after preprocessing) and its country label should be **excluded** from the output (`x_processed` and `y_processed`).

**Task**: Complete the `preprocess_data(data, labels)` function. The function takes **a list of tweets** and **a corresponding list of country labels** as input, and returns **two lists**. For the first list, each element is a bag-of-words representation of a tweet (represented using a python dictionary). For the second list, each element is a corresponding country label. Note that while we do not need to preprocess the country labels (`y`), we need to have a new output list (`y_processed`) because some tweets maybe removed after the preprocessing (due to having an empty set of bag-of-words).

**Check**: Use the assertion statements in <b>"For your testing"</b> below for the expected output.


```python
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

tt = TweetTokenizer()
stopwords = set(stopwords.words('english')) #note: stopwords are all in lowercase

def preprocess_data(data, labels):
    
    ###
    # Your answer BEGINS HERE
    ###

    # bag-of-word
    def get_BOW(text):
        BOW = {}
        for word in text:
            BOW[word] = BOW.get(word,0) + 1
        return BOW

    x_processed = []
    y_processed = []

    for i in range(len(data)):
        
        token = tt.tokenize(data[i])
        new_data = []
        for raw_word in token: 

            # filter the word with alphabet           
            if any(char.isalpha() for char in raw_word):
                # lower the word
                word = raw_word.lower()
                # remove stopwords
                if word not in stopwords:
                    new_data.append(word)

        # remain the data containing alphabet
        if len(new_data) > 0:
            x_processed.append(get_BOW(new_data))
            y_processed.append(labels[i])

    return x_processed, y_processed
    ###
    # Your answer ENDS HERE
    ###

x_processed, y_processed = preprocess_data(x, y)

print("Number of preprocessed tweets =", len(x_processed))
print("Number of preprocessed labels =", len(y_processed))
print("\nSamples of preprocessed data:")
for i in range(10):
    print("Country =", y_processed[i], "\tTweet =", x_processed[i])
```

    Number of preprocessed tweets = 943
    Number of preprocessed labels = 943
    
    Samples of preprocessed data:
    Country = us 	Tweet = {'@addictd2success': 1, 'thx': 1, 'u': 1, 'following': 1}
    Country = us 	Tweet = {"let's": 1, 'say': 1, 'ever': 1, 'switch': 1, 'teams': 1, 'khalesi': 1, 'would': 1, 'top': 1, 'list': 1, '#girlcrush': 1}
    Country = ph 	Tweet = {'taemin': 1, 'jonghyun': 1, 'birits': 1, 'make': 1, 'go': 1, 'http://t.co/le8z3dntla': 1}
    Country = id 	Tweet = {'depart.senior': 1, 'rapat': 1, 'perdana': 1, 'nyayu': 1, 'anita': 1, 'others': 1, 'ruang': 1, 'aescullap': 1, 'fk': 1, 'unsri': 1, 'madang': 1, 'https://t.co/swrallnkrq': 1}
    Country = ph 	Tweet = {'done': 1, 'internship': 1, 'pretty': 1, 'little': 1, 'lady': 1, 'metropolitan': 1, 'medical': 1, 'center': 1, 'w': 1, 'others': 1, 'pic': 1, 'http://t.co/1qh61r1t5r': 1}
    Country = gb 	Tweet = {'wow': 1, "boruc's": 1, 'clanger': 1, 'haha': 1, 'sunday': 1, 'league': 1, 'stuff': 1, 'giroud': 1, 'believe': 1, 'luck': 1, '#clown': 1}
    Country = my 	Tweet = {"i'm": 1, 'sushi': 1, 'zanmai': 1, 'petaling': 1, 'jaya': 1, 'selangor': 1, 'w': 1, 'others': 1, 'http://t.co/bcnobykz': 1}
    Country = us 	Tweet = {'mega': 1, 'fest': 1, 'going': 1, '@bishopjakes': 1}
    Country = gb 	Tweet = {'@ellexxxpharrell': 1, 'wow': 1, 'love': 1, 'pic': 1, 'babe': 1, 'xx': 1}
    Country = us 	Tweet = {'clue': 1, 'much': 1, 'hurt': 1}


    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/messifr/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


**For your testing**:


```python
assert(len(x_processed) == len(y_processed))
assert(len(x_processed) > 800)
```

**Instructions**: Hashtags (i.e. topic tags which start with #) pose an interesting tokenisation problem because they often include multiple words written without spaces or capitalization. Run the code below to collect all unique hashtags in the preprocessed data. **No implementation is needed.**




```python
def get_all_hashtags(data):
    hashtags = set([])
    for d in data:
        for word, frequency in d.items():
            if word.startswith("#") and len(word) > 1:
                hashtags.add(word)
    return hashtags

hashtags = get_all_hashtags(x_processed)
print("Number of hashtags =", len(hashtags))
print(sorted(hashtags))
```

    Number of hashtags = 425
    ['#100percentpay', '#1stsundayofoctober', '#1yearofalmostisneverenough', '#2011prdctn', '#2015eebritishfilmacademyawards', '#2k16', '#2littlebirds', '#365picture', '#5sosacousticatlanta', '#5sosfam', '#8thannualpubcrawl', '#affsuzukicup', '#aflpowertigers', '#ahimacon14', '#aim20', '#airasia', '#allcity', '#alliswell', '#allwedoiscurls', '#amazing', '#anferneehardaway', '#ariona', '#art', '#arte', '#artwork', '#ashes', '#asian', '#asiangirl', '#askcrawford', '#askherforfback', '#askolly', '#asksteven', '#at', '#australia', '#awesome', '#awesomepict', '#barcelona', '#bart', '#bayofislands', '#beautiful', '#bedimages', '#bell', '#beringmy', '#bettybooppose', '#bff', '#big', '#bigbertha', '#bigbreakfast', '#blackhat', '#blessedmorethanicanimagine', '#blessedsunday', '#blogtourambiente', '#bluemountains', '#bonekachika', '#boomtaob', '#booyaa', '#bored', '#boredom', '#bradersisterhood', '#breaktime', '#breedingground', '#bringithomemy', '#brooksengland', '#burgers', '#butitsalsokindofaphone', '#bye', '#camera', '#canadaelections', '#cbb', '#cbcolympics', '#cctv', '#cdnpoli', '#celebritycrush', '#chargers', '#chocolate', '#ciosummit', '#cleansidewalk', '#clown', '#coffeespoonart', '#colts', '#confused', '#cornell', '#country', '#craftbeer', '#creative', '#crepes', '#cumannanya', '#danny4mayor', '#data', '#date', '#datingsiteforyou', '#dearmind', '#deed', '#delightful', '#dennisrodman', '#design', '#devacurl', '#die', '#difd', '#diner', '#dinner', '#dragoncon', '#dus', '#dynamounlock', '#earrings', '#eeeeeehhh', '#election2015', '#electriccircus2014', '#endomondo', '#engine', '#english', '#europapark', '#excitables', '#fabulous', '#factorycampus', '#fall', '#familydinner', '#ff', '#fire', '#flambees', '#flashback', '#fly', '#focusateneo', '#followher', '#followme', '#foodporn', '#fotografías', '#fotorus', '#four', '#freaks', '#friday', '#fridaynight', '#fried', '#friends', '#friendshipflow', '#fries', '#frozenyoghurt', '#fun', '#future', '#galaxy', '#getfreetattooaviciipasses', '#girl', '#girlcrush', '#girls', '#givesmehope', '#goingout', '#google', '#graduated', '#grafunkthiremepls', '#grammyfans', '#grandmarnier', '#greenfood', '#grilled', '#gudnytall', '#gunner', '#gym', '#handbuiltbicycle', '#happybirthdaysandarapark', '#happyfriday', '#harimaumalaya', '#hb60', '#hens', '#hippy', '#holiday', '#hollywoodmusicawards', '#homemadegranola', '#hometomama', '#hot', '#hungergames', '#hungry', '#icu', '#ididntchoosethestudentlifeitchoseme', '#iloveyou', '#imsobored', '#imsosore', '#innoretail', '#insanity', '#insightmedia', '#insightmediasingapore', '#instaframeplus', '#instagood', '#instalook', '#isibaya', '#javaboy', '#jed', '#jewelry', '#jo', '#jordaan', '#jrsurfboards', '#justshare', '#kllive', '#ladygaga', '#latepost', '#laugh', '#laundryservice', '#lazysunday', '#learningcommunties', '#lebedeintennis', '#letmesleep', '#lfc', '#lgbt', '#life', '#lipstickfree', '#littlemonsters', '#loadsoffun', '#lol', '#longranger', '#lotsoflove', '#love', '#lovers', '#lovethisgirl', '#lovevibes', '#lte', '#magazinesandtvscreens', '#magic', '#makeupfree', '#makingemuklajawabnya', '#mamajeanneandme', '#mancrush', '#march', '#massacreconspiracy', '#mauce', '#mavic', '#me', '#meetup', '#melbourne', '#michaelkors', '#mindfulness', '#mkmedi', '#mobile', '#morning', '#mountains', '#movies', '#mtlnewtech', '#mtvhottest', '#mushroom', '#music', '#mustfollow', '#mwc14', '#mybabyemilia', '#myfriendsarebetterthanyours', '#nced', '#ncga', '#ncpol', '#nevergetsold', '#new', '#newlooks', '#newrecord', '#nextlevel', '#nfl', '#nickryrie', '#nochillzone', '#nofilter', '#notersnew2014', '#notreally', '#np', '#nye', '#of', '#offtochurch', '#ohyeah', '#oilandgas', '#olah', '#on', '#oops', '#openspace', '#oscars', '#oui', '#palacefansinthemorning', '#panther', '#panthers', '#partyhardpartyy', '#pats', '#pechanga', '#penny', '#peperoni', '#peppermoney', '#photo', '#photoby', '#photography', '#pic', '#pll', '#pmattheashes', '#pop-up', '#positivity', '#potd', '#procrastination', '#promise', '#purplefriday', '#rainorshine', '#reachingyougpeople', '#realgoodbikes', '#redbull', '#retail', '#revfcwh', '#rippaulwalker', '#ritenow', '#rollercoaster', '#rollersmusicawards', '#rollsroyce', '#rose', '#rosegold', '#rt', '#rundude', '#ryanpurrdler', '#sad', '#safm', '#saints', '#samsung', '#sandwich', '#sarahm', '#saulbass', '#scifigeek', '#security', '#seniorbabysenior', '#sexy', '#sfgiants', '#sggirls', '#shakes', '#shakethatbooty', '#shellvpower', '#shenanigans', '#shopaholic', '#siberuang', '#silver', '#singapore', '#singlefighter', '#sinvsmas', '#skeemsaam', '#skullsearcher', '#sl', '#socal', '#sorrynotsorry', '#southampton', '#spafy', '#spicy', '#spider', '#squad', '#stampede2014', '#startupfest', '#startuphub', '#starving', '#statoftheday', '#stayfreshsaturdaydbn', '#stkilda', '#stop', '#stopcomplaining', '#strictlyus', '#studio', '#summer', '#sun', '#sunrise', '#sunshine', '#supremelaundry', '#surfshop', '#swedumtl', '#sycip', '#taintedlove', '#takeabreak', '#tcschleissheim', '#tdwpliveinkl', '#teamcanada', '#tennis', '#thaiexpress', '#thaifood', '#thankful', '#thankyoulord', '#thankyoupatients', '#thecolorrun2014', '#thedisadvantagesofakindle', '#them', '#thenext5goldenyears', '#thevoiceau', '#thewalkingdead', '#thomassabo', '#throwback', '#thursday', '#ticklish', '#toronto', '#tpcoach', '#transit', '#truegrip', '#tuesday', '#tune', '#turbine', '#txlege', '#ujackbastards', '#umg', '#uniexamonasaturday', '#universal', '#uptonogood', '#urbangardening', '#uss', '#usydhereicome', '#usydoweek', '#utopia', '#vanilla', '#vca', '#vegan', '#veganfood', '#vegetables', '#vegetarian', '#video', '#vma', '#voteonedirection', '#vsco', '#vscocam', '#walking', '#watch', '#weare90s', '#wearesocial', '#white', '#wings', '#wok', '#wood', '#work', '#workmates', '#world', '#worldcup2014', '#yellow', '#yiamas', '#ynwa', '#youtube', '#yummy', '#yws13', '#zweihandvollfarm']


### Question 2 (1.0 mark)

**Instructions**: Our task here to tokenize the hashtags, by implementing the **MaxMatch algorithm** discussed in class.

NLTK has a list of words that you can use for matching, see starter code below (`words`). Be careful about efficiency with respect to doing word lookups. One extra challenge you have to deal with is that the provided list of words (`words`) includes only lemmas: your MaxMatch algorithm should match inflected forms by converting them into lemmas using the NLTK lemmatizer before matching (provided by the function `lemmatize(word)`). Note that the list of words (`words`) is the only source that you'll use for matching (i.e. you do not need to find  other external word lists). If you are unable to make any longer match, your code should default to matching a single letter.

For example, given "#newrecords", the algorithm should produce: \["#", "new", "records"\].

**Task**: Complete the `tokenize_hashtags(hashtags)` function by implementing the MaxMatch algorithm. The function takes as input **a set of hashtags**, and returns **a dictionary** where key="hashtag" and value="a list of tokenised words".

**Check**: Use the assertion statements in <b>"For your testing"</b> below for the expected output.


```python
from nltk.corpus import wordnet
nltk.download('words')
nltk.download('wordnet')

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
words = set(nltk.corpus.words.words()) #a list of words provided by NLTK
words = set([ word.lower() for word in words ]) #lowercase all the words for better matching

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma


def tokenize_hashtags(hashtags):
    ###
    # Your answer BEGINS HERE
    ###
    tokenized_hashtags = {}
    
    for tag in hashtags:
        # remain '#' at the start of list
        token = ['#']
        tag_str = tag[1:]
        
        while (tag_str):
            
            # start searching from the end of hashtag
            for i in range(len(tag_str), -1, -1):
                if lemmatize(tag_str[:i]) in words:
                    token.append(tag_str[:i])
                    tag_str = tag_str[i:]
                    break

                # skip this character
                if i == 0:
                    token.append(tag_str[0])
                    tag_str = tag_str[1:]

        tokenized_hashtags[tag] = token

    return tokenized_hashtags

    ###
    # Your answer ENDS HERE
    ###

#tokenise hashtags with MaxMatch
tokenized_hashtags = tokenize_hashtags(hashtags)

#print results
for k, v in sorted(tokenized_hashtags.items())[-30:]:
    print(k, v)
```

    [nltk_data] Downloading package words to /Users/messifr/nltk_data...
    [nltk_data]   Package words is already up-to-date!
    [nltk_data] Downloading package wordnet to /Users/messifr/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


    #vanilla ['#', 'vanilla']
    #vca ['#', 'v', 'ca']
    #vegan ['#', 'vega', 'n']
    #veganfood ['#', 'vega', 'n', 'food']
    #vegetables ['#', 'vegetables']
    #vegetarian ['#', 'vegetarian']
    #video ['#', 'video']
    #vma ['#', 'v', 'ma']
    #voteonedirection ['#', 'vote', 'one', 'direction']
    #vsco ['#', 'vs', 'c', 'o']
    #vscocam ['#', 'vs', 'coca', 'm']
    #walking ['#', 'walking']
    #watch ['#', 'watch']
    #weare90s ['#', 'wear', 'e', '9', '0', 's']
    #wearesocial ['#', 'weares', 'o', 'c', 'i', 'al']
    #white ['#', 'white']
    #wings ['#', 'wings']
    #wok ['#', 'wo', 'k']
    #wood ['#', 'wood']
    #work ['#', 'work']
    #workmates ['#', 'work', 'mates']
    #world ['#', 'world']
    #worldcup2014 ['#', 'world', 'cup', '2', '0', '1', '4']
    #yellow ['#', 'yellow']
    #yiamas ['#', 'y', 'i', 'ama', 's']
    #ynwa ['#', 'yn', 'wa']
    #youtube ['#', 'you', 'tube']
    #yummy ['#', 'yummy']
    #yws13 ['#', 'y', 'ws', '1', '3']
    #zweihandvollfarm ['#', 'z', 'wei', 'hand', 'vol', 'l', 'farm']


**For your testing:**


```python
assert(len(tokenized_hashtags) == len(hashtags))
assert(tokenized_hashtags["#newrecord"] == ["#", "new", "record"])
```

### Question 3 (1.0 mark)

**Instructions**: Our next task is to tokenize the hashtags again, but this time using a **reversed version of the MaxMatch algorithm**, where matching begins at the end of the hashtag and progresses backwards (e.g. for <i>#helloworld</i>, we would process it right to left, starting from the last character <i>d</i>). Just like before, you should use the provided word list (`words`) for word matching.

**Task**: Complete the `tokenize_hashtags_rev(hashtags)` function by the MaxMatch algorithm. The function takes as input **a set of hashtags**, and returns **a dictionary** where key="hashtag" and value="a list of tokenised words".

**Check**: Use the assertion statements in <b>"For your testing"</b> below for the expected output.


```python
def tokenize_hashtags_rev(hashtags):
    ###
    # Your answer BEGINS HERE
    ###
    tokenized_hashtags_rev = {}
    
    for tag in hashtags:
        token = []
        tag_str = tag[1:]
        
        while (tag_str):
            
            # start search from the start of hashtag
            for i in range(len(tag_str)+1):
                if lemmatize(tag_str[i:]) in words:
                    token.insert(0, tag_str[i:])
                    tag_str = tag_str[:i]
                    break
                
                # skip this character
                if i == len(tag_str):
                    token.insert(0, tag_str[-1])
                    tag_str = tag_str[:-1]

        # '#' at the start of list
        token.insert(0, '#')
        tokenized_hashtags_rev[tag] = token

    return tokenized_hashtags_rev
    ###
    # Your answer ENDS HERE
    ###

    
#tokenise hashtags with the reversed version of MaxMatch
tokenized_hashtags_rev = tokenize_hashtags_rev(hashtags)

#print results
for k, v in sorted(tokenized_hashtags_rev.items())[-30:]:
    print(k, v)
```

    #vanilla ['#', 'vanilla']
    #vca ['#', 'v', 'ca']
    #vegan ['#', 'v', 'e', 'gan']
    #veganfood ['#', 'v', 'e', 'gan', 'food']
    #vegetables ['#', 'vegetables']
    #vegetarian ['#', 'vegetarian']
    #video ['#', 'video']
    #vma ['#', 'v', 'ma']
    #voteonedirection ['#', 'vote', 'one', 'direction']
    #vsco ['#', 'vs', 'c', 'o']
    #vscocam ['#', 'vs', 'c', 'o', 'cam']
    #walking ['#', 'walking']
    #watch ['#', 'watch']
    #weare90s ['#', 'we', 'are', '9', '0', 's']
    #wearesocial ['#', 'we', 'are', 'social']
    #white ['#', 'white']
    #wings ['#', 'wings']
    #wok ['#', 'w', 'ok']
    #wood ['#', 'wood']
    #work ['#', 'work']
    #workmates ['#', 'work', 'mates']
    #world ['#', 'world']
    #worldcup2014 ['#', 'world', 'cup', '2', '0', '1', '4']
    #yellow ['#', 'yellow']
    #yiamas ['#', 'y', 'i', 'a', 'mas']
    #ynwa ['#', 'yn', 'wa']
    #youtube ['#', 'you', 'tube']
    #yummy ['#', 'yummy']
    #yws13 ['#', 'y', 'ws', '1', '3']
    #zweihandvollfarm ['#', 'z', 'wei', 'hand', 'vol', 'l', 'farm']


**For your testing:**


```python
assert(len(tokenized_hashtags_rev) == len(hashtags))
assert(tokenized_hashtags_rev["#newrecord"] == ["#", "new", "record"])
```

### Question 4 (1.0 mark)

**Instructions**: The two versions of MaxMatch will produce different results for some of the hashtags. For a hastag that has different results, our task here is to use a **unigram language model** (lecture 3) to score them to see which is better. Recall that in a unigram language model we compute P(<i>#</i>, <i>hello</i>, <i>world</i> = P(<i>#</i>)\*P(<i>hellow</i>)\*P(<i>world</i>).

You should: (1) use the NLTK's Brown corpus (`brown_words`) for collecting word frequencies (note: the words are already tokenised so no further tokenisation is needed); (2) lowercase all words in the corpus; (3) use add-one smoothing when computing the unigram probabilities; and (4) work in the log space to prevent numerical underflow.

**Task**: Build a unigram language model with add-one smoothing using the word counts from the Brown corpus. Iterate through the hashtags, and for each hashtag where MaxMatch and reversed MaxMatch produce different results, print the following: (1) the hashtag; (2) the results produced by MaxMatch and reversed MaxMatch; and (3) the log probability of each result as given by the unigram language model. Note: you **do not** need to print the hashtags where MaxMatch and reversed MaxMatch produce the same results.

An example output:
```
1. #abcd
MaxMatch = [#, a, bc, d]; LogProb = -2.3
Reversed MaxMatch = [#, a, b, cd]; LogProb = -3.5

2. #efgh
MaxMatch = [#, ef, g, h]; LogProb = -4.2
Reversed MaxMatch = [#, e, fgh]; LogProb = -3.1

```

Have a look at the output, and see if the sequences with better language model scores (i.e. less negative) are generally more coherent.


```python
from nltk.corpus import brown

#words from brown corpus
brown_words = brown.words()

###
# Your answer BEGINS HERE
###
import math

# the word counts of brown_words
word_counts = nltk.FreqDist(word.lower() for word in brown_words)

M = len(brown_words) # number of total words
V = len(word_counts) # number of unique words

def unigram_log_prob(word):
    return math.log((word_counts[word]+1)/(M+V))

# used to calculate the average log-prob
log_probs = []
log_probs_rev = []
number = 1

for key in tokenized_hashtags.keys():

    # if the results from MaxMatch and Reversed MaxMatch are same then skip
    if tokenized_hashtags[key] == tokenized_hashtags_rev[key]:
        continue

    print(f'{number}. {key}')

    # calculate the log-probability of tokenized hashtags using MaxMatch 
    val_tokenized_hashtag = tokenized_hashtags[key]
    log_prob_tokenized_hashtags = sum([unigram_log_prob(word) for word in val_tokenized_hashtag])
    print(f'MaxMatch = {val_tokenized_hashtag}; LogProb = {log_prob_tokenized_hashtags:.1f}')
    log_probs.append(log_prob_tokenized_hashtags)

    # calculate the log-probability of tokenized hashtags using Reversed MaxMatch 
    val_tokenized_hashtag_rev = tokenized_hashtags_rev[key]
    log_prob_tokenized_hashtags_rev = sum([unigram_log_prob(word) for word in val_tokenized_hashtag_rev])
    print(f'Reversed MaxMatch = {val_tokenized_hashtag_rev}; LogProb = {log_prob_tokenized_hashtags_rev:.1f}')
    log_probs_rev.append(log_prob_tokenized_hashtags_rev)

    number += 1

print(f'\nNumber of Different Hashtags: {number-1}')
print(f'Average MaxMatch LogProb = {sum(log_probs)/len(log_probs):.2f}')
print(f'Average Reversed MaxMatch LogProb = {sum(log_probs_rev)/len(log_probs_rev):.2f}')

###
# Your answer ENDS HERE
###
```

    1. #flambees
    MaxMatch = ['#', 'flamb', 'e', 'es']; LogProb = -52.3
    Reversed MaxMatch = ['#', 'flam', 'bees']; LogProb = -39.2
    2. #bradersisterhood
    MaxMatch = ['#', 'brad', 'ers', 'ist', 'er', 'hood']; LogProb = -80.6
    Reversed MaxMatch = ['#', 'brad', 'er', 'sisterhood']; LogProb = -55.3
    3. #cornell
    MaxMatch = ['#', 'cornel', 'l']; LogProb = -39.2
    Reversed MaxMatch = ['#', 'cor', 'nell']; LogProb = -41.3
    4. #foodporn
    MaxMatch = ['#', 'food', 'po', 'r', 'n']; LogProb = -57.0
    Reversed MaxMatch = ['#', 'food', 'p', 'or', 'n']; LogProb = -48.7
    5. #grammyfans
    MaxMatch = ['#', 'gram', 'my', 'fans']; LogProb = -43.4
    Reversed MaxMatch = ['#', 'g', 'rammy', 'fans']; LogProb = -50.1
    6. #askolly
    MaxMatch = ['#', 'ask', 'o', 'l', 'ly']; LogProb = -59.0
    Reversed MaxMatch = ['#', 'as', 'kol', 'ly']; LogProb = -47.1
    7. #laundryservice
    MaxMatch = ['#', 'laundrys', 'er', 'vice']; LogProb = -52.3
    Reversed MaxMatch = ['#', 'laundry', 'service']; LogProb = -34.5
    8. #vscocam
    MaxMatch = ['#', 'vs', 'coca', 'm']; LogProb = -51.6
    Reversed MaxMatch = ['#', 'vs', 'c', 'o', 'cam']; LogProb = -59.6
    9. #ritenow
    MaxMatch = ['#', 'rite', 'now']; LogProb = -32.6
    Reversed MaxMatch = ['#', 'rit', 'enow']; LogProb = -42.0
    10. #txlege
    MaxMatch = ['#', 't', 'x', 'leg', 'e']; LogProb = -55.1
    Reversed MaxMatch = ['#', 't', 'x', 'l', 'e', 'ge']; LogProb = -69.3
    11. #peperoni
    MaxMatch = ['#', 'pep', 'er', 'on', 'i']; LogProb = -52.7
    Reversed MaxMatch = ['#', 'pep', 'e', 'ro', 'ni']; LogProb = -66.3
    12. #yiamas
    MaxMatch = ['#', 'y', 'i', 'ama', 's']; LogProb = -55.9
    Reversed MaxMatch = ['#', 'y', 'i', 'a', 'mas']; LogProb = -49.6
    13. #thewalkingdead
    MaxMatch = ['#', 'thew', 'alk', 'ing', 'dead']; LogProb = -64.9
    Reversed MaxMatch = ['#', 'the', 'walking', 'dead']; LogProb = -35.7
    14. #loadsoffun
    MaxMatch = ['#', 'loads', 'off', 'un']; LogProb = -44.5
    Reversed MaxMatch = ['#', 'loads', 'of', 'fun']; LogProb = -39.3
    15. #lebedeintennis
    MaxMatch = ['#', 'l', 'e', 'bed', 'e', 'in', 'tennis']; LogProb = -70.2
    Reversed MaxMatch = ['#', 'l', 'e', 'be', 'de', 'in', 'tennis']; LogProb = -65.2
    16. #insightmediasingapore
    MaxMatch = ['#', 'insight', 'media', 'sing', 'a', 'pore']; LogProb = -63.6
    Reversed MaxMatch = ['#', 'insight', 'me', 'di', 'as', 'inga', 'pore']; LogProb = -74.7
    17. #nevergetsold
    MaxMatch = ['#', 'never', 'gets', 'old']; LogProb = -38.8
    Reversed MaxMatch = ['#', 'never', 'get', 'sold']; LogProb = -39.0
    18. #uptonogood
    MaxMatch = ['#', 'up', 'ton', 'og', 'o', 'od']; LogProb = -70.6
    Reversed MaxMatch = ['#', 'up', 'to', 'no', 'good']; LogProb = -38.0
    19. #usydoweek
    MaxMatch = ['#', 'us', 'y', 'dow', 'e', 'e', 'k']; LogProb = -78.3
    Reversed MaxMatch = ['#', 'us', 'y', 'do', 'week']; LogProb = -48.9
    20. #siberuang
    MaxMatch = ['#', 'sib', 'er', 'uang']; LogProb = -56.0
    Reversed MaxMatch = ['#', 'si', 'ber', 'uang']; LogProb = -55.3
    21. #bettybooppose
    MaxMatch = ['#', 'betty', 'boo', 'p', 'pose']; LogProb = -61.1
    Reversed MaxMatch = ['#', 'betty', 'bo', 'oppose']; LogProb = -51.2
    22. #swedumtl
    MaxMatch = ['#', 's', 'wed', 'um', 't', 'l']; LogProb = -71.0
    Reversed MaxMatch = ['#', 's', 'we', 'dum', 't', 'l']; LogProb = -65.8
    23. #8thannualpubcrawl
    MaxMatch = ['#', '8', 'than', 'nu', 'alp', 'u', 'b', 'crawl']; LogProb = -90.1
    Reversed MaxMatch = ['#', '8', 'th', 'annual', 'pub', 'crawl']; LogProb = -71.6
    24. #weare90s
    MaxMatch = ['#', 'wear', 'e', '9', '0', 's']; LogProb = -66.3
    Reversed MaxMatch = ['#', 'we', 'are', '9', '0', 's']; LogProb = -57.4
    25. #startuphub
    MaxMatch = ['#', 'start', 'up', 'hub']; LogProb = -41.0
    Reversed MaxMatch = ['#', 'star', 'tup', 'hub']; LogProb = -50.3
    26. #pmattheashes
    MaxMatch = ['#', 'p', 'matt', 'he', 'ashes']; LogProb = -53.2
    Reversed MaxMatch = ['#', 'p', 'mat', 'the', 'ashes']; LogProb = -50.8
    27. #longranger
    MaxMatch = ['#', 'long', 'ranger']; LogProb = -34.3
    Reversed MaxMatch = ['#', 'l', 'on', 'granger']; LogProb = -44.4
    28. #anferneehardaway
    MaxMatch = ['#', 'an', 'fern', 'e', 'eh', 'ar', 'daw', 'ay']; LogProb = -97.2
    Reversed MaxMatch = ['#', 'an', 'f', 'er', 'nee', 'hard', 'away']; LogProb = -74.8
    29. #mtlnewtech
    MaxMatch = ['#', 'm', 't', 'l', 'newt', 'e', 'c', 'h']; LogProb = -88.1
    Reversed MaxMatch = ['#', 'm', 't', 'l', 'new', 'tech']; LogProb = -64.4
    30. #getfreetattooaviciipasses
    MaxMatch = ['#', 'get', 'freet', 'at', 'too', 'a', 'vic', 'i', 'i', 'passes']; LogProb = -86.0
    Reversed MaxMatch = ['#', 'get', 'free', 'tattoo', 'a', 'vic', 'i', 'i', 'passes']; LogProb = -81.8
    31. #isibaya
    MaxMatch = ['#', 'is', 'iba', 'ya']; LogProb = -45.2
    Reversed MaxMatch = ['#', 'i', 'si', 'baya']; LogProb = -46.8
    32. #thevoiceau
    MaxMatch = ['#', 'the', 'voice', 'a', 'u']; LogProb = -40.8
    Reversed MaxMatch = ['#', 'the', 'v', 'o', 'i', 'c', 'ea', 'u']; LogProb = -78.3
    33. #bedimages
    MaxMatch = ['#', 'bedim', 'ages']; LogProb = -38.1
    Reversed MaxMatch = ['#', 'bed', 'images']; LogProb = -33.5
    34. #palacefansinthemorning
    MaxMatch = ['#', 'palace', 'fans', 'in', 'them', 'or', 'ning']; LogProb = -65.5
    Reversed MaxMatch = ['#', 'palace', 'fan', 'sin', 'the', 'morning']; LogProb = -56.9
    35. #coffeespoonart
    MaxMatch = ['#', 'coffees', 'poon', 'art']; LogProb = -50.7
    Reversed MaxMatch = ['#', 'coffee', 'spoon', 'art']; LogProb = -44.4
    36. #focusateneo
    MaxMatch = ['#', 'focus', 'aten', 'e', 'o']; LogProb = -59.3
    Reversed MaxMatch = ['#', 'fo', 'c', 'u', 'sate', 'neo']; LogProb = -76.6
    37. #blessedsunday
    MaxMatch = ['#', 'blesseds', 'un', 'day']; LogProb = -46.9
    Reversed MaxMatch = ['#', 'blessed', 'sunday']; LogProb = -34.8
    38. #imsobored
    MaxMatch = ['#', 'i', 'ms', 'o', 'bored']; LogProb = -54.4
    Reversed MaxMatch = ['#', 'i', 'm', 'so', 'bored']; LogProb = -48.4
    39. #oilandgas
    MaxMatch = ['#', 'oil', 'and', 'gas']; LogProb = -36.6
    Reversed MaxMatch = ['#', 'o', 'i', 'land', 'gas']; LogProb = -48.2
    40. #datingsiteforyou
    MaxMatch = ['#', 'datings', 'it', 'e', 'for', 'you']; LogProb = -54.0
    Reversed MaxMatch = ['#', 'dating', 'site', 'for', 'you']; LogProb = -47.0
    41. #bringithomemy
    MaxMatch = ['#', 'bring', 'it', 'home', 'my']; LogProb = -42.4
    Reversed MaxMatch = ['#', 'brin', 'git', 'home', 'my']; LogProb = -54.9
    42. #burgers
    MaxMatch = ['#', 'burg', 'ers']; LogProb = -42.0
    Reversed MaxMatch = ['#', 'bur', 'gers']; LogProb = -42.0
    43. #samsung
    MaxMatch = ['#', 'sams', 'un', 'g']; LogProb = -50.6
    Reversed MaxMatch = ['#', 'sam', 'sung']; LogProb = -34.7
    44. #instalook
    MaxMatch = ['#', 'ins', 'tal', 'o', 'ok']; LogProb = -64.9
    Reversed MaxMatch = ['#', 'ins', 'ta', 'look']; LogProb = -50.0
    45. #hometomama
    MaxMatch = ['#', 'home', 'toma', 'ma']; LogProb = -46.7
    Reversed MaxMatch = ['#', 'home', 'tom', 'ama']; LogProb = -44.9
    46. #usydhereicome
    MaxMatch = ['#', 'us', 'y', 'd', 'here', 'i', 'come']; LogProb = -64.2
    Reversed MaxMatch = ['#', 'u', 'syd', 'here', 'i', 'come']; LogProb = -59.8
    47. #grandmarnier
    MaxMatch = ['#', 'grandma', 'r', 'ni', 'er']; LogProb = -63.7
    Reversed MaxMatch = ['#', 'grand', 'm', 'arni', 'er']; LogProb = -63.3
    48. #shopaholic
    MaxMatch = ['#', 'shop', 'aho', 'li', 'c']; LogProb = -61.1
    Reversed MaxMatch = ['#', 'sho', 'paho', 'li', 'c']; LogProb = -65.2
    49. #olah
    MaxMatch = ['#', 'o', 'la', 'h']; LogProb = -45.6
    Reversed MaxMatch = ['#', 'o', 'l', 'ah']; LogProb = -46.8
    50. #harimaumalaya
    MaxMatch = ['#', 'ha', 'rima', 'um', 'ala', 'ya']; LogProb = -79.7
    Reversed MaxMatch = ['#', 'h', 'ar', 'i', 'mau', 'mala', 'ya']; LogProb = -84.7
    51. #alliswell
    MaxMatch = ['#', 'all', 'is', 'well']; LogProb = -32.0
    Reversed MaxMatch = ['#', 'al', 'li', 'swell']; LogProb = -50.7
    52. #blogtourambiente
    MaxMatch = ['#', 'blo', 'g', 'tour', 'ambient', 'e']; LogProb = -73.7
    Reversed MaxMatch = ['#', 'b', 'log', 'tou', 'ram', 'bien', 'te']; LogProb = -89.1
    53. #thankyoulord
    MaxMatch = ['#', 'thank', 'youl', 'or', 'd']; LogProb = -54.1
    Reversed MaxMatch = ['#', 'thank', 'you', 'lord']; LogProb = -39.8
    54. #reachingyougpeople
    MaxMatch = ['#', 'reaching', 'you', 'g', 'people']; LogProb = -48.6
    Reversed MaxMatch = ['#', 'reaching', 'yo', 'ug', 'people']; LogProb = -59.5
    55. #cumannanya
    MaxMatch = ['#', 'cum', 'anna', 'n', 'ya']; LogProb = -62.7
    Reversed MaxMatch = ['#', 'c', 'u', 'mannan', 'ya']; LogProb = -61.0
    56. #vegan
    MaxMatch = ['#', 'vega', 'n']; LogProb = -38.4
    Reversed MaxMatch = ['#', 'v', 'e', 'gan']; LogProb = -49.0
    57. #ladygaga
    MaxMatch = ['#', 'lady', 'gag', 'a']; LogProb = -40.0
    Reversed MaxMatch = ['#', 'lady', 'g', 'aga']; LogProb = -48.8
    58. #veganfood
    MaxMatch = ['#', 'vega', 'n', 'food']; LogProb = -47.4
    Reversed MaxMatch = ['#', 'v', 'e', 'gan', 'food']; LogProb = -58.0
    59. #southampton
    MaxMatch = ['#', 'south', 'am', 'p', 'ton']; LogProb = -52.1
    Reversed MaxMatch = ['#', 's', 'out', 'ham', 'p', 'ton']; LogProb = -63.3
    60. #surfshop
    MaxMatch = ['#', 'surfs', 'hop']; LogProb = -40.9
    Reversed MaxMatch = ['#', 'surf', 'shop']; LogProb = -37.2
    61. #beringmy
    MaxMatch = ['#', 'beri', 'n', 'g', 'my']; LogProb = -56.4
    Reversed MaxMatch = ['#', 'be', 'ring', 'my']; LogProb = -36.2
    62. #google
    MaxMatch = ['#', 'goo', 'g', 'l', 'e']; LogProb = -60.7
    Reversed MaxMatch = ['#', 'go', 'ogle']; LogProb = -35.6
    63. #ariona
    MaxMatch = ['#', 'arion', 'a']; LogProb = -32.0
    Reversed MaxMatch = ['#', 'ar', 'i', 'ona']; LogProb = -47.5
    64. #toronto
    MaxMatch = ['#', 'toro', 'n', 'to']; LogProb = -42.2
    Reversed MaxMatch = ['#', 'tor', 'onto']; LogProb = -37.9
    65. #melbourne
    MaxMatch = ['#', 'mel', 'bourn', 'e']; LogProb = -49.4
    Reversed MaxMatch = ['#', 'm', 'elb', 'our', 'ne']; LogProb = -58.1
    66. #magazinesandtvscreens
    MaxMatch = ['#', 'magazines', 'and', 't', 'vs', 'creen', 's']; LogProb = -75.4
    Reversed MaxMatch = ['#', 'magazine', 'sand', 't', 'v', 'screens']; LogProb = -66.9
    67. #rainorshine
    MaxMatch = ['#', 'rain', 'ors', 'hin', 'e']; LogProb = -62.1
    Reversed MaxMatch = ['#', 'r', 'ai', 'nor', 'shine']; LogProb = -58.6
    68. #photoby
    MaxMatch = ['#', 'photo', 'by']; LogProb = -31.7
    Reversed MaxMatch = ['#', 'pho', 'toby']; LogProb = -42.0
    69. #openspace
    MaxMatch = ['#', 'opens', 'pace']; LogProb = -35.4
    Reversed MaxMatch = ['#', 'open', 'space']; LogProb = -31.0
    70. #instagood
    MaxMatch = ['#', 'ins', 'tag', 'o', 'od']; LogProb = -64.9
    Reversed MaxMatch = ['#', 'ins', 'ta', 'good']; LogProb = -49.3
    71. #thankyoupatients
    MaxMatch = ['#', 'thank', 'youp', 'ati', 'en', 'ts']; LogProb = -78.4
    Reversed MaxMatch = ['#', 'thank', 'you', 'patients']; LogProb = -40.7
    72. #asksteven
    MaxMatch = ['#', 'asks', 'te', 'v', 'en']; LogProb = -61.7
    Reversed MaxMatch = ['#', 'ask', 'steven']; LogProb = -37.2
    73. #blessedmorethanicanimagine
    MaxMatch = ['#', 'blessed', 'more', 'than', 'i', 'can', 'imagine']; LogProb = -60.1
    Reversed MaxMatch = ['#', 'blessed', 'more', 'th', 'ani', 'can', 'imagine']; LogProb = -75.0
    74. #seniorbabysenior
    MaxMatch = ['#', 'senior', 'babys', 'en', 'io', 'r']; LogProb = -74.1
    Reversed MaxMatch = ['#', 'senior', 'baby', 'senior']; LogProb = -44.8
    75. #arte
    MaxMatch = ['#', 'art', 'e']; LogProb = -33.0
    Reversed MaxMatch = ['#', 'ar', 'te']; LogProb = -42.0
    76. #cbcolympics
    MaxMatch = ['#', 'c', 'b', 'coly', 'm', 'pics']; LogProb = -71.7
    Reversed MaxMatch = ['#', 'c', 'b', 'col', 'ym', 'pics']; LogProb = -74.6
    77. #startupfest
    MaxMatch = ['#', 'start', 'up', 'fest']; LogProb = -43.4
    Reversed MaxMatch = ['#', 'star', 'tup', 'fest']; LogProb = -52.8
    78. #socal
    MaxMatch = ['#', 'soc', 'al']; LogProb = -38.8
    Reversed MaxMatch = ['#', 'so', 'cal']; LogProb = -33.7
    79. #fotografías
    MaxMatch = ['#', 'fot', 'og', 'ra', 'f', 'í', 'as']; LogProb = -84.9
    Reversed MaxMatch = ['#', 'f', 'oto', 'gra', 'f', 'í', 'as']; LogProb = -81.9
    80. #wearesocial
    MaxMatch = ['#', 'weares', 'o', 'c', 'i', 'al']; LogProb = -64.1
    Reversed MaxMatch = ['#', 'we', 'are', 'social']; LogProb = -33.8
    81. #innoretail
    MaxMatch = ['#', 'inn', 'ore', 'tail']; LogProb = -50.5
    Reversed MaxMatch = ['#', 'in', 'no', 'retail']; LogProb = -35.3
    82. #skeemsaam
    MaxMatch = ['#', 'skee', 'ms', 'aam']; LogProb = -54.9
    Reversed MaxMatch = ['#', 's', 'k', 'e', 'ems', 'aam']; LogProb = -74.8
    83. #fotorus
    MaxMatch = ['#', 'fot', 'or', 'us']; LogProb = -41.2
    Reversed MaxMatch = ['#', 'fo', 'torus']; LogProb = -42.0
    84. #goingout
    MaxMatch = ['#', 'going', 'out']; LogProb = -28.4
    Reversed MaxMatch = ['#', 'go', 'in', 'gout']; LogProb = -38.5
    85. #1stsundayofoctober
    MaxMatch = ['#', '1', 'st', 'sunday', 'ofo', 'c', 'tobe', 'r']; LogProb = -92.7
    Reversed MaxMatch = ['#', '1', 'st', 'sunday', 'of', 'october']; LogProb = -58.7
    86. #cleansidewalk
    MaxMatch = ['#', 'cleans', 'ide', 'walk']; LogProb = -50.7
    Reversed MaxMatch = ['#', 'clean', 'sidewalk']; LogProb = -34.7
    87. #imsosore
    MaxMatch = ['#', 'i', 'ms', 'os', 'ore']; LogProb = -59.7
    Reversed MaxMatch = ['#', 'i', 'm', 'so', 'sore']; LogProb = -48.7
    88. #makeupfree
    MaxMatch = ['#', 'make', 'up', 'free']; LogProb = -36.2
    Reversed MaxMatch = ['#', 'ma', 'keup', 'free']; LogProb = -47.5
    89. #learningcommunties
    MaxMatch = ['#', 'learning', 'c', 'om', 'munt', 'ies']; LogProb = -75.1
    Reversed MaxMatch = ['#', 'learning', 'c', 'om', 'm', 'unties']; LogProb = -72.3
    90. #scifigeek
    MaxMatch = ['#', 's', 'c', 'if', 'i', 'geek']; LogProb = -59.9
    Reversed MaxMatch = ['#', 's', 'c', 'i', 'fi', 'geek']; LogProb = -67.6
    91. #jrsurfboards
    MaxMatch = ['#', 'j', 'rs', 'urf', 'boards']; LogProb = -64.2
    Reversed MaxMatch = ['#', 'j', 'r', 'surfboards']; LogProb = -50.4
    92. #endomondo
    MaxMatch = ['#', 'end', 'om', 'on', 'do']; LogProb = -48.0
    Reversed MaxMatch = ['#', 'en', 'do', 'mon', 'do']; LogProb = -52.8
    93. #allwedoiscurls
    MaxMatch = ['#', 'all', 'wed', 'o', 'is', 'curls']; LogProb = -61.7
    Reversed MaxMatch = ['#', 'all', 'w', 'edo', 'is', 'curls']; LogProb = -64.3
    94. #brooksengland
    MaxMatch = ['#', 'brooks', 'en', 'gland']; LogProb = -48.8
    Reversed MaxMatch = ['#', 'brook', 'sen', 'gland']; LogProb = -52.3
    95. #letmesleep
    MaxMatch = ['#', 'let', 'mes', 'leep']; LogProb = -50.1
    Reversed MaxMatch = ['#', 'let', 'me', 'sleep']; LogProb = -38.8
    96. #wok
    MaxMatch = ['#', 'wo', 'k']; LogProb = -39.6
    Reversed MaxMatch = ['#', 'w', 'ok']; LogProb = -38.4
    97. #difd
    MaxMatch = ['#', 'di', 'f', 'd']; LogProb = -45.3
    Reversed MaxMatch = ['#', 'd', 'if', 'd']; LogProb = -40.5
    98. #pechanga
    MaxMatch = ['#', 'pech', 'an', 'ga']; LogProb = -47.1
    Reversed MaxMatch = ['#', 'p', 'e', 'changa']; LogProb = -48.0
    99. #singapore
    MaxMatch = ['#', 'sing', 'a', 'pore']; LogProb = -41.3
    Reversed MaxMatch = ['#', 's', 'inga', 'pore']; LogProb = -51.8
    100. #potd
    MaxMatch = ['#', 'pot', 'd']; LogProb = -34.7
    Reversed MaxMatch = ['#', 'po', 'td']; LogProb = -41.3
    101. #mamajeanneandme
    MaxMatch = ['#', 'mam', 'a', 'jeanne', 'and', 'me']; LogProb = -56.6
    Reversed MaxMatch = ['#', 'm', 'ama', 'jeanne', 'and', 'me']; LogProb = -63.2
    102. #thenext5goldenyears
    MaxMatch = ['#', 'then', 'ex', 't', '5', 'golden', 'years']; LogProb = -69.8
    Reversed MaxMatch = ['#', 'the', 'next', '5', 'golden', 'years']; LogProb = -51.3
    103. #meetup
    MaxMatch = ['#', 'meet', 'up']; LogProb = -29.5
    Reversed MaxMatch = ['#', 'me', 'e', 'tup']; LogProb = -45.2
    104. #nickryrie
    MaxMatch = ['#', 'nick', 'r', 'yr', 'ie']; LogProb = -63.1
    Reversed MaxMatch = ['#', 'nick', 'r', 'y', 'rie']; LogProb = -61.3
    105. #statoftheday
    MaxMatch = ['#', 'st', 'at', 'oft', 'he', 'day']; LogProb = -59.1
    Reversed MaxMatch = ['#', 's', 'tat', 'of', 'the', 'day']; LogProb = -52.0
    106. #nced
    MaxMatch = ['#', 'n', 'ce', 'd']; LogProb = -48.4
    Reversed MaxMatch = ['#', 'n', 'c', 'ed']; LogProb = -44.9
    107. #booyaa
    MaxMatch = ['#', 'boo', 'ya', 'a']; LogProb = -43.7
    Reversed MaxMatch = ['#', 'boo', 'y', 'aa']; LogProb = -52.8
    108. #butitsalsokindofaphone
    MaxMatch = ['#', 'but', 'its', 'also', 'kind', 'of', 'a', 'phone']; LogProb = -58.9
    Reversed MaxMatch = ['#', 'bu', 'tits', 'also', 'kin', 'do', 'fa', 'phone']; LogProb = -91.7
    109. #jordaan
    MaxMatch = ['#', 'jo', 'r', 'da', 'an']; LogProb = -53.4
    Reversed MaxMatch = ['#', 'j', 'or', 'da', 'an']; LogProb = -48.7
    110. #myfriendsarebetterthanyours
    MaxMatch = ['#', 'my', 'friends', 'are', 'better', 'than', 'yours']; LogProb = -60.6
    Reversed MaxMatch = ['#', 'my', 'friend', 'sare', 'better', 'than', 'yours']; LogProb = -69.2
    111. #ahimacon14
    MaxMatch = ['#', 'ah', 'ima', 'con', '1', '4']; LogProb = -67.2
    Reversed MaxMatch = ['#', 'a', 'hi', 'macon', '1', '4']; LogProb = -58.8
    112. #uniexamonasaturday
    MaxMatch = ['#', 'unie', 'x', 'am', 'ona', 'saturday']; LogProb = -71.6
    Reversed MaxMatch = ['#', 'u', 'ni', 'ex', 'a', 'mona', 'saturday']; LogProb = -80.0
    113. #ciosummit
    MaxMatch = ['#', 'c', 'ios', 'um', 'mi', 't']; LogProb = -71.9
    Reversed MaxMatch = ['#', 'c', 'io', 'summit']; LogProb = -48.0
    114. #cdnpoli
    MaxMatch = ['#', 'c', 'd', 'n', 'pol', 'i']; LogProb = -63.1
    Reversed MaxMatch = ['#', 'c', 'd', 'n', 'po', 'li']; LogProb = -70.9
    115. #takeabreak
    MaxMatch = ['#', 'take', 'ab', 'reak']; LogProb = -48.2
    Reversed MaxMatch = ['#', 'ta', 'kea', 'break']; LogProb = -51.5
    116. #excitables
    MaxMatch = ['#', 'excitable', 's']; LogProb = -38.9
    Reversed MaxMatch = ['#', 'ex', 'c', 'i', 'tables']; LogProb = -51.8
    117. #skullsearcher
    MaxMatch = ['#', 'skulls', 'ear', 'che', 'r']; LogProb = -61.2
    Reversed MaxMatch = ['#', 'skull', 'searcher']; LogProb = -40.6
    118. #happybirthdaysandarapark
    MaxMatch = ['#', 'happy', 'birthdays', 'anda', 'rap', 'ark']; LogProb = -78.3
    Reversed MaxMatch = ['#', 'happy', 'birthday', 'sand', 'ara', 'park']; LogProb = -68.6
    119. #blackhat
    MaxMatch = ['#', 'black', 'hat']; LogProb = -32.7
    Reversed MaxMatch = ['#', 'b', 'lac', 'khat']; LogProb = -51.4
    120. #ididntchoosethestudentlifeitchoseme
    MaxMatch = ['#', 'id', 'id', 'n', 'tch', 'o', 'ose', 'the', 'student', 'life', 'itch', 'ose', 'me']; LogProb = -143.7
    Reversed MaxMatch = ['#', 'i', 'didnt', 'choose', 'the', 'student', 'life', 'it', 'cho', 'seme']; LogProb = -95.9
    121. #1yearofalmostisneverenough
    MaxMatch = ['#', '1', 'year', 'of', 'almost', 'is', 'never', 'enough']; LogProb = -60.9
    Reversed MaxMatch = ['#', '1', 'year', 'of', 'al', 'mos', 'tis', 'never', 'enough']; LogProb = -86.9
    122. #affsuzukicup
    MaxMatch = ['#', 'a', 'f', 'fs', 'u', 'z', 'u', 'k', 'i', 'cup']; LogProb = -104.6
    Reversed MaxMatch = ['#', 'a', 'f', 'f', 'suz', 'u', 'k', 'i', 'cup']; LogProb = -91.4
    123. #askherforfback
    MaxMatch = ['#', 'ask', 'her', 'for', 'f', 'back']; LogProb = -51.5
    Reversed MaxMatch = ['#', 'ask', 'her', 'f', 'orf', 'back']; LogProb = -60.7
    124. #strictlyus
    MaxMatch = ['#', 'strictly', 'us']; LogProb = -32.0
    Reversed MaxMatch = ['#', 'strict', 'l', 'yus']; LogProb = -50.7
    
    Number of Different Hashtags: 124
    Average MaxMatch LogProb = -56.19
    Average Reversed MaxMatch LogProb = -53.99


# Text Classification (4 marks)

### Question 5 (1.0 mark)

**Instructions**: Here we are interested to do text classification, to predict the country of origin of a given tweet. The task here is to create training, development and test partitions from the preprocessed data (`x_processed`) and convert the bag-of-words representation into feature vectors.

**Task**: Create training, development and test partitions with a 70%/15%/15% ratio. Remember to preserve the ratio of the classes for all your partitions. That is, say we have only 2 classes and 70% of instances are labelled class A and 30% of instances are labelled class B, then the instances in training, development and test partitions should also preserve this 7:3 ratio. You may use sklearn's builtin functions for doing data partitioning.

Next, turn the bag-of-words dictionary of each tweet into a feature vector. You may also use sklearn's builtin functions for doing this (but if you don't want to use sklearn that's fine).

You should produce 6 objects: `x_train`, `x_dev`, `x_test` which contain the input feature vectors, and `y_train`, `y_dev` and `y_test` which contain the labels.


```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

x_train, x_dev, x_test = None, None, None
y_train, y_dev, y_test = None, None, None

###
# Your answer BEGINS HERE
###

vectorizer = DictVectorizer()

# split the train and dev&test data
# use stratify to preserve the ratio of the classes for all partitions
x_train, X_test, y_train, Y_test = train_test_split(x_processed, y_processed, test_size=0.3, stratify=y_processed, random_state=42)

x_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(X_test)

# split the dev and test data
x_dev, x_test, y_dev, y_test = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test, random_state=42)

###
# Your answer ENDS HERE
###
```

### Question 6 (1.0 mark)

**Instructions**: Now, let's build some classifiers. Here, we'll be comparing Naive Bayes and Logistic Regression. For each, you need to first find a good value for their main regularisation hyper-parameters, which you should identify using the scikit-learn docs or other resources. Use the development set you created for this tuning process; do **not** use cross-validation in the training set, or involve the test set in any way. You don't need to show all your work, but you do need to print out the **accuracy** with enough different settings to strongly suggest you have found an optimal or near-optimal choice. We should not need to look at your code to interpret the output.

**Task**: Implement two text classifiers: Naive Bayes and Logistic Regression. Tune the hyper-parameters of these classifiers and print the task performance (accuracy) for different hyper-parameter settings.


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

###
# Your answer BEGINS HERE
###
from sklearn.metrics import accuracy_score

# Tuning Hyperparameters for MultinomialNB
# Alpha: additive smoothing parameter
alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]

print('='*20)
print(f'MultinomialNB\nalpha = {alphas}')

accuracy = []
for a in alphas:
    clf = MultinomialNB(alpha=a)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_dev)
    accuracy.append((a, accuracy_score(y_dev, y_pred)))
    

# print the paramter and accuracy
sorted_alphas = sorted(accuracy, key=lambda x:x[1], reverse=True)
for item in sorted_alphas[:5]:
    print(f'alpha = {item[0]:.2f}, acc = {item[1]:.6f}')

best_alpha = sorted_alphas[0][0]
print(f'Best alpha = {best_alpha}')

# Tuning Hyperparameters for LogisticRegression
# C: Inverse of regularization strength
C = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]

print('='*20)
print(f'LogisticRegression\nC = {C}')

accuracy = []
for c in C:
    clf = LogisticRegression(C=c)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_dev)
    accuracy.append((c, accuracy_score(y_dev, y_pred)))

# print the parameter and accuracy
sorted_C = sorted(accuracy, key=lambda x:(x[1]), reverse=True)
for item in sorted_C[:5]:
    print(f'C = {item[0]:.2f}, acc = {item[1]:.6f}')

best_c = sorted_C[0][0]
print(f'Best C = {best_c}')

###
# Your answer ENDS HERE
###
```

    ====================
    MultinomialNB
    alpha = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]
    alpha = 0.01, acc = 0.276596
    alpha = 0.02, acc = 0.276596
    alpha = 0.05, acc = 0.262411
    alpha = 0.10, acc = 0.262411
    alpha = 0.20, acc = 0.248227
    Best alpha = 0.01
    ====================
    LogisticRegression
    C = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]
    C = 5.00, acc = 0.241135
    C = 10.00, acc = 0.241135
    C = 0.50, acc = 0.234043
    C = 0.20, acc = 0.226950
    C = 1.00, acc = 0.226950
    Best C = 5


### Question 7 (1.0 mark)

**Instructions**: Using the best settings you have found, compare the two classifiers based on performance in the test set. Print out both **accuracy** and **macro-averaged F-score** for each classifier. Be sure to label your output. You may use sklearn's inbuilt functions.

**Task**: Compute test performance in terms of accuracy and macro-averaged F-score for both Naive Bayes and Logistic Regression, using their optimal hyper-parameter settings based on their development performance.


```python
###
# Your answer BEGINS HERE
###
from sklearn.metrics import f1_score

clf_MultinomialNB = MultinomialNB(alpha=best_alpha).fit(x_train, y_train)
y_pred = clf_MultinomialNB.predict(x_test)
acc = accuracy_score(y_test, y_pred)
# Macro-averaged F-score
f1 = f1_score(y_test, y_pred, average='macro')
print('='*20)
print(f'MultinomialNB\nBest alpha={best_alpha}\n')
print(f'Accuracy = {acc:.4f}\nMacro-averaged F-score = {f1:.4f}')

clf_LogisticRegression = LogisticRegression(C=best_c).fit(x_train, y_train)
y_pred = clf_LogisticRegression.predict(x_test)
acc = accuracy_score(y_test, y_pred)
# Macro-averaged F-score
f1 = f1_score(y_test, y_pred, average='macro')
print('='*20)
print(f'LogisticRegression\nBest C={best_c}\n')
print(f'Accuracy = {acc:.4f}\nMacro-averaged F-score = {f1:.4f}')


###
# Your answer ENDS HERE
###
```

    ====================
    MultinomialNB
    Best alpha=0.01
    
    Accuracy = 0.2817
    Macro-averaged F-score = 0.2715
    ====================
    LogisticRegression
    Best C=5
    
    Accuracy = 0.2394
    Macro-averaged F-score = 0.2258


### Question 8 (1.0 mark)

**Instructions**: Print the most important features and their weights for each class for the two classifiers.


**Task**: For each of the classifiers (Logistic Regression and Naive Bayes) you've built in the previous question, print out the top-20 features (words) with the highest weight for each class (countries).

An example output:
```
Classifier = Logistic Regression

Country = au
aaa (0.999) bbb (0.888) ccc (0.777) ...

Country = ca
aaa (0.999) bbb (0.888) ccc (0.777) ...

Classifier = Naive Bayes

Country = au
aaa (-1.0) bbb (-2.0) ccc (-3.0) ...

Country = ca
aaa (-1.0) bbb (-2.0) ccc (-3.0) ...
```

Have a look at the output, and see if you notice any trend/pattern in the words for each country.


```python
###
# Your answer BEGINS HERE
###

# Logistic Regression
print('Classifier = Logistic Regression\n')
LR_weights = clf_LogisticRegression.coef_

# get output feature names for transformation
feature_names = vectorizer.get_feature_names_out(LR_weights)

for i in range(len(clf_LogisticRegression.classes_)):

    # get class names and their feature weights
    class_name = clf_LogisticRegression.classes_[i]
    class_feature = LR_weights[i]
    print(f'Country = {class_name}')

    top20_features_string = ''
    for feature in sorted(zip(class_feature, feature_names), reverse=True)[:20]:
        top20_features_string += f'{feature[1]} ({feature[0]:.3f}) '
    print(top20_features_string)


# Naive Bayes
print('\nClassifier = Naive Bayes\n')
NB_weights = clf_MultinomialNB.feature_log_prob_
feature_names = vectorizer.get_feature_names_out(NB_weights)

for i in range(len(clf_MultinomialNB.classes_)):

    class_name = clf_MultinomialNB.classes_[i]
    class_feature = NB_weights[i]
    print(f'Country = {class_name}')

    top20_features_string = ''
    for feature in sorted(zip(class_feature, feature_names), reverse=True)[:20]:
        top20_features_string += f'{feature[1]} ({feature[0]:.3f}) '
    print(top20_features_string)

###
# Your answer ENDS HERE
###
```

    Classifier = Logistic Regression
    
    Country = au
    australia (1.931) #melbourne (1.836) melbourne (1.783) little (1.671) @micksunnyg (1.610) literally (1.508) summerpoyi's (1.499) https://t.co/7rcjjptvl7 (1.499) lachie (1.473) @dasheryoung (1.473) @whennboys (1.348) @jackgilinsky (1.348) australian (1.317) please (1.303) http://t.co/blhj9cmxit (1.257) @christorrano (1.257) green (1.222) ha (1.215) beach (1.207) @thomjrob (1.149) 
    Country = ca
    @aliclarke (1.906) happen (1.771) hate (1.602) thing (1.521) gonna (1.517) bed (1.464) really (1.451) xoxo (1.419) @cheetahbiatch (1.419) learning (1.320) healthy (1.303) @samanthasharris (1.282) @lola9793 (1.281) #nochillzone (1.272) joking (1.232) right (1.227) cutest (1.200) awesome (1.143) think (1.136) manor (1.131) 
    Country = de
    roseninsel (1.775) https://t.co/df7ficsci3 (1.775) jessica (1.457) hyde (1.457) https://t.co/brkwmsvzrb (1.457) gauting (1.457) #utopia (1.457) #truegrip (1.457) done (1.370) night (1.315) https://t.co/psveyka5g3 (1.301) adlib (1.301) workout (1.273) https://t.co/i7j5pmd3mx (1.273) cannot (1.236) @kelsxmclaughlin (1.236) posted (1.227) yay (1.215) ja (1.153) http://t.co/iqyszvhrus (1.153) 
    Country = gb
    x (2.077) stressed (1.913) this'll (1.663) prom (1.663) interesting (1.663) http://t.co/w0tjrah9y7 (1.663) chelsea (1.500) @jackclaudereadi (1.468) http://t.co/4yniyezb0n (1.444) well (1.426) always (1.424) http://t.co/wbale6zrsq (1.406) favouriteeee (1.406) tweet (1.393) plz (1.379) sure (1.365) @ryan_hildyard (1.325) modelling (1.301) http://t.co/giq4dl9jcq (1.301) seeing (1.267) 
    Country = id
    pic (2.223) http://t.co/0uccot9gdn (2.017) https://t.co/xxbmsbuf0r (1.926) https://t.co/exshmyqmsl (1.847) http://t.co/32szhx0tlk (1.567) world (1.545) @justinbieber (1.534) pranciska (1.522) https://t.co/3sxyzl4crq (1.522) http://t.co/wyldhnlap4 (1.522) @smandatas_261 (1.522) selamat (1.426) perfect (1.338) @viccent22 (1.304) change (1.291) strong (1.249) rt (1.232) sore (1.157) https://t.co/vs4bfdcvdc (1.143) @brunomars (1.143) 
    Country = my
    thank (2.014) @siti_nurbalqish (1.981) http://t.co/hf8harhgbu (1.712) @poemporns (1.712) selangor (1.708) johor (1.658) kl (1.515) depressing (1.513) @markthewise (1.513) @12_maiii (1.513) hunny (1.475) @aqmarnaimi_ (1.475) back (1.454) care (1.429) http://t.co/6qowzr40be (1.427) goodnight (1.378) hahaha (1.319) cave (1.271) http://t.co/utnamcqqx5 (1.264) #gunner (1.264) 
    Country = ph
    @home (1.941) @fuccyoudis2o9 (1.688) @veronicavispo (1.667) #sorrynotsorry (1.667) @qweeesha (1.666) thankyou (1.636) http://t.co/fznyhrrj (1.538) lol (1.488) evening (1.434) ng (1.426) space (1.357) city (1.351) subdivision (1.331) http://t.co/tnbxjbaocu (1.331) fairlane (1.331) ko (1.327) perfection (1.251) pastries (1.251) yet (1.248) dis (1.172) 
    Country = sg
    singapore (3.441) @tutiandonlyher (2.649) https://t.co/potgoiy5rv (2.128) https://t.co/bwenaysyrq (1.701) @moonbowcloth_id (1.701) yeap (1.547) @stylomiloteen (1.547) @megaekaputri_ (1.476) https://t.co/fcnojxrgld (1.408) caterpillar (1.408) heart (1.384) https://t.co/7yswk6lzsi (1.376) https://t.co/rwqvgoxm26 (1.312) @thebeatles (1.312) songs (1.279) hahahha (1.279) keri (1.179) watching (1.165) addictive (1.158) yesterday (1.091) 
    Country = us
    freakin (2.055) much (1.773) http://t.co/ykkwnmzsir (1.683) @savannerdd (1.644) @ansleytolbert (1.644) get (1.618) tonight (1.521) craving (1.510) chipotle (1.510) ahhahaha (1.421) @b_diddddy (1.421) happy (1.361) testimony (1.318) @rcpolar (1.318) beer (1.276) past (1.238) pack (1.238) clue (1.227) better (1.213) ever (1.212) 
    Country = za
    man (2.307) https://t.co/2fsyuigben (1.963) good (1.660) akubabuze (1.600) @zimtweets (1.600) god (1.457) u (1.452) https://t.co/upndf1513q (1.448) @ferlo_mabkay (1.448) goodnight (1.350) feel (1.346) kosciiiiiiielny (1.338) @giftnana7 (1.323) close (1.311) blessed (1.290) @fbotha1 (1.290) sir (1.273) @enyawmm (1.273) difficult (1.262) consultation (1.252) 
    
    Classifier = Naive Bayes
    
    Country = au
    melbourne (-4.590) one (-4.772) i'm (-4.772) great (-4.772) little (-4.994) australia (-4.994) make (-5.281) love (-5.281) keep (-5.281) even (-5.281) come (-5.281) win (-5.685) want (-5.685) visit (-5.685) victoria (-5.685) vca (-5.685) two (-5.685) tomorrow (-5.685) today (-5.685) team (-5.685) 
    Country = ca
    maybe (-4.792) i'm (-4.792) school (-5.014) right (-5.014) like (-5.014) great (-5.014) first (-5.014) u (-5.301) think (-5.301) thing (-5.301) thanks (-5.301) learning (-5.301) got (-5.301) good (-5.301) big (-5.301) year (-5.705) would (-5.705) without (-5.705) way (-5.705) wait (-5.705) 
    Country = de
    i'm (-4.308) posted (-4.594) night (-4.594) year (-4.998) week (-4.998) remstal (-4.998) photodesign (-4.998) photo (-4.998) painting (-4.998) mkmedi (-4.998) miss (-4.998) kernen (-4.998) im (-4.998) good (-4.998) enough (-4.998) done (-4.998) could (-4.998) bad (-4.998) #rosegold (-4.998) yay (-5.686) 
    Country = gb
    sure (-4.696) got (-4.696) x (-4.919) i'm (-4.919) well (-5.206) plz (-5.206) need (-5.206) know (-5.206) home (-5.206) get (-5.206) always (-5.206) yeah (-5.610) xx (-5.610) work (-5.610) want (-5.610) use (-5.610) tweet (-5.610) today (-5.610) time (-5.610) that's (-5.610) 
    Country = id
    pic (-3.981) rt (-4.386) posted (-5.078) photo (-5.078) others (-5.078) i'm (-5.078) hiks (-5.078) way (-5.481) sore (-5.481) shaw (-5.481) senayan (-5.481) selamat (-5.481) room (-5.481) overboard (-5.481) one (-5.481) mikleo (-5.481) love (-5.481) kawah (-5.481) ini (-5.481) hari (-5.481) 
    Country = my
    i'm (-3.497) selangor (-4.323) jaya (-4.659) w (-4.882) thank (-4.882) take (-4.882) petaling (-4.882) johor (-4.882) happy (-4.882) birthday (-4.882) nk (-5.169) ni (-5.169) hahaha (-5.169) haha (-5.169) cendol (-5.169) care (-5.169) world (-5.572) wait (-5.572) u (-5.572) tp (-5.572) 
    Country = ph
    follow (-4.507) city (-4.689) w (-4.912) sm (-4.912) please (-4.912) others (-4.912) manila (-4.912) make (-4.912) i'm (-4.912) yesterday (-5.198) true (-5.198) thing (-5.198) ng (-5.198) mo (-5.198) lol (-5.198) like (-5.198) ka (-5.198) evening (-5.198) dreams (-5.198) come (-5.198) 
    Country = sg
    singapore (-3.772) i'm (-4.224) w (-4.560) others (-4.782) work (-5.069) make (-5.069) week (-5.473) watching (-5.473) universal (-5.473) time (-5.473) studios (-5.473) starbucks (-5.473) side (-5.473) shaw (-5.473) see (-5.473) say (-5.473) posted (-5.473) point (-5.473) pic (-5.473) photo (-5.473) 
    Country = us
    get (-4.443) like (-4.625) tonight (-4.847) i'm (-4.847) happy (-4.847) going (-4.847) u (-5.134) much (-5.134) love (-5.134) ever (-5.134) day (-5.134) birthday (-5.134) beer (-5.134) wait (-5.538) true (-5.538) today (-5.538) time (-5.538) think (-5.538) that's (-5.538) texas (-5.538) 
    Country = za
    good (-4.523) u (-4.705) way (-4.928) love (-4.928) got (-4.928) god (-4.928) de (-4.928) day (-4.928) weekend (-5.215) man (-5.215) made (-5.215) like (-5.215) guys (-5.215) feel (-5.215) yebo (-5.619) years (-5.619) ur (-5.619) twitter (-5.619) today (-5.619) time (-5.619) 



```python

```
