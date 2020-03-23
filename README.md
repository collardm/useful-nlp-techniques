Bag-of-useful Natural Language Processing techniques
=======================

> This notebook aims at gather the fundamentals techniques of NLP, a subfield of linguistics, computer science, information science, and artificial intelligence. Here, you will find how to process and simplify text in order to build features for use in Machine Learning Models.

You can open the [Jupyter](http://jupyter.org/) notebooks with :

* Using [jupyter.org's notebook viewer](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.1-Tokenization.ipynb)
  * note: [github.com's notebook viewer](https://github.com/collardm/useful-nlp-techniques/blob/master/1.1-Tokenization.ipynb) also works but it is slower.
  * by cloning this repository and running Jupyter locally. This option lets you play around with the code. In this case, follow the installation instructions below.
  * or by running the notebooks in [Deepnote](https://beta.deepnote.com). This allows you to play around with the code online in your browser.

# Table of contents

* Installation

* Part 1 : Processing and simplifying text

  * [1.1 Tokenization](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.1-Tokenization.ipynb)
  * [1.2 Stopword Removal](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.2-Stopword-Removal.ipynb)
  * [1.3 Frequency of Words](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.3-Frequency-of-Words.ipynb)
  * [1.4 Stemming](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.4-Stemming.ipynb)
  * [1.5 Parts of Speech](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.5-Parts-of-speech.ipynb)
  * [1.6 Lemmatization](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.6-Lemmatization.ipynb)
  * [1.7 N-Grams](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.7-N-grams.ipynb)

* Part 2 : Building features from text data for ML Models

  * [2.1 Bag of Words Encoding](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.1-Bag-of-words-Encoding.ipynb)
  * [2.2 Bag-of-n-grams Encoding](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.1-Bag-of-n-grams-Encoding.ipynb)
  * [2.3 Bag-of-words using TF-IDF](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.3-Bag-of-words-using-TF-IDF.ipynb)
  * [2.4 Word Embeddings with Word2Vec](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.4-Word-Embeddings-with-Word2Vec.ipynb)
  * [2.5 Feature Hashing](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.5-Feature-Hashing.ipynb)
  * [2.6 Locality-Sensitive Hashing (or LSH)](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.6-Locality-Sensitive-Hashing.ipynb)

# Installation

First, you will need to install [git](https://git-scm.com/), if you don't have it already.

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/collardm/useful-nlp-techniques.git
    $ cd useful-nlp-techniques

Of course, you obviously need Python. Python 3 is already preinstalled on many systems nowadays. You can check which version you have by typing the following command (you may need to replace `python3` with `python`):

    $ python3 --version  # for Python 3

On Linux, unless you know what you are doing, you should use your system's packaging system. For example, on Debian or Ubuntu, type:

    $ sudo apt-get update
    $ sudo apt-get install python3 python3-pip

We need to install several Python libraries that are necessary for this project. For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew).

    $ python3 -m pip install --user --upgrade pip 

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

    $ python3 -m pip install --user --upgrade virtualenv
    $ python3 -m virtualenv -p `which python3` env

This creates a new directory called `env` in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace `` `which python3` `` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

    $ source ./env/bin/activate

On Windows, the command is slightly different:

    $ .\env\Scripts\activate

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the `--user` option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using `sudo pip3` instead of `pip3` on Linux).

    $ python3 -m pip install --upgrade -r requirements.txt

Okay! You can now start Jupyter, simply type:

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree).

# Part 1 : Processing and simplifying text

Text processing is important in Machine Learning, we must :

* Remove noise data
* Minimize the size and dimension of features

## [1.1 Tokenization](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.1-Tokenization.ipynb)

**Tokenization** : *Tokenization* is the process of breaking down or splitting textual data into smaller **meaningful** components called *tokens*.

**NLTK provides multiple tokenizers to handle different tokenization scenarios**

* Sentences versus words
* Means of identifying token breaks
* Trained versus untrained

## [1.2 Stopword Removal](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.2-Stopword-Removal.ipynb)

A *stopword* is a word that has little or no significance to creating language features. (Example: "The", "a", "an", etc...)

Be careful, they are no universal definition, they are language and **domain** specific.

Stopword removal is important because it rids you of the **meaningless worsds** that often greatly outnumber the meaningful words.

## [1.3 Frequency of Words](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.3-Frequency-of-Words.ipynb)

**Frequency Filtering** is the process of removing words (tokens), based upon their frequency in a corpus or document, using frequency to classify them as not containing information pertinent to feature creation.

## [1.4 Stemming](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.4-Stemming.ipynb)

**Stemming** is the process of reducing inflected (or sometimes derived) words to their stem, base, or root form.

## [1.5 Parts of Speech](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.5-Parts-of-speech.ipynb)

**Parts-of-speech** tagging is the process of marking up words in a text (corpus) as corresponding to a part of speech (noun, verb, ..), based on both its definition and its context, i.e, its relationship with adjacent and related words in a phrase, sentence, or paragraph.

## [1.6 Lemmatization](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.6-Lemmatization.ipynb)

**Lemmatization** is the process of grouping together the inflected forms of a word so they can be analyzed as a single item, identified by the word's lemma, or dictionary form

## [1.7 N-Grams](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.7-N-grams.ipynb)

An **n-gram** is a contiguous sequence of n items from a given sample of text. The multiple words combined have different meaning than the words independently.

# Part 2 : Building features from text data for ML Models

Machine Learning algorithms prefer features represented as numbers.
Different techniques allows us to do that:

* One-hot and count vector encoding
* TF-IDF encoding
* Feature hashing
* etc ...

## [2.1 Bag-of-Words Encoding](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.1-Bag-of-words-Encoding.ipynb)

Text (such as a sentence or a document) is represented as the **bag** (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

Here we create our bag-of-words using *CountVectorizer* (Count vector encodings capture in each document tensor the frequency of each word in the vocabulary).

## [2.2 Bag-of-n-grams Encoding](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.2-Bag-of-n-grams-Encoding.ipynb)

We can implement a bag-of-n-grams using count vector encodings.

## [2.3 Bag-of-words using TF-IDF](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.3-Bag-of-words-using-TF-IDF.ipynb)

Another powerful technique is using TF-IDF.  
**Term Frequency-Inverse Document Frequency (TF-IDF)** : TF-IDF models capture how often a word occurs in a document as well relative to how often within the corpus.

Frequently Used in a Single Document : **Might be important**  
Frequently Used in the Corpus : **Likely a common word**

## [2.4 Word Embeddings with Word2Vec](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.4-Word-Embeddings-with-Word2Vec.ipynb)

**Words Embeddings** is a learned representation of text where words of similar meaning have a similar representation. Each word is encoded as a real valued vector/tensor of other words.  
Similarities are identified by calculating the distance between tensors.

**Word2Vec** is a shallow two-layer neural netord trained to reconstruc linguistic context of words.
Two models are possible :

* Continuous bag of words (CBOW) which uses context to predict a word
* And Skip-gram which predicts context from a word

## [2.5 Feature Hashing](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.5-Feature-Hashing.ipynb)

What is **Feature Hashing**:  
It is a mapping of data performed y a *hash function*. Hashes are then stored in a fixed size *hash table* that allows quick lookup and typically much smaller than the data. These reductions can, and often do, cause collisions.

Precisely, the feature hashing or "The Hashing Trick":

* Calculate a hash value for each token
* Distribute the hash values along a tensor of fixed size
* Select a size for the tensor that is adequate to minimize collisions
* Represent the tensor as a sparse array

Feature hashing is great at removing dimensionality, it reduces dimension **but it has an issue** it does not keep similar data points together.

## [2.6 Locality-Sensitive Hashing (or LSH)](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/2.6-Locality-Sensitive-Hashing.ipynb)

Fortunately, the issue of basic feature hashing that does not keep similar data points together can be addressed by using **locality-sensitive hashing**

LSH is an hashing algorithm that maps similar inputs into the same bucket with high probability.

Definitions:

* **Jaccard Index :** A value derived from a formula that determines the similarity of two sets. Value is always between `O` and `1` (0 is not similarity, 1 is identical). Essentially, this is calculated as :
$${len(A \cap B)\over len(A \cup B)}$$

* **Min-hashing :** A hashing function taht determines how similar two sets of items are. Performed by hashing two sets with many different hash functions and identifying how many hash to the same value.

* **Locality-sensitive Hashing :** Grouping together documents that have a *Jaccard Index* above a treshold *t*.