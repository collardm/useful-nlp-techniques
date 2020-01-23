Bag-of-useful Natural Language Processing techniques
=======================

> This notebook aims at gather the fundamentals techniques of NLP, a subfield of linguistics, computer science, information science, and artificial intelligence. Here, you will find how to process and simplify text in order to build features for use in Machine Learning Models.

You can open the [Jupyter](http://jupyter.org/) notebooks with :

* Using [jupyter.org's notebook viewer](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.1-Tokenization.ipynb)
    * note: [github.com's notebook viewer](https://github.com/collardm/useful-nlp-techniques/blob/master/1.1-Tokenization.ipynb) also works but it is slower and the math formulas are not displayed correctly,
* by cloning this repository and running Jupyter locally. This option lets you play around with the code. In this case, follow the installation instructions below,
* or by running the notebooks in [Deepnote](https://beta.deepnote.com). This allows you to play around with the code online in your browser. For example, here's a link to the first chapter: [<img height="22"  src="https://beta.deepnote.com/buttons/launch-in-deepnote.svg">](https://)


# Table of contents

* Installation

* Part 1 : Processing and simplifying text
    * [1.1 Tokenization](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.1-Tokenization.ipynb)
    * [1.2 Stopword Removal](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.2-Stopword-Removal.ipynb)
    * [1.3 Frequency of Words](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.3-Frequency-of-Words.ipynb)
    * [1.4 Stemming](http://nbviewer.jupyter.org/github/collardm/useful-nlp-techniques/blob/master/1.4-Stemming.ipynb)

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

Stopword removal is important because it rids you of the **meaningless words** that often greatly outnumber the meaningful words.

