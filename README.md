# Text preprocessing module
Note that only French and English are supported on our end. Support for all languages in Spacy will be added at later time.

## Installation
- Just run pip3 install . from the root of the directory
- You may need to install data files if you're going to use PoS tagging. For that, you should head to the spacy.io website for instructions. 

## How to use
- You need to import the PreProcessing class in your module, and instantiate it as below (all keywords below are defaults)

```python
from text_preprocessing import PreProcessing

preproc = Preprocessing(
        word_regex=r"\w+",          # regex for tokenizing if passing a string
        sentence_regex=r"[.!?]+",   # regex for senting splitting if passing a string
        language="french",
        stemmer=False,
        lemmatizer=None,            # path to file with form/lemma separated by tab, or just "spacy to spacy lemmatizer
        modernize=False,
        ngrams=None,
        ngram_gap=0,
        ngram_word_order=True,
        stopwords=None,             # path to file
        strip_punctuation=True,
        strip_numbers=True,
        strip_tags=False,
        lowercase=True,
        min_word_length=2,
        ascii=False,                # convert tokens to ASCII representation
        convert_entities=False,
        with_pos=False,             # return token object with PoS info
        pos_to_keep=[],             # return tokens that match list of POS (for POS available, see Spacy docs)
        is_philo_db=False,          # if parsing from a words_and_philo_ids file generated by PhiloLogic
        text_object_type="doc",     # define text object using PhiloLogic text object model
        return_type"words",         # return a list of words, or list of sentences
        hash_tokens=False, 
        workers=None,               # number of workers
        post_processing_function=None,  # optional post-processing function before each text object is returned
        progress=True               # show progress info
        )

```

- To process a list of files, you just pass a list of files to process_files(). It will return a Tokens object (which is very much like a list of strings with some added features) which contains a list of Token objects:
```python
for text_object in preproc.process_files(["file1", "file2"]):
        resulting_string = " ".join(text_object) # create a string containing every token separated by a space
        surface_forms = " ".join(token.surface_form for token in text_object) # create a string containing every surface form of a token separated by a space
        print(text_object.metadata) # print the text object metadata (a dictionary)
 ```
 
 - process_files takes an optional keep_all=True keyword that will store all filtered words in the surface_form attribute of Token objects (e.g. "token.surface_form")
 
 - Token objects have four attributes:
    - text: the final form of the token after processing
    - surface_form: the original form of the token before processing. If `keep_all=True` was passed to process_files, you will also find tokens that have been filtered out.
    - pos_ = POS speech info if available
    - ext = contains additional metadata from PhiloLogic such as start_byte and end_byte. 
    
    To print the contents of each token, you can do: 
    ```python 
    print(repr(token))
    ```
- You can iterate over every token in returned text objects like so:
```python
for text_object in preproc.process_files(["file1", "file2"]):
        for token in text_object:
                print(token) # shorthand for print(token.text)
                print(token.pos_) # print part of speech 
                print(token.ext["start_byte"], token.ext["end_byte"]) # print start and end byte of each token (if available)
 ``` 
 
