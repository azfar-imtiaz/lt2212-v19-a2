import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

import re
from nltk.corpus import stopwords
from collections import Counter, defaultdict

# this function explores all directories within the provided path
# for each sub-folder found, all text files are read into a list of dicts
# each dict here contains filename, filepath and text of file
def load_text_from_files(path_to_files):
    dirs = os.listdir(path_to_files)
    all_files_data = []
    for sub_dir in dirs:
        if sub_dir.startswith(".DS_Store"):
            continue
        full_dir_path = os.path.join(path_to_files, sub_dir)
        files = os.listdir(full_dir_path)
        
        # need to remove this later!
        # files = files[:30]
        
        for filename in files:
            # if file is not .txt format, ignore it
            if not filename.lower().endswith(".txt"):
                continue

            full_file_path = os.path.join(full_dir_path, filename)
            text = ''
            with open(full_file_path, 'r') as rfile:
                text = " ".join(rfile.readlines())
            data = {
                'filename': filename,
                'filepath': full_dir_path,
                'text': text
            }
            all_files_data.append(data)
            
    return all_files_data


# this function strips away all punctuation marks, strips leading and trailing whitespaces and converts to lowercase
def preprocess_text(text):
    # The regex to remove punct marks here needs to be more sophisticated
    # Punctuation marks in the documents are prepended and appended by a space. "U . S .", "department ' s"
    text = text.lower()
    text = " ".join([a for a in text.split() if a not in stopwords.words('english')])
    # regex to convert numbers like 1563 . 1 to 1563.1 or 100 , 000 to 100,000
    text = re.sub(r'(?<=\d)\s([^\w\s]+)(\s|$)(?=\d+)', r'\1', text)
    # remove all punctuation marks that do not occur between numbers, such as "My name, indeed, is Azfar"
    text = re.sub(r'(?<!\d)[^ \w](?!\d)', '', text)
    # replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # rg_punct = r'[^A-Za-z\d ]'
    # text = re.sub(rg_punct, '', text)
    text = text.strip()
    return text


# preprocess text of all files
def preprocess_files_text(all_files_data):
    for index, data in enumerate(all_files_data):
        cleaned_text = preprocess_text(data['text'])
        all_files_data[index]['cleaned_text'] = cleaned_text
    return all_files_data


# this function gets word counts of all words in the corpus
def get_all_word_counts(all_files_data):
    word_counts = defaultdict(lambda: 0)
    for index, data in enumerate(all_files_data):
        cleaned_text = all_files_data[index]['cleaned_text']
        tokens = cleaned_text.split()
        for token in tokens:
            word_counts[token] += 1

    return word_counts


# this function adds word counts in all_files_data per document
# if relevant_words is not None, this means that we get counts for only those words in documents which are in relevant_words
# if relevant_words is None, we get counts of all words in documents
def create_word_counts_per_doc(all_files_data, relevant_words):
    for index, data in enumerate(all_files_data):
        tokens = data['cleaned_text'].split()
        if relevant_words is not None:
            for token in tokens:
                try:
                    relevant_words[token]
                    all_files_data[index][token] = tokens.count(token)
                except KeyError:
                    # this means this word is not in relevant_words; ignore
                    continue
        else:
            for token in tokens:
                all_files_data[index][token] = tokens.count(token)

    return all_files_data


parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()

# add a check to ensure that args.basedims is not greater than or equal to args.svddims, since that isn't possible
if args.basedims and args.svddims:
    if args.basedims >= args.svddims:
        print("The number of dimensions for SVD should be less than the number of base dimensions!")
        exit(1)

# ensure that the output file specified is csv
if not args.outputfile.lower().endswith("csv"):
    print("Please specify an output file with .csv extension!")
    exit(1)

print("Loading data from directory {}.".format(args.foldername))
all_files_data = load_text_from_files(args.foldername)

print("Preprocessing text...")
all_files_data = preprocess_files_text(all_files_data)

relevant_words = None

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))
    # get counts of all words in all documents - this returns a dictionary
    word_counts_all = get_all_word_counts(all_files_data)
    # create a Counter object of all word counts
    word_counts_counter = Counter(word_counts_all)
    # get M most common words from corpus by using most_common method of Counter object
    relevant_words = dict(word_counts_counter.most_common(args.basedims))

# get word counts of all documents, on document level
all_files_data = create_word_counts_per_doc(all_files_data, relevant_words)

# create a dataframe object
df = pd.DataFrame(all_files_data)

# fill all null values with 0
df.fillna(0, inplace=True)

# get all duplicates in the dataframe, dropping columns like filepath, filename, text and cleaned_text which might hinder duplicate detection
duplicates = df.duplicated(subset=[a for a in df.columns if a not in ['filepath', 'filename', 'text', 'cleaned_text']])
for index in range(len(duplicates)):
    if bool(duplicates[index]) is True:
        print("%s/%s will be dropped!" % (df['filepath'].iloc[index], df['filename'].iloc[index]))

# drop all duplicates
df_deduplicated = df.drop_duplicates(subset=[a for a in df.columns if a not in ['filepath', 'filename', 'text', 'cleaned_text']])

# get filenames and filepaths - these will be used for multi-index later
filenames = df_deduplicated['filename']
filepaths = df_deduplicated['filepath']

# drop the filename and filepath columns; these will be later used for a multi-index
df_word_counts = df_deduplicated.drop(['filepath', 'filename', 'text', 'cleaned_text'], axis=1, inplace=False)

if args.svddims:
    if args.svddims >= len(df_word_counts.columns):
        print("The number of dimensions for SVD should be less than the total number of features!")
        exit(1)
word_scores = df_word_counts

# if tfidf is specified, apply it to transform scores
if args.tfidf:
    print("Applying tf-idf to raw counts.")
    tfidf_transformer = TfidfTransformer()
    word_scores = tfidf_transformer.fit_transform(word_scores)

# if svd is specified, use svd to reduce dimensions to specified amount
if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
    svd = TruncatedSVD(n_components=args.svddims)
    word_scores = svd.fit_transform(word_scores)

try:
    # this is needed to convert the sparse matrix to a dataframe; it needs to be convered to a numpy array first
    word_scores = word_scores.toarray()
except:
    # it is already numpy array; it was converted to one through SVD
    pass

try:
    # create a dataframe with the same columns as df_word_counts
    word_scores = pd.DataFrame(word_scores, columns=df_word_counts.columns)
except:
    # if SVD was applied, then the column names from df_word_counts can't be applied since the dimensionality has been changed
    word_scores = pd.DataFrame(word_scores)

# create a multi-index using filepaths and filenames
new_index = pd.MultiIndex.from_tuples(list(zip(filepaths, filenames)))
word_scores.index = new_index

print("Writing matrix to {}.".format(args.outputfile))

word_scores.to_csv(args.outputfile)

