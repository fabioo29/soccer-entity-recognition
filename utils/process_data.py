import os
import json
import nltk
import pandas as pd
import numpy as np

from re import L, findall
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def process_data(dataset_path: str, csv_file: str, vocab_path: str, words_path: str, labels_path: str, embedding_size: int) -> None:
    """ Process the dataset to be used by the NER supervised learning model"""

    stemmer = PorterStemmer() # initialize nltk stemmer

    vocab = open(vocab_path, 'r').read().split('\n') # read vocab file
    stopwords = open('utils/en_stopwords.txt', 'r').read().split('\n') # read stopwords file TODO: fix path
    # read soccer_games.csv transcripts
    data = pd.read_csv(os.path.join(dataset_path, csv_file))

    en_sentences = []
    for sentence in data['Text'].tolist():
        s_words = sentence.split()
        mid_word = int(len(s_words) / 2)
        if s_words[mid_word] in vocab and s_words[mid_word+1] in vocab:
            en_sentences.append(sentence)

    en_dataset = ' '.join(en_sentences)
    
    processed_words = [w for w in word_tokenize(en_dataset) if not w.startswith('[')]
    dataset_words = [w.lower() for w in en_dataset.split() if not w.startswith('[')]
    
    # read utils/en_words.txt
    words = open(os.path.join(dataset_path, words_path)).read().split('\n')
    labels = ['O' for _ in range(len(dataset_words))]
    soccer_data = json.load(open(os.path.join(dataset_path, 'soccer_data.json')))
    ner_tags = open(os.path.join(dataset_path, labels_path)).read().split('\n')
    
    # get words vocab into words.txt file
    saved_vocab = set()
    fp = open(os.path.join(dataset_path, words_path), 'w')
    for w in dataset_words:
        if '/' in w:
            w = w.split('/')[0]
        if w not in saved_vocab: 
            fp.write(w + '\n')
            saved_vocab.add(w)
    fp.write('\nUNK\n')

    # get every Capitalized word from dataset_words and their index into a dict()
    possible_labels = set()
    soccer_values = [item.replace('fc','').replace('cf','').strip() for values in soccer_data.values() for item in values]
    tmp_dataset_words = ' '.join([' '] + dataset_words + processed_words + [' '])
    for idx, w in enumerate(soccer_values): 
        for x in w.split():
            if x in  vocab: # clean dataset names with stop words on it
                break
        if f' {w} ' in tmp_dataset_words:
            possible_labels.add(w)
            
    for idx, ner_data in enumerate(soccer_data.values(), start=1):
        ner_data = [item.replace('fc','').replace('cf','').strip() for item in ner_data]
        for words in tqdm(possible_labels, desc=f'NER tag {ner_tags[idx]}'):
            w_bag = words.split()
            for i, w in enumerate([x.lower().strip() for x in dataset_words]):
                if len(w_bag) == 1:
                    if w in w_bag and words in ner_data:
                        labels[i] = ner_tags[idx]
                else:
                    if i < len(dataset_words) - 1:
                        if f' {dataset_words[i]} {dataset_words[i+1]} ' in f' {words} ' and words in ner_data:
                            labels[i] = ner_tags[idx]
                    if i > 0:    
                        if f' {dataset_words[i-1]} {dataset_words[i]} ' in f' {words} ' and words in ner_data:
                            labels[i] = ner_tags[idx]
    
    # add a new line each embedding_size words
    sentences = [' '.join(dataset_words[i:i+embedding_size]) for i in range(0, len(dataset_words), embedding_size)]
    labels = [' '.join(labels[i:i+embedding_size]) for i in range(0, len(labels), embedding_size)]

    # remove sentences and labels by idx if labels are fullfill with 'O'
    filtered = [(sentence, label) for idx, (sentence, label) in enumerate(zip(sentences, labels)) if labels[idx].count('O') <= embedding_size-2]
    #filtered = [(sentence, label) for idx, (sentence, label) in enumerate(zip(sentences, labels))]
    fsentences, flabels = zip(*filtered)


    # split full dataset(sentences and labels) into tran/val/test sets (0.8/0.1/0.1)
    splits = [int(len(fsentences) * 0.8), int(len(fsentences) * 0.9)]

    train_sentences = fsentences[:splits[0]]
    train_labels = flabels[:splits[0]]

    val_sentences = fsentences[splits[0]:splits[1]]
    val_labels = flabels[splits[0]:splits[1]]

    test_sentences = fsentences[splits[1]:]
    test_labels = flabels[splits[1]:]

    # save tran/val/test sets into files
    os.makedirs(os.path.join(dataset_path, 'train'), exist_ok=True)
    with open(os.path.join(dataset_path, 'train', 'sentences.txt'), 'w') as fp:
        fp.write('\n'.join(train_sentences))
    with open(os.path.join(dataset_path, 'train', 'labels.txt'), 'w') as fp:
        fp.write('\n'.join(train_labels))

    os.makedirs(os.path.join(dataset_path, 'val'), exist_ok=True)
    with open(os.path.join(dataset_path, 'val', 'sentences.txt'), 'w') as fp:
        fp.write('\n'.join(val_sentences))
    with open(os.path.join(dataset_path, 'val', 'labels.txt'), 'w') as fp:
        fp.write('\n'.join(val_labels))

    os.makedirs(os.path.join(dataset_path, 'test'), exist_ok=True)
    with open(os.path.join(dataset_path, 'test', 'sentences.txt'), 'w') as fp:
        fp.write('\n'.join(test_sentences))
    with open(os.path.join(dataset_path, 'test', 'labels.txt'), 'w') as fp:
        fp.write('\n'.join(test_labels))
        

if __name__ == "__main__":
    process_data(
        'dataset',
        'soccer_games.csv',
        os.path.join('utils', 'en_words.txt'),
        'words.txt',
        'tags.txt',
        50
    )