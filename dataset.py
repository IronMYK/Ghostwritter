import os
from collections import Counter
import numpy as np
import re

def get_all_lyrics(artist):
    file_paths = os.listdir(os.path.join('data', artist))
    lyrics = []
    for file_path in file_paths:
        path = os.path.join('data', artist, file_path)
        with open(path, 'r', encoding='utf-8') as f:
            lyrics.append(f.read())
    
    return '\n'.join(lyrics).lower()

def preprocess(lyrics):
    lyrics = re.sub(r'\[.*\]', ' ', lyrics)
    lyrics = re.sub(r'x\d', ' ', lyrics)
    lyrics = re.sub(r'\*.*\*', ' ', lyrics)
    lyrics = re.sub(r'[\(\)!?,:;\.]', ' ', lyrics)

    lyrics = [line for line in lyrics.split('\n') if line.strip() != '']
    return '\n'.join(lyrics)

def get_data_from_file(artist_name, batch_size, seq_size):
    lyrics = get_all_lyrics(artist_name)
    lyrics = preprocess(lyrics)
    lyrics = lyrics.split()
    word_counts = Counter(lyrics)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)
    
    int_text = [vocab_to_int[w] for w in lyrics]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


if __name__ == '__main__':
    lyrics = get_all_lyrics('Alpha-wann')
    lyrics = preprocess(lyrics)


