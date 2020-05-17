import argparse
import torch
import numpy as np
import json
import os
from model import RNNModule, get_loss_and_train_op
from dataset import get_data_from_file, get_batches
from predict import predict

with open("config.json") as cfg_file:
    config = json.load(cfg_file)

def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, length, top_k=5):
    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]], dtype=torch.long).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(length):
        ix = torch.tensor([[choice]], dtype=torch.long).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist_url", type=str, help='Genius artist page url')
    parser.add_argument("--length", type=int, default=50, help="Length of the generated lyrics")
    args = parser.parse_args()

    artist_url = args.artist_url
    artist_name = artist_url.split('/')[-1]

    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        artist_name, config['batch_size'], config['seq_size'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = RNNModule(n_vocab, config['seq_size'],
                    config['embedding_size'], config['lstm_size'])
    net = net.to(device)

    state_dict = torch.load(os.path.join('checkpoints', artist_name, 'model.ckpt'))
    net.load_state_dict(state_dict)

    line = ' '.join(predict(device, net, config['initial_words'], n_vocab,
                        vocab_to_int, int_to_vocab, args.length, top_k=5))

    print(line)


    

