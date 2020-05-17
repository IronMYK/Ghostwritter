import argparse
import torch
import json
import os
from model import RNNModule, get_loss_and_train_op
from dataset import get_data_from_file, get_batches
from generate import predict

with open("config.json") as cfg_file:
    config = json.load(cfg_file)

def main(artist_url):
    artist_name = artist_url.split('/')[-1]
    save_path = os.path.join('checkpoints', artist_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        artist_name, config['batch_size'], config['seq_size'])

    net = RNNModule(n_vocab, config['seq_size'],
                    config['embedding_size'], config['lstm_size'])
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    iteration = 0
    for e in range(50):
        batches = get_batches(in_text, out_text, config['batch_size'], config['seq_size'])
        state_h, state_c = net.zero_state(config['batch_size'])
        
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            
            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x, dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), config['gradients_norm'])

            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, 200),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 1000 == 0:
                predict(device, net, config['initial_words'], n_vocab,
                        vocab_to_int, int_to_vocab, top_k=5)
                torch.save(net.state_dict(),
                    os.path.join(save_path, "model.ckpt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist_url", type=str, help='Genius artist page url')
    args = parser.parse_args()
    main(args.artist_url)