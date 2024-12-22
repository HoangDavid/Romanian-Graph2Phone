import torch.nn as nn
import random
import torch as tc
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as f
import time
import json
import os

# Ensure the model directory exists
os.makedirs('model', exist_ok=True)

def dict_load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


class G2P(nn.Module):
    def __init__(self, hidden_size, lr, lr_decay=0.5, min_lr=1e-5) -> None:
        super(G2P, self).__init__()
        self.x_train = tc.load('data_mx/X_train.pt')
        self.y_train = tc.load('data_mx/Y_train.pt')
        self.x_dev = tc.load('data_mx/X_dev.pt')
        self.y_dev = tc.load('data_mx/Y_dev.pt')
        self.x_test = tc.load('data_mx/X_test.pt')
        self.y_test = tc.load('data_mx/Y_test.pt')

        input_size = self.x_train.shape[2]
        output_size = self.y_train.shape[2]

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.decoder = Decoder(
            output_size=output_size,
            output_seq_len=self.y_train.shape[1],
            hidden_size=hidden_size * 2  # Multiply by 2 for bidirectional
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = tc.optim.Adam(self.parameters(), lr=lr)  # Changed to Adam
        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr

        self.graph2idx = dict_load('indexing/graph2idx.json')
        self.idx2phone = dict_load('indexing/idx2phone.json')

    def train_model(self, batch_size, epochs, log_every):
        dataset = TensorDataset(self.x_train, self.y_train)
        start_time = time.time()
        train_loss = 0
        n_total = 0

        best_val_loss = float('inf')
        val_loss = 0
        bad_loss = 0
        min_bad_loss = 4

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            current_batch = 0
            stop = False

            for batch_x, batch_y in batches:
                batch_x = batch_x.permute(1, 0, 2)  # [seq_len, batch, dim]
                batch_y = batch_y.permute(1, 0, 2)  # [seq_len, batch, dim]

                encoder_outputs, h = self.encoder(batch_x)
                sos_token = tc.zeros(1, batch_y.shape[1], self.decoder.output_size, device=batch_x.device)
                preds, targets = self.decoder(sos_token, h, encoder_outputs, seq_y=batch_y)

                loss = self.criterion(preds, targets)
                loss.backward()
                tc.nn.utils.clip_grad_norm_(self.parameters(), 2.3, 'inf')
                self.optimizer.step()
                self.optimizer.zero_grad()

                current_batch += 1
                n_total += batch_x.shape[1]

                train_loss += loss.item() * batch_x.shape[1]

                if current_batch % log_every == 0:
                    val_loss = self.validation()
                    print(f'Time: {time.time() - start_time:5.0f} | '
                          f'Train loss: {train_loss / n_total :.5f} | '
                          f'Val loss: {val_loss :.5f} | '
                          f'Batch: {current_batch : 4.0f} / {len(batches)} ')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        bad_loss = 0
                        tc.save(self.state_dict(), 'model/g2p_best_state.pt')
                    else:
                        bad_loss += 1

                    if bad_loss > min_bad_loss:
                        self.lr *= self.lr_decay
                        print(f'=> Change learning rate to: {self.lr}')
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr
                        bad_loss = 0

                    if self.lr < self.min_lr:
                        stop = True
                        break

            if stop:
                tc.save(self.state_dict(), 'model/g2p.pt')
                print("Early stopping triggered.")
                break

    def validation(self):
        val_loss = 0
        dataset = TensorDataset(self.x_dev, self.y_dev)
        batches = DataLoader(dataset, batch_size=1)

        with tc.no_grad():
            for batch_x, batch_y in batches:
                batch_x = batch_x.permute(1, 0, 2)  # [seq_len, batch, dim]
                batch_y = batch_y.permute(1, 0, 2)  # [seq_len, batch, dim]
                encoder_outputs, h = self.encoder(batch_x)
                sos_token = tc.zeros(1, batch_y.shape[1], self.decoder.output_size, device=batch_x.device)
                preds, targets = self.decoder(sos_token, h, encoder_outputs, seq_y=batch_y, predict=True)
                loss = self.criterion(preds, targets)
                val_loss += loss.item()

        return val_loss / len(batches)

    def test(self):
        dataset = TensorDataset(self.x_test, self.y_test)  # Changed to test set
        batches = DataLoader(dataset, batch_size=1, shuffle=True)
        self.load_state_dict(tc.load('model/g2p_best_state.pt'))

        all_preds = []
        all_targets = []
        with tc.no_grad():
            for batch_x, batch_y in batches:
                batch_x = batch_x.permute(1, 0, 2)  # [seq_len, batch, dim]
                batch_y = batch_y.permute(1, 0, 2)  # [seq_len, batch, dim]

                encoder_outputs, h = self.encoder(batch_x)
                sos_token = tc.zeros(1, batch_y.shape[1], self.decoder.output_size, device=batch_x.device)
                preds, targets = self.decoder(sos_token, h, encoder_outputs, seq_y=batch_y, predict=True)
                
                preds = f.softmax(preds.permute(2, 0, 1), dim=2)
                preds = tc.argmax(preds, dim=2).permute(1, 0)

                all_preds.append(preds)
                all_targets.append(targets)

        all_preds = tc.cat(all_preds, dim=0)
        all_targets = tc.cat(all_targets, dim=0)
        
        print(f'PER (Phoneme Error Rate): {self.PER(all_preds, all_targets) * 100:.3f}%')
        print(f'WER (Word Error Rate): {self.WER(all_preds, all_targets) * 100:.3f}%')

    def PER(self, preds, targets):
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape."

        # Flatten all phonemes for PER calculation
        flat_preds = preds.view(-1)
        flat_targets = targets.view(-1)

        # Phoneme Error Rate (PER)
        total_phonemes = len(flat_targets)
        phoneme_errors = (flat_preds != flat_targets).sum().item()
        per = phoneme_errors / total_phonemes

        return per
    
    def WER(self, preds, targets):
        # Ensure predictions and targets are the same shape
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape."

        # Word Error Rate (WER)
        word_errors = sum([pred.tolist() != target.tolist() for pred, target in zip(preds, targets)])
        total_words = len(targets)
        wer = word_errors / total_words

        return wer
    
    def user_test(self, ip, fixed_size=21, fixed_classes=32):
        ip= ip.lower()
        input_indices = [self.graph2idx[char] for char in ip if char in self.graph2idx]
        one_hot = tc.zeros((fixed_size, 1, len(self.graph2idx)))
        for i, index in enumerate(input_indices[:fixed_size]):
            one_hot[i, 0, index] = 1.0

        one_hot_input = tc.zeros((fixed_size, 1, fixed_classes))
        for i, index in enumerate(input_indices[:fixed_size]):  # Truncate if longer than fixed_size
            one_hot_input[i, 0, index] = 1.0
        
        batch_x = one_hot_input.permute(1, 0, 2)
        self.load_state_dict(tc.load('model/g2p_best_state.pt'))
        with tc.no_grad():
            encoder_outputs, h = self.encoder(batch_x.permute(1, 0, 2))
            sos_token = tc.zeros(1, batch_x.size(0), self.decoder.output_size, device=batch_x.device)
            preds, _ = self.decoder(sos_token, h, encoder_outputs, seq_y=None, predict=True)

        preds = f.softmax(preds.permute(2, 0, 1), dim=2)
        predicted_indices = tc.argmax(preds, dim=2).permute(1, 0)
        output_indices = predicted_indices.squeeze(0).tolist()

        phoneme_sequence = [
            self.idx2phone[str(idx)] for idx in output_indices if self.idx2phone[str(idx)] not in ("<eos>", "<pad>")
        ]

        phoneme_string = " ".join(phoneme_sequence)

        return phoneme_string
        

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True) -> None:
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=bidirectional)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, seq_x):
        encoder_outputs, h = self.rnn(seq_x)  # [seq_len, batch, hidden_size * num_directions]
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            h = tc.cat((h[-2, :, :], h[-1, :, :]), dim=1).unsqueeze(0)  # [1, batch, hidden_size * 2]
        return encoder_outputs, h


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, decoder_hidden_dim]
        # encoder_outputs: [seq_len, batch, encoder_hidden_dim]
        seq_len = encoder_outputs.size(0)

        # Repeat hidden state seq_len times
        hidden = hidden.repeat(seq_len, 1, 1)  # [seq_len, batch, decoder_hidden_dim]

        # Concatenate hidden and encoder outputs
        concatenated = tc.cat((hidden, encoder_outputs), dim=2)  # [seq_len, batch, encoder_hidden_dim + decoder_hidden_dim]

        energy = tc.tanh(self.attn(concatenated))  # [seq_len, batch, decoder_hidden_dim]

        # Compute attention scores
        attention = self.v(energy).squeeze(2)  # [seq_len, batch]

        # Apply softmax over the sequence length dimension
        attention = f.softmax(attention, dim=0)  # [seq_len, batch]

        return attention


class Decoder(nn.Module):
    def __init__(self, output_size, output_seq_len, hidden_size) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_seq_len = output_seq_len
        self.teacher_force_prob = 0.5

        # Input now includes context vector from attention
        self.rnn = nn.RNN(output_size + hidden_size, hidden_size)
        # Correctly set decoder_hidden_dim to hidden_size (800)
        self.attention = Attention(encoder_hidden_dim=hidden_size, decoder_hidden_dim=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        self.phone2idx = dict_load('indexing/phone2idx.json')

    def forward(self, sos_token, h, encoder_outputs, seq_y=None, predict=False):
        # sos_token: [1, batch, output_size]
        # h: [1, batch, hidden_size]
        # encoder_outputs: [seq_len, batch, encoder_hidden_dim]
        preds = []
        targets = None
        input_at_t = sos_token  # [1, batch, output_size]

        for t in range(self.output_seq_len):
            # Compute attention weights
            attn_weights = self.attention(h, encoder_outputs)  # [seq_len, batch]
            attn_weights = attn_weights.permute(1, 0).unsqueeze(1)  # [batch, 1, seq_len]

            encoder_outputs_transposed = encoder_outputs.transpose(0, 1)  # [batch, seq_len, encoder_hidden_dim]

            # Perform batch matrix multiplication
            context = tc.bmm(attn_weights, encoder_outputs_transposed)  # [batch, 1, encoder_hidden_dim]

            # Transpose context to match RNN input requirements
            context = context.transpose(0, 1)  # [1, batch, encoder_hidden_dim]

            # Concatenate input and context
            rnn_input = tc.cat((input_at_t, context), dim=2)  # [1, batch, output_size + encoder_hidden_dim]

            # Pass through RNN
            output, h = self.rnn(rnn_input, h)  # output: [1, batch, hidden_size]

            # Generate prediction
            out = self.linear(output)  # [1, batch, output_size]
            preds.append(out)

            if predict:
                # During prediction, use the predicted token as next input
                input_token = f.softmax(out, dim=2)
                input_token = tc.argmax(input_token, dim=2).squeeze(0)  # [batch]
                input_at_t = f.one_hot(input_token, num_classes=self.output_size).float().unsqueeze(0)  # [1, batch, output_size]
            else:
                # During training, decide whether to use teacher forcing
                if random.random() < self.teacher_force_prob and seq_y is not None:
                    input_at_t = seq_y[t].unsqueeze(0)  # [1, batch, output_size]
                else:
                    input_token = f.softmax(out, dim=2)
                    input_token = tc.argmax(input_token, dim=2).squeeze(0)  # [batch]
                    input_at_t = f.one_hot(input_token, num_classes=self.output_size).float().unsqueeze(0)  # [1, batch, output_size]

        # Concatenate all predictions
        preds = tc.cat(preds, dim=0)  # [seq_len, batch, output_size]
        preds = preds[:self.output_seq_len-1, :, :]  # [seq_len-1, batch, output_size]
        preds = preds.permute(1, 2, 0)  # [batch, output_size, seq_len-1]

        if seq_y is not None:
            targets = tc.argmax(seq_y, dim=2)  # [seq_len, batch]
            targets = targets[1:, :]  # [seq_len-1, batch]
            targets = targets.transpose(0,1)  # [batch, seq_len-1]

        return preds, targets

