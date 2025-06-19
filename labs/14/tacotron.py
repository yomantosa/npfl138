#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch

import npfl138
npfl138.require_version("2425.14.2")
from npfl138.datasets.tts_dataset import TTSDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--attention_dim", default=128, type=int, help="Attention dimension.")
parser.add_argument("--attention_rnn_dim", default=1024, type=int, help="Attention RNN dimension.")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--dataset", default="ljspeech_tiny", type=str, help="TTS dataset to use.")
parser.add_argument("--decoder_dim", default=1024, type=int, help="Decoder dimension.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--encoder_layers", default=3, type=int, help="Encoder CNN layers.")
parser.add_argument("--encoder_dim", default=512, type=int, help="Dimension of the encoder.")
parser.add_argument("--locations_filters", default=32, type=int, help="Location sensitive attention filters.")
parser.add_argument("--locations_kernel", default=31, type=int, help="Location sensitive attention kernel.")
parser.add_argument("--hop_length", default=256, type=int, help="Hop length.")
parser.add_argument("--mels", default=80, type=int, help="Mel filterbanks.")
parser.add_argument("--postnet_dim", default=512, type=int, help="Post-net channels.")
parser.add_argument("--postnet_layers", default=5, type=int, help="Post-net CNN layers.")
parser.add_argument("--prenet_dim", default=200, type=int, help="Pre-net dimensions.")
parser.add_argument("--prenet_layers", default=2, type=int, help="Pre-net layers.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--sample_rate", default=22050, type=int, help="Sample rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window_length", default=1024, type=int, help="Window length.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Encoder(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, num_characters: int) -> None:
        super().__init__()
        # TODO: Create the required encoder layers. The architecture of the encoder is as follows:
        # - Convert the texts to embeddings using `torch.nn.Embedding(num_characters, args.encoder_dim)`.
        # - Pass the result throught a `torch.nn.Dropout(args.dropout)` layer.
        # - After moving the channels to the front (i.e., dim=1), pass the result through
        #   `args.encoder_layers` number of layers, each consisting of
        #   - 1D convolution with `args.encoder_dim` channels, kernel size 5, padding 2, and no bias,
        #   - batch normalization,
        #   - ReLU activation.
        # - Then move the channels back to the last dimension.
        # - Perform a `torch.nn.Dropout(args.dropout)` layer.
        # - Apply a bidirectional LSTM layer with `args.encoder_dim` dimension. Correctly handle
        #   the variable-length texts that use `TTSDataset.PAD` padding value. The results from the
        #   forward and backward direction should be summed together.
        # - Pass the result through another `torch.nn.Dropout(args.dropout)` layer and return it.
        # It does not matter if you use a shared single dropout layer or invididual independent dropout layers.
        # raise NotImplementedError()
        self.embedding = torch.nn.Embedding(num_characters, args.encoder_dim)
        self.drop1 = torch.nn.Dropout(args.dropout)

        # Build CNN stack
        self.cnn_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(args.encoder_dim, args.encoder_dim, kernel_size=5, padding=2, bias=False),
                torch.nn.BatchNorm1d(args.encoder_dim),
                torch.nn.ReLU()
            )
            for _ in range(args.encoder_layers)
        ])

        # Bidirectional LSTM
        self.bi_lstm = torch.nn.LSTM(
            args.encoder_dim, args.encoder_dim,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.drop2 = torch.nn.Dropout(args.dropout)

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass of the encoder.
        hidden = self.embedding(texts)
        hidden = self.drop1(hidden)

        hidden = hidden.transpose(1, 2)
        for block in self.cnn_blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = self.drop2(hidden)

        lengths = (texts != TTSDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, lengths,
                                                         batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bi_lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        feat_dim = self.bi_lstm.hidden_size
        summed = lstm_out[:, :, :feat_dim] + lstm_out[:, :, feat_dim:]
        result = self.drop2(summed)

        if npfl138.first_time("Encoder.forward"):
            print(f"The torch.std of the first batch returned by Encoder: {torch.std(result):.4f}")

        return result


class Attention(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # The architecture of the attention is recurrent, depending on its previous outputs.
        # The required layers are already prepared for you.
        self.attention_rnn = torch.nn.LSTMCell(args.prenet_dim + args.encoder_dim, args.attention_rnn_dim)
        self.location_sensitive_conv = torch.nn.Conv1d(
            2, args.locations_filters, args.locations_kernel, padding=args.locations_kernel // 2)
        self.location_sensitive_output = torch.nn.Linear(args.locations_filters, args.attention_dim)
        self.attention_query_layer = torch.nn.Linear(args.attention_rnn_dim, args.attention_dim)
        self.attention_memory_layer = torch.nn.Linear(args.encoder_dim, args.attention_dim)
        self.attention_output_layer = torch.nn.Linear(args.attention_dim, 1)

    def reset(self, text: torch.Tensor, encoded_text: torch.Tensor) -> None:
        # TODO: The `reset` method initializes the attention module for a new batch of texts.
        # - `texts` is a batch of input texts with shape `[batch_size, max_input_length]`;
        # - `encoded_text` is the output of the encoder for these texts with the shape
        #   of `[batch_size, max_input_length, encoder_dim]`.
        # You should:
        # - store the `encoded_text` in this attention module instance (i.e., in `self`),
        # - process the `encoded_text` using `self.attention_memory_layer` and store the result,
        # - store an attention mask that is `-1e9` for the `TTSDataset.PAD` values in the `text`
        #   and 0 otherwise (it will be used to mask the attention logits before applying softmax),
        # - you should zero-initialize the following `self` variables:
        #   - the state (`h`) and memory cell (`c`) of the `self.attention_rnn`,
        #   - the previously-computed attention weights,
        #   - the cummulative attention weights (the sum of all computed attention weights so far),
        #   - the previously-computed attention context vector (the previous attention output).
        # raise NotImplementedError()
        self.enc_seq = encoded_text
        self.proc_mem = self.attention_memory_layer(encoded_text)
        mask = torch.where(
            text == TTSDataset.PAD,
            torch.tensor(-1e9, device=text.device),
            torch.tensor(0.0, device=text.device))
        self.mask = mask

        batch_size, seq_len, _ = encoded_text.size()
        dev = text.device
        h0 = torch.zeros(batch_size, self.attention_rnn.hidden_size, device=dev)
        c0 = torch.zeros(batch_size, self.attention_rnn.hidden_size, device=dev)

        self.att_h = h0
        self.att_c = c0
        self.prev_weights = torch.zeros(batch_size, seq_len, device=dev)
        self.cum_weights = torch.zeros(batch_size, seq_len, device=dev)
        self.prev_context = torch.zeros(batch_size, encoded_text.size(-1), device=dev)

    def forward(self, prenet: torch.Tensor) -> torch.Tensor:
        # TODO: Implement a single step of the attention mechanism, relying on the previously-computed
        # values stored in `self`.
        # - First run the `self.attention_rnn` on the concatenation of the pre-net output and the previous
        #   context vector, using and updating the stored attention RNN state and memory cell values.
        #   The resulting RNN state (`h`) will be used as the query for the attention.
        # - Then perform the location-sensitive part of the attention, by:
        #   - concatenating the stored cummulative attention weights and previous attention weights,
        #   - passing them through the `self.location_sensitive_conv` layer,
        #   - moving the channels to the last dimension,
        #   - passing the result through the `self.location_sensitive_output` layer.
        # - Now compute the attention logits for each position in the encoded text:
        #   - compute the query by passing the attention RNN state through `self.attention_query_layer`;
        #     this query is used for all positions in the encoded text,
        #   - add the already pre-computed output of the `self.memory_layer` applied to the `encoded_text`,
        #   - add the already computed output of the `self.location_sensitive_output` layer,
        #   - apply the tanh activation and pass the result through the `self.attention_output_layer`,
        #   - add the previously-computed attention mask to the attention logits to mask the padding positions.
        # - Now compute the attention weights (=probabilities) by applying softmax to the attention logits.
        # - Finally compute the attention context vector as the weighted sum of the `encoded_text` and return it.
        # During the computation, you need to store the following values in the attention module instance:
        # - the updated attention RNN state and memory cell,
        # - the updated cummulative attention weights (the sum of all computed attention weights so far),
        # - the current attention weights, which become the previous attention weights in the next step,
        # - the current attention context vector, which becomes the previous context in the next step.
        
        inp = torch.cat([prenet, self.prev_context], dim=-1)
        self.att_h, self.att_c = self.attention_rnn(inp, (self.att_h, self.att_c))
        loc = torch.stack([self.cum_weights, self.prev_weights], dim=1)
        loc = self.location_sensitive_conv(loc).transpose(1, 2)
        loc = self.location_sensitive_output(loc)
        qry = self.attention_query_layer(self.att_h).unsqueeze(1)
        scores = self.attention_output_layer(torch.tanh(qry + self.proc_mem + loc)).squeeze(-1)
        scores = scores + self.mask

        weights = torch.softmax(scores, dim=-1)
        self.prev_weights = weights
        self.cum_weights += weights

        context = torch.sum(weights.unsqueeze(-1) * self.enc_seq, dim=1)
        self.prev_context = context

        if npfl138.first_time("Attention.forward"):
            print(f"The torch.std of the first batch returned by Attention: {torch.std(context):.4f}")

        return context


class Decoder(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Create the layers of the decoder. Start by defining the pre-net module, which is
        # composed of `args.prenet_layers` layers, each consisting of:
        # - a linear layer with `args.prenet_dim` output dimension,
        # - ReLU activation,
        # - dropout with `args.dropout` rate.
        layers = []
        for i in range(args.prenet_layers):
            in_dim = args.mels if i == 0 else args.prenet_dim
            layers.extend([
                torch.nn.Linear(in_dim, args.prenet_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(args.dropout)
            ])
        self.prenet = torch.nn.Sequential(*layers)


        # The LSTM decoder cell is already prepared for you.
        self.decoder = torch.nn.LSTMCell(args.prenet_dim + args.encoder_dim, args.decoder_dim)

        # The `decoder_start` is a learnable parameter that is used as the initial input to the decoder.
        self.decoder_start = torch.nn.Parameter(torch.zeros(args.prenet_dim, dtype=torch.float32))

        # TODO: Create the output layer with no activation that maps decoder states
        # to mel spectrograms with `args.mels` output channels.
        self.output_layer = torch.nn.Linear(args.decoder_dim, args.mels)

        # TODO: Create the gate layer that maps the decoder states to a single value predicting
        # whether this step of the decoder should be the last one.
        self.gate_layer = torch.nn.Linear(args.decoder_dim, 1)

    def reset(self, texts: torch.Tensor) -> None:
        # TODO: Similarly to the `Attention.reset`, the `reset` method initializes the decoder
        # for a new batch of texts. You should
        # - store properly tiled (repeated) `self.decoder_start` as the next input to the decoder,
        # - zero-initialize the decoder state (`h`) and memory cell (`c`) of the `self.decoder`.
        # raise NotImplementedError()
        bsz = texts.size(0)
        self.next_in = self.decoder_start.unsqueeze(0).expand(bsz, -1).to(texts.device)
        h0 = torch.zeros(bsz, self.decoder.hidden_size, device=texts.device)
        c0 = torch.zeros(bsz, self.decoder.hidden_size, device=texts.device)
        self.dec_h = h0
        self.dec_c = c0

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement a single step of the decoder.
        #
        # - First run the `self.decoder` on the concatenation the stored next decoder input and
        #   the given context vector, using and updating the stored decoder RNN state and memory cell values.
        # - Then pass the computed decoder RNN state through the `self.output_layer` to obtain
        #   the predicted `mel_frame` for the current step.
        # - The `mel_frame` is passed through the `self.prenet` to obtain the next input to the decoder,
        #   which should be stored as an instance variable of `self`.
        # - Finally, pass the decoder RNN state through the `self.gate_layer` and a sigmoid activation
        #   to obtain the gate output indicating whether the decoder should stop or continue.
        # Return the output mel spectrogram frame and the gate output.
        inp = torch.cat([self.next_in, context], dim=-1)
        self.dec_h, self.dec_c = self.decoder(inp, (self.dec_h, self.dec_c))

        mel_frame = self.output_layer(self.dec_h)
        gate = torch.sigmoid(self.gate_layer(self.dec_h))

        self.next_in = self.prenet(mel_frame)

        if npfl138.first_time("Decoder.forward"):
            print("The torch.std of the first batch returned by Decoder:",
                  f"({torch.std(mel_frame):.4f}, {torch.std(gate):.4f})")

        return mel_frame, gate

class Postnet(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: The post-net is composed of `args.postnet_layers` convolutional layers.
        # - The first `args.postnet_layers - 1` of them consist of
        #   - a 1D convolution with `args.postnet_dim` output channels, kernel size 5, padding 2, and no bias,
        #   - a batch normalization,
        #   - the tanh activation.
        # - The last layer consists of a 1D convolution with the same hyperparameters, but with `args.mels`
        #   output channels, followed by a batch normalization; no activation is applied.
        # raise NotImplementedError()
        self.blocks = torch.nn.ModuleList()

        if args.postnet_layers == 1:
            conv = torch.nn.Conv1d(args.mels, args.mels, kernel_size=5, padding=2, bias=False)
            self.blocks.append(torch.nn.Sequential(conv, torch.nn.BatchNorm1d(args.mels)))
        else:
            for i in range(args.postnet_layers - 1):
                in_ch = args.mels if i == 0 else args.postnet_dim
                conv = torch.nn.Conv1d(in_ch, args.postnet_dim, kernel_size=5, padding=2, bias=False)
                self.blocks.append(torch.nn.Sequential(conv, torch.nn.BatchNorm1d(args.postnet_dim), torch.nn.Tanh()))
                
            conv = torch.nn.Conv1d(args.postnet_dim, args.mels, kernel_size=5, padding=2, bias=False)
            self.blocks.append(torch.nn.Sequential(conv, torch.nn.BatchNorm1d(args.mels)))


    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        # TODO: Given a batch of mel spectrograms with shape `[batch_size, max_spectrogram_len, mels]`,
        # - move the channels to the front (i.e., dim=1),
        # - pass the spectrograms through the post-net,
        # - move the channels back to the last dimension.
        # Finally, return the sum of the original and processed spectrograms.
        hidden = spectrograms.transpose(1, 2)
        for blk in self.blocks:
            hidden = blk(hidden)
        hidden = hidden.transpose(1, 2)
        result = spectrograms + hidden

        if npfl138.first_time("Postnet.forward"):
            print(f"The torch.std of the first batch returned by Postnet: {torch.std(result):.4f}")

        return result


class Tacotron(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, num_characters: int) -> None:
        super().__init__()
        # TODO: Create the Tacotron 2 model consisting of the encoder, attention, decoder, and post-net modules.
        self.encoder = Encoder(args, num_characters)
        self.attention = Attention(args)
        self.decoder = Decoder(args)
        self.postnet = Postnet(args)
        self.args = args

    def forward(self, texts: torch.Tensor, spectrograms_len: torch.Tensor) -> torch.Tensor:
        # TODO: Start by encoding the texts using the encoder.
        encoded_texts = self.encoder(texts)

        # TODO: Then, reset the attention and decoder modules using the `reset` method with
        # appropriate arguments.
        self.attention.reset(texts, encoded_texts)
        self.decoder.reset(texts)

        # Now, compute the sequence of mel spectrogram frames and the gate outputs.
        mel_frames, gates = [], []
        for _ in range(spectrograms_len):
            # TODO: Run the `self.attention` module on the current decoder input (which
            # is stored somewhere in the `self.decoder` instance) to obtain the context vector.
            context = self.attention(self.decoder.next_in)

            # TODO: Then run the `self.decoder` module on the obtained context vector.
            mel_frame, gate = self.decoder(context)

            # TODO: Append the obtained mel frame and gate output to the `mel_frames` and `gates` lists.
            mel_frames.append(mel_frame)
            gates.append(gate)

        # TODO: Stack the `mel_frames` and `gates` lists into tensors; the first two dimensions of
        # the resulting tensors should be `[batch_size, max_spectrogram_len]`.
        mel_frames = torch.stack(mel_frames, dim=1)
        gates = torch.stack(gates, dim=1)

        # TODO: Finally, pass the `mel_frames` through the post-net.
        mel_frames = self.postnet(mel_frames)

        return mel_frames, gates

    def compute_loss(self, y_pred: tuple[torch.Tensor, torch.Tensor], y_true: tuple[torch.Tensor, torch.Tensor],
                     texts: torch.Tensor, spectograms_len: torch.Tensor) -> torch.Tensor:
        # Unpack the predicted and true values.
        mel_frames, gates = y_pred
        spectrograms, spectrogram_lens = y_true

        # TODO: We need to ignore padding values during loss computation; therefore, use
        # `torch.masked_select` to select only the non-padding values from predicted and true values.
        max_frames = mel_frames.size(1)
        frame_idx = torch.arange(max_frames, device=mel_frames.device)[None, :]
        valid_mask = (frame_idx < spectrogram_lens[:, None]).unsqueeze(-1)

        mel_pred_vals = torch.masked_select(mel_frames, valid_mask)
        mel_true_vals = torch.masked_select(spectrograms[:, :max_frames], valid_mask)
        mse_loss = torch.nn.functional.mse_loss(mel_pred_vals, mel_true_vals)


        # TODO: The loss is a sum of the following two terms:
        # - the mean squared error between the predicted `mel_frames` and true `spectrograms`,
        # - the binary cross-entropy between the predicted `gates` and true values derived
        #   from `spectrogram_lens`. As an example, if a spectrogram has length 3, the gates should
        #   be 0 for the first frame and second frame, and 1 for the third frame.
        gate_targets = torch.zeros_like(gates)
        for i, L in enumerate(spectrogram_lens):
            if 0 < L <= gate_targets.size(1):
                gate_targets[i, L-1] = 1.0

        gate_pred_vals = torch.masked_select(gates, valid_mask)
        gate_true_vals = torch.masked_select(gate_targets, valid_mask)
        bce_loss = torch.nn.functional.binary_cross_entropy(gate_pred_vals, gate_true_vals)

        if npfl138.first_time("Tacotron.compute_loss"):
            print(f"The first batch loss values: (mse={mse_loss:.4f}, bce={bce_loss:.4f})")

        return mse_loss + bce_loss


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example: TTSDataset.Element) -> tuple[torch.Tensor, torch.Tensor]:
        # The input `example` is a dictionary with keys "text" and "mel_spectrogram".

        # TODO: Prepare a single example for training, returning a pair consisting of:
        # - the text converted to a sequence of character indices according to `self.dataset.char_vocab`,
        # - the unmodified mel spectrogram.
        # raise NotImplementedError()
        text_ids = torch.tensor(
            [self.dataset.char_vocab.index(ch) for ch in example["text"]],
            dtype=torch.long
        )
        mel_spec = torch.tensor(example["mel_spectrogram"], dtype=torch.float32)
        return text_ids, mel_spec

    def collate(self, batch: list) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        text_ids, spectrograms = zip(*batch)
        # TODO: Construct a single batch from a list of individual examples.
        # - The `text_ids` should be padded to the same length using `torch.nn.utils.rnn.pad_sequence`,
        #   using `TTSDataset.PAD` (which is guaranteed to be 0) as the padding value.
        # - The lengths of the unpadded spectrograms should be stored in a tensor `spectrogram_lens`.
        # - Finally, the `spectrograms` should also be padded to a common minimal length.
        padded_text_ids = torch.nn.utils.rnn.pad_sequence(
            text_ids,
            batch_first=True,
            padding_value=TTSDataset.PAD
        )
        
        spectrogram_lens = torch.tensor(
            [m.size(0) for m in spectrograms],
            dtype=torch.long
        )        
        
    
        max_len = int(spectrogram_lens.max())
        n_mels = spectrograms[0].size(1)
        padded_spectrograms = torch.zeros(len(spectrograms), max_len, n_mels)
        for i, m in enumerate(spectrograms):
            padded_spectrograms[i, : m.size(0)] = m

        # As input, apart from text ids, we return the maximum spectrogram length to indicate
        # how many mel frames to produce during training. During inference, this value will be
        # set to -1 to indicate that the model should produce mel frames until it decides to stop.
        return (padded_text_ids, spectrogram_lens.max()), (padded_spectrograms, spectrogram_lens)


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed, the number of threads, and optionally the ReCodEx mode.
    npfl138.startup(args.seed, args.threads, recodex=args.recodex)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the TTS data.
    tts_dataset = TTSDataset(args.dataset, args.sample_rate, args.window_length, args.hop_length, args.mels)
    train = TrainableDataset(tts_dataset.train).dataloader(args.batch_size, shuffle=True, seed=args.seed)

    # Create the model.
    tacotron = Tacotron(args, len(tts_dataset.train.char_vocab))

    # Train the model.
    tacotron.configure(
        optimizer=torch.optim.Adam(tacotron.parameters()),
        logdir=args.logdir,
    )
    logs = tacotron.fit(train, epochs=args.epochs)

    # Return the logs for ReCodEx to validate.
    return logs


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
