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
        # TODO(tacotron): Create the required encoder layers. The architecture of the encoder is as follows:
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
        self.dropout = torch.nn.Dropout(args.dropout)
        self.conv_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(args.encoder_dim, args.encoder_dim, kernel_size=5, padding=2, bias=False),
                torch.nn.BatchNorm1d(args.encoder_dim),
                torch.nn.ReLU()
            )
            for _ in range(args.encoder_layers)
        ])

        self.bi_lstm = torch.nn.LSTM(
            args.encoder_dim, args.encoder_dim,
            num_layers=1, bidirectional=True, batch_first=True
        )

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        # TODO(tacotron): Implement the forward pass of the encoder.
        
        emb = self.embedding(texts)
        emb = self.dropout(emb)

        conv_in = emb.transpose(1, 2)
        for block in self.conv_blocks:
            conv_in = block(conv_in)
        conv_out = conv_in.transpose(1, 2)
        conv_out = self.dropout(conv_out)

        lengths = (texts != TTSDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            conv_out, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bi_lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        h_dim = self.bi_lstm.hidden_size
        summed = lstm_out[:, :, :h_dim] + lstm_out[:, :, h_dim:]
        result = self.dropout(summed)

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
        # TODO(tacotron): The `reset` method initializes the attention module for a new batch of texts.
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
        
        self.encoded_text = encoded_text
        self.processed_memory = self.attention_memory_layer(encoded_text)
        self.attention_mask = torch.where(
            text == TTSDataset.PAD,
            torch.tensor(-1e9, device=text.device),
            torch.tensor(0.0, device=text.device)
        )

        batch_size, seq_len = text.size()
        device = text.device
        hidden_size = self.attention_rnn.hidden_size

        self.attention_h = torch.zeros(batch_size, hidden_size, device=device)
        self.attention_c = torch.zeros(batch_size, hidden_size, device=device)

        self.prev_attention_weights = torch.zeros(batch_size, seq_len, device=device)
        self.cumulative_attention_weights = torch.zeros(batch_size, seq_len, device=device)

        _, _, encoder_dim = encoded_text.size()
        self.prev_context = torch.zeros(batch_size, encoder_dim, device=device)

        self.all_attention_logits: list[torch.Tensor] = []

    def forward(self, prenet: torch.Tensor) -> torch.Tensor:
        # TODO(tacotron): Implement a single step of the attention mechanism, relying on the previously-computed
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
        
        rnn_input = torch.cat([prenet, self.prev_context], dim=-1)
        self.attention_h, self.attention_c = self.attention_rnn(
            rnn_input, (self.attention_h, self.attention_c)
        )

        loc_in = torch.stack(
            [self.cumulative_attention_weights, self.prev_attention_weights],
            dim=1
        )
        loc_feats = self.location_sensitive_conv(loc_in)
        loc_feats = loc_feats.transpose(1, 2)
        loc_feats = self.location_sensitive_output(loc_feats)
        query = self.attention_query_layer(self.attention_h).unsqueeze(1)
        logits = self.attention_output_layer(
            torch.tanh(query + self.processed_memory + loc_feats)
        ).squeeze(-1)

        logits = logits + self.attention_mask

        attention_weights = torch.softmax(logits, dim=-1)

        self.prev_attention_weights = attention_weights
        self.cumulative_attention_weights = (
            self.cumulative_attention_weights + attention_weights
        )
        self.all_attention_logits.append(logits)
        context = torch.sum(
            attention_weights.unsqueeze(-1) * self.encoded_text, dim=1
        )
        self.prev_context = context

        if npfl138.first_time("Attention.forward"):
            print(f"The torch.std of the first batch returned by Attention: {torch.std(context):.4f}")

        return context


class Decoder(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO(tacotron): Create the layers of the decoder. Start by defining the pre-net module, which is
        # composed of `args.prenet_layers` layers, each consisting of:
        # - a linear layer with `args.prenet_dim` output dimension,
        # - ReLU activation,
        # - dropout with `args.dropout` rate.
        prenet_layers = []
        for i in range(args.prenet_layers):
            in_dim = args.mels if i == 0 else args.prenet_dim
            prenet_layers += [
                torch.nn.Linear(in_dim, args.prenet_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(args.dropout),
            ]
        self.prenet = torch.nn.Sequential(*prenet_layers)

        # The LSTM decoder cell is already prepared for you.
        self.decoder = torch.nn.LSTMCell(args.prenet_dim + args.encoder_dim, args.decoder_dim)

        # The `decoder_start` is a learnable parameter that is used as the initial input to the decoder.
        self.decoder_start = torch.nn.Parameter(torch.zeros(args.prenet_dim, dtype=torch.float32))

        # TODO(tacotron): Create the output layer with no activation that maps decoder states
        # to mel spectrograms with `args.mels` output channels.
        self.output_layer = torch.nn.Linear(
            args.decoder_dim,
            args.mels
        )

        # TODO(tacotron): Create the gate layer that maps the decoder states to a single value predicting
        # whether this step of the decoder should be the last one.
        self.gate_layer = torch.nn.Linear(
            args.decoder_dim,
            1
        )

    def reset(self, texts: torch.Tensor) -> None:
        # TODO(tacotron): Similarly to the `Attention.reset`, the `reset` method initializes the decoder
        # for a new batch of texts. You should
        # - store properly tiled (repeated) `self.decoder_start` as the next input to the decoder,
        # - zero-initialize the decoder state (`h`) and memory cell (`c`) of the `self.decoder`.
        # raise NotImplementedError()
        
        batch_size = texts.size(0)
        device = texts.device
        self.next_input = self.decoder_start.unsqueeze(0).expand(batch_size, -1).to(device)
        hidden_size = self.decoder.hidden_size
        self.decoder_h = torch.zeros(batch_size, hidden_size, device=device)
        self.decoder_c = torch.zeros(batch_size, hidden_size, device=device)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO(tacotron): Implement a single step of the decoder.
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

        rnn_input = torch.cat([self.next_input, context], dim=-1)
        self.decoder_h, self.decoder_c = self.decoder(
            rnn_input, (self.decoder_h, self.decoder_c)
        )

        mel_frame = self.output_layer(self.decoder_h)
        gate = torch.sigmoid(self.gate_layer(self.decoder_h))
        self.next_input = self.prenet(mel_frame)

        if npfl138.first_time("Decoder.forward"):
            print("The torch.std of the first batch returned by Decoder:",
                  f"({torch.std(mel_frame):.4f}, {torch.std(gate):.4f})")

        return mel_frame, gate


class Postnet(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO(tacotron): The post-net is composed of `args.postnet_layers` convolutional layers.
        # - The first `args.postnet_layers - 1` of them consist of
        #   - a 1D convolution with `args.postnet_dim` output channels, kernel size 5, padding 2, and no bias,
        #   - a batch normalization,
        #   - the tanh activation.
        # - The last layer consists of a 1D convolution with the same hyperparameters, but with `args.mels`
        #   output channels, followed by a batch normalization; no activation is applied.
        # raise NotImplementedError()
        self.conv_layers = torch.nn.ModuleList()

        if args.postnet_layers == 1:
            self.conv_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        args.mels, args.mels,
                        kernel_size=5, padding=2, bias=False
                    ),
                    torch.nn.BatchNorm1d(args.mels)
                )
            )
        else:
            for i in range(args.postnet_layers - 1):
                in_channels = args.mels if i == 0 else args.postnet_dim
                self.conv_layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_channels,
                            args.postnet_dim,
                            kernel_size=5, padding=2, bias=False
                        ),
                        torch.nn.BatchNorm1d(args.postnet_dim),
                        torch.nn.Tanh()
                    )
                )
            self.conv_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        args.postnet_dim,
                        args.mels,
                        kernel_size=5, padding=2, bias=False
                    ),
                    torch.nn.BatchNorm1d(args.mels)
                )
            )

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        # TODO(tacotron): Given a batch of mel spectrograms with shape `[batch_size, max_spectrogram_len, mels]`,
        # - move the channels to the front (i.e., dim=1),
        # - pass the spectrograms through the post-net,
        # - move the channels back to the last dimension.
        # Finally, return the sum of the original and processed spectrograms.
        x = spectrograms.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        result = spectrograms + x

        if npfl138.first_time("Postnet.forward"):
            print(f"The torch.std of the first batch returned by Postnet: {torch.std(result):.4f}")

        return result


class Tacotron(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, num_characters: int) -> None:
        super().__init__()
        # TODO(tacotron): Create the Tacotron 2 model consisting of the encoder, attention, decoder, and post-net modules.
        self.encoder = Encoder(args, num_characters)
        self.attention = Attention(args)
        self.decoder = Decoder(args)
        self.postnet = Postnet(args)
        self.args = args

    def forward(self, texts: torch.Tensor, spectrograms_len: torch.Tensor) -> torch.Tensor:
        # TODO(tacotron): Start by encoding the texts using the encoder.
        encoded_texts = self.encoder(texts)

        # TODO(tacotron): Then, reset the attention and decoder modules using the `reset` method with
        # appropriate arguments.
        self.attention.reset(texts, encoded_texts)
        self.decoder.reset(texts)

        # Now, compute the sequence of mel spectrogram frames and the gate outputs.
        mel_frames, gates = [], []
        for _ in range(spectrograms_len):
            # TODO(tacotron): Run the `self.attention` module on the current decoder input (which
            # is stored somewhere in the `self.decoder` instance) to obtain the context vector.
            context = self.attention(self.decoder.next_input)

            # TODO(tacotron): Then run the `self.decoder` module on the obtained context vector.
            mel_frame, gate = self.decoder(context)
            mel_frames.append(mel_frame)

            # TODO(tacotron): Append the obtained mel frame and gate output to the `mel_frames` and `gates` lists.
            gates.append(gate)
            
        # TODO(tacotron): Stack the `mel_frames` and `gates` lists into tensors; the first two dimensions of
        # the resulting tensors should be `[batch_size, max_spectrogram_len]`.
        mel_frames = torch.stack(mel_frames, dim=1)
        gates = torch.stack(gates, dim=1)
            
        # TODO(tacotron): Finally, pass the `mel_frames` through the post-net.
        mel_frames = self.postnet(mel_frames)

        return mel_frames, gates

    def compute_loss(self, y_pred: tuple[torch.Tensor, torch.Tensor], y_true: tuple[torch.Tensor, torch.Tensor],
                     texts: torch.Tensor, spectograms_len: torch.Tensor) -> torch.Tensor:
        # Unpack the predicted and true values.
        mel_frames, gates = y_pred
        spectrograms, spectrogram_lens = y_true

        # TODO(tacotron): We need to ignore padding values during loss computation; therefore, use
        # `torch.masked_select` to select only the non-padding values from predicted and true values.
        device = mel_frames.device
        max_t  = mel_frames.size(1)
        mask   = (torch.arange(max_t, device=device)[None, :] < spectrogram_lens[:, None])
        mask   = mask.float().unsqueeze(-1)
        
        # TODO(tacotron): The loss is a sum of the following two terms:
        # - the mean squared error between the predicted `mel_frames` and true `spectrograms`,
        # - the binary cross-entropy between the predicted `gates` and true values derived
        #   from `spectrogram_lens`. As an example, if a spectrogram has length 3, the gates should
        #   be 0 for the first frame and second frame, and 1 for the third frame.
        mse_loss = torch.nn.functional.mse_loss(
            mel_frames * mask,
            spectrograms[:, :max_t] * mask,
            reduction='sum'
        ) / (mask.sum() * mel_frames.size(2))

        # 3) BCE on gate (stop-token)  
        gate_targets = torch.zeros_like(gates)
        for i, L in enumerate(spectrogram_lens):
            if 0 < L <= gate_targets.size(1):
                gate_targets[i, L-1] = 1.0

        bce_loss = torch.nn.functional.binary_cross_entropy(
            gates * mask,
            gate_targets * mask,
            reduction='sum'
        ) / mask.sum()

        # TODO: Additionally, maximize the sum of probabilities of all monotonic alignments between
        # the mel spectrogram frames and the text characters. To this end:
        # - Obtain all (masked) attention logits computed in the `Attention.forward` method.
        #   You need to store them in the `self.attention` instance during `Attention.forward`
        #   (or return them from `Attention.forward` instead of storing them to `self.attention`)
        #   and collect all of them in `self.forward`. The resulting tensor should have shape
        #   `[max_spectrogram_length, batch_size, max_text_length]`.
        # - Then, you need to include the logit for the blank token required by the CTC loss:
        #   prepend a fixed logit of -1 as the first "row" of the attention logits, obtaining
        #   attention tensor with shape `[max_spectrogram_lengthm, batch_size, max_text_length + 1]`.
        # - Compute the log softmax of the attention logits along a suitable dimension.
        #   Together with `spectrogram_lens`, this will be the predicted input to the CTC loss.
        # - The CTC loss targets are sequences `[1, 2, ..., text_length]`, with the length
        #   being equal to the (non-padding) length of the input texts.
        # - Finally, compute the CTC loss using `torch.nn.functional.ctc_loss`, with the
        #   `zero_infinity=True` argument, and add it to the `loss`.
        
        attn_logits = torch.stack(self.attention.all_attention_logits, dim=0)
        blank = torch.full(
            (attn_logits.size(0), attn_logits.size(1), 1),
            -1.0, device=attn_logits.device
        )
        logits_with_blank = torch.cat([blank, attn_logits], dim=2)
        log_probs = torch.nn.functional.log_softmax(logits_with_blank, dim=2)

        targets, target_lens = [], []
        for txt in texts:
            L = (txt != TTSDataset.PAD).sum().item()
            targets.append(torch.arange(1, L+1, device=device))
            target_lens.append(L)
        targets     = torch.cat(targets)
        target_lens = torch.tensor(target_lens, device=device)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs, targets,
            input_lengths = spectrogram_lens,
            target_lengths= target_lens,
            blank=0, zero_infinity=True
        )

        if npfl138.first_time("Tacotron.compute_loss"):
            print(f"The first batch loss values: (mse={mse_loss:.4f}, bce={bce_loss:.4f}, ctc={ctc_loss:.4f})")

        return mse_loss + bce_loss + ctc_loss


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example: TTSDataset.Element) -> tuple[torch.Tensor, torch.Tensor]:
        # The input `example` is a dictionary with keys "text" and "mel_spectrogram".

        # TODO(tacotron): Prepare a single example for training, returning a pair consisting of:
        # - the text converted to a sequence of character indices according to `self.dataset.char_vocab`,
        # - the unmodified mel spectrogram.
        # raise NotImplementedError()
        text_indices = torch.tensor(
            [self.dataset.char_vocab.index(c) for c in example["text"]],
            dtype=torch.long
        )
        mel_spectrogram = torch.tensor(example["mel_spectrogram"], dtype=torch.float32)
        
        return text_indices, mel_spectrogram

    def collate(self, batch: list) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        text_ids, spectrograms = zip(*batch)
        # TODO(tacotron): Construct a single batch from a list of individual examples.
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
            [spec.size(0) for spec in spectrograms],
            dtype=torch.long
        )

        max_spec_len = int(spectrogram_lens.max().item())
        n_mels = spectrograms[0].size(1)
        padded_spectrograms = torch.zeros(
            len(spectrograms), max_spec_len, n_mels,
            dtype=torch.float32
        )
        for i, spec in enumerate(spectrograms):
            padded_spectrograms[i, : spec.size(0)] = spec

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
