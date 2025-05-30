#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.9")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=10, type=int, help="Show results every given batch.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        # TODO: Define
        # - `self._source_embedding` as an embedding layer of source characters into `args.cle_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units processing embedded source chars
        self._source_embedding = torch.nn.Embedding(len(self._source_vocab), args.cle_dim)
        self._source_rnn = torch.nn.GRU(
            input_size=args.cle_dim,
            hidden_size=args.rnn_dim,
            bidirectional=True,
            batch_first=True
        )

        # TODO: Then define
        # - `self._target_rnn` as a unidirectional GRU layer with `args.rnn_dim` units processing
        #   embedded target characters
        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_rnn = torch.nn.GRU(
            input_size=args.cle_dim,
            hidden_size=args.rnn_dim,
            batch_first=True
        )
        self._target_output_layer = torch.nn.Linear(args.rnn_dim, len(self._target_vocab))

        # Create self._target_rnn_cell, which is the single cell of `self._target_rnn`.
        self._target_rnn_cell = torch.nn.GRUCell(args.rnn_dim, args.rnn_dim)
        for name, _ in self._target_rnn_cell.named_parameters():
            setattr(self._target_rnn_cell, name, getattr(self._target_rnn, "{}_l0".format(name)))

        if not args.tie_embeddings:
            # TODO: Define the `self._target_embedding` as an embedding layer of the target
            # characters into `args.cle_dim` dimensions.
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab), args.cle_dim)

        else:
            assert args.cle_dim == args.rnn_dim, "When tying embeddings, cle_dim and rnn_dim must match."
            # TODO: Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `args.rnn_dim`.
            self._target_embedding = lambda x: torch.nn.functional.embedding(x, self._target_output_layer.weight * (args.rnn_dim ** 0.5))

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, words: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        # TODO: Embed the inputs using `self._source_embedding`.

        # TODO: Run the `self._source_rnn` on the embedded sequences, correctly handling
        # padding. The result should be the last hidden states of the forward and
        # backward direction, summed together.
        embedded = self._source_embedding(words)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, (words != MorphoDataset.PAD).sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self._source_rnn(packed_embedded)
        hidden_fw, hidden_bw = hidden[0], hidden[1]
        return hidden_fw + hidden_bw

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO: Generate inputs for the decoder, which are obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets`.

        # TODO: Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - the `self._target_rnn` layer,  additionally passing the encoder
        #   output (`encoded`) as the initial state to the GRU (the second argument);
        #   again correctly handle the padding of the inputs,
        # - the `self._target_output_layer` to obtain logits,
        # - finally, permute dimensions so that the logits are in the dimension 1,
        # and return the result.
        inputs = torch.full_like(targets[:, :1], MorphoDataset.BOW)
        inputs = torch.cat([inputs, targets[:, :-1]], dim=1)

        embedded_inputs = self._target_embedding(inputs)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_inputs, (targets != MorphoDataset.PAD).sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self._target_rnn(packed_embedded, encoded.unsqueeze(0))
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        logits = self._target_output_layer(outputs)
        logits = logits.permute(0, 2, 1)
        return logits

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: the time index, initialized to 0;
        # - `inputs`: a tensor of shape `[batch_size]` containing the `MorphoDataset.BOW` symbols,
        # - `states`: initial RNN state from the encoder, i.e., `encoded`.
        # - `results`: an empty list, where generated outputs will be stored;
        # - `result_lengths`: a tensor of shape `[batch_size]` filled with `max_length`,
        index = 0
        inputs = torch.full((batch_size,), MorphoDataset.BOW, dtype=torch.long, device=encoded.device)
        states = encoded
        results = []
        result_lengths = torch.full((batch_size,), max_length, dtype=torch.long, device=encoded.device)


        while index < max_length and torch.any(result_lengths == max_length):
            # TODO:
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn_cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            embedded_inputs = self._target_embedding(inputs)
            states = self._target_rnn_cell(embedded_inputs, states)
            logits = self._target_output_layer(states)
            predictions = logits.argmax(dim=-1)

            # Store the predictions in the `results` and update the `result_lengths`
            # by setting it to current `index` if an EOW was generated for the first time.
            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

            # TODO: Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if self.training:  # In training regime, convert logits to most likely predictions.
            y_pred = y_pred.argmax(dim=-2)
        # Compare the lemmas with the predictions using exact match accuracy.
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        self.metrics["accuracy"].update(torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1))
        return {name: metric.compute() for name, metric in self.metrics.items()}  # Return all metrics.

    def train_step(self, xs, y):
        result = super().train_step(xs, y)

        self._batches += 1
        if self._show_results_every_batch and self._batches % self._show_results_every_batch == 0:
            self.log_console("{}: {} -> {}".format(
                self._batches,
                "".join(self._source_vocab.strings(xs[0][0][xs[0][0] != MorphoDataset.PAD].numpy(force=True))),
                "".join(self._target_vocab.strings(self.predict_step((xs[0][:1],))[0]))))

        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.forward(*xs)
            # Trim the predictions at the first EOW
            batch = [lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0] for lemma in batch]
            return [lemma.numpy(force=True) for lemma in batch] if as_numpy else batch


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        # TODO: Return `example["words"]` as inputs and `example["lemmas"]` as targets.
        # raise NotImplementedError()
        return example["words"], example["lemmas"]


    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples generated by `transform`.
        words, lemmas = zip(*batch)
        words = [word for sentence in words for word in sentence]
        lemmas = [lemma for sentence in lemmas for lemma in sentence]
        # TODO: The `words` are a list of list of strings. Flatten it into a single list of strings
        # and then map the characters to their indices using the `self.dataset.words.char_vocab` vocabulary.
        # Then create a tensor by padding the words to the length of the longest one in the batch.
        word_indices = [torch.tensor(self.dataset.words.char_vocab.indices(word), dtype=torch.long) for word in words]
        words = torch.nn.utils.rnn.pad_sequence(word_indices, batch_first=True, padding_value=MorphoDataset.PAD)
        
        # TODO: Process `lemmas` analogously to `words`, but use `self.dataset.lemmas.char_vocab`,
        # and additionally, append `MorphoDataset.EOW` to the end of each lemma.
        lemma_indices = [torch.cat([torch.tensor(self.dataset.lemmas.char_vocab.indices(lemma), dtype=torch.long),
                            torch.tensor([MorphoDataset.EOW], dtype=torch.long)]) for lemma in lemmas]
        lemmas = torch.nn.utils.rnn.pad_sequence(lemma_indices, batch_first=True, padding_value=MorphoDataset.PAD)
        
        # TODO: Return a pair (inputs, targets), where
        # - the inputs are words during inference and (words, lemmas) pair during training;
        # - the targets are lemmas.
        if self._training:
            return (words, lemmas), lemmas
        else:
            return words, lemmas

def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data.
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, training=False).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO: Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO: Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation.
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),
        # TODO: Create a `torchmetrics.MeanMetric()` metric, where we will manually
        # collect lemmatization accuracy.
        metrics={"accuracy": torchmetrics.MeanMetric()},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return all metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items()}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
