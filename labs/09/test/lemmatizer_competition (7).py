#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import math

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.9")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--cle_dim", default=128, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN layer dimension.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--show_results_every_batch", default=10, type=int, help="Show results every given batch.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        self._project_encoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._project_decoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._output_layer = torch.nn.Linear(attention_dim, 1)

    def setup_memory(self, encoded):
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        projected_decoder_state = self._project_decoder_layer(states)
        sum_proj = self._encoded_projected + projected_decoder_state.unsqueeze(1)
        attention_weights = torch.nn.functional.softmax(self._output_layer(torch.tanh(sum_proj)), dim=1)
        attention = torch.sum(attention_weights * self._encoded, dim=1)
        combined_input = torch.cat([inputs, attention], dim=-1)
        return self._cell(combined_input, states)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset, analyses) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab
        self._tags_vocab = train.tags.char_vocab

        self._source_embedding = torch.nn.Embedding(len(self._source_vocab), args.cle_dim)
        self._source_rnn = torch.nn.GRU(args.cle_dim, args.rnn_dim, bidirectional=True, batch_first=True)
        self._target_rnn_cell = WithAttention(torch.nn.GRUCell(args.cle_dim + args.rnn_dim, args.rnn_dim), attention_dim=args.rnn_dim)
        self._target_output_layer = torch.nn.Linear(args.rnn_dim, len(self._target_vocab))
        self._dropout = torch.nn.Dropout(p=0.5)  

        
        if not args.tie_embeddings: 
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab), args.cle_dim)
        else:
            assert args.cle_dim == args.rnn_dim, "When tying embeddings, cle_dim and rnn_dim must match."
            self._target_embedding = lambda x: torch.nn.functional.embedding(x, self._target_output_layer.weight * math.sqrt(args.rnn_dim))

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, words: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        embedded = self._source_embedding(words).to(device)
        lengths = (words != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self._source_rnn(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        forward_outputs = outputs[:, :, :outputs.size(2)//2]
        backward_outputs = outputs[:, :, outputs.size(2)//2:]
        
        forward_outputs = self._dropout(forward_outputs)
        backward_outputs = self._dropout(backward_outputs)
        
        return forward_outputs + backward_outputs

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.full_like(targets[:, :1], MorphoDataset.BOW).to(device)
        inputs = torch.cat([inputs, targets[:, :-1]], dim=1)

        self._target_rnn_cell.setup_memory(encoded)

        embedded_inputs = self._target_embedding(inputs).to(device)
        states = encoded[:, 0].to(device)

        outputs = []
        for i in range(inputs.size(1)):
            states = self._target_rnn_cell(embedded_inputs[:, i], states)
            outputs.append(states)

        outputs = torch.stack(outputs, dim=1)
        logits = self._target_output_layer(outputs)
        return logits.permute(0, 2, 1)

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        self._target_rnn_cell.setup_memory(encoded)

        index = 0
        inputs = torch.full((batch_size,), MorphoDataset.BOW, dtype=torch.long, device=encoded.device)
        states = encoded[:, 0].to(device)
        results = []
        result_lengths = torch.full((batch_size,), max_length, dtype=torch.long, device=encoded.device)

        while index < max_length and torch.any(result_lengths == max_length):
            embedded_inputs = self._target_embedding(inputs).to(device)
            states = self._target_rnn_cell(embedded_inputs, states)
            logits = self._target_output_layer(states)
            predictions = torch.argmax(logits, dim=1)

            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

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
        if self._batches % self._show_results_every_batch == 0:
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
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool, analyses) -> None:
        super().__init__(dataset)
        self._training = training
        self.analyses = analyses
        
    def get_analyses_features(self, word):
        analysis = self.analyses.get(word)
        if len(analysis) > 0:
            return analysis[0].lemma
        return ''

    def transform(self, example):
        # TODO: Return `example["words"]` as inputs and `example["lemmas"]` as targets.
        # raise NotImplementedError()
        words = example["words"]
        lemmas = example["lemmas"]
        combined_words_lemmas = [word + self.get_analyses_features(word) for word in words]
        
        return combined_words_lemmas, lemmas


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

        
        
def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Create the model and train it.
    train = TrainableDataset(morpho.train, training=True, analyses=analyses).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, training=False, analyses=analyses).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test, training=False, analyses=analyses).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train, analyses).to(device)

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

    model = model.to(device) 
    model.fit(train, dev=dev, epochs=args.epochs)


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following prediction
        # command if you use a different output structure than in lemmatizer_noattn.
        predictions = iter(model.predict(test, data_with_labels=True))

        for sentence in morpho.test.words.strings:
            for word in sentence:
                lemma = next(predictions)
                print("".join(morpho.test.lemmas.char_vocab.strings(lemma)), file=predictions_file)
            print(file=predictions_file)

        print("python3 -m npfl138.datasets.morpho_dataset --task=lemmatizer --evaluate="+os.path.join(args.logdir, "lemmatizer_competition.txt")+" --dataset=dev")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)