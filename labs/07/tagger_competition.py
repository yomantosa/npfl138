#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class TaggerModel(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train_data: MorphoDataset.Dataset) -> None:
        super().__init__()
        self.word_embedding = torch.nn.Embedding(len(train_data.words.string_vocab), 128)
        self.analyzer_embedding = torch.nn.Embedding(len(train_data.tags.char_vocab), 16, padding_idx=MorphoDataset.PAD)
        self.rnn = torch.nn.LSTM(368, 128, bidirectional=True, batch_first=True)
        self.output_layer = torch.nn.Linear(256, len(train_data.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor, analyzer_ids: torch.Tensor) -> torch.Tensor:
        word_emb = self.word_embedding(word_ids)
        analyzer_emb = self.analyzer_embedding(analyzer_ids)
        analyzer_emb = analyzer_emb.flatten(2)
        
        combined = torch.cat([word_emb, analyzer_emb], dim=-1)

        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(combined, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=word_ids.shape[1])

        logits = self.output_layer(encoded)
        logits = logits.permute(0, 2, 1)
        return logits


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset, analyzer: MorphoAnalyzer) -> None:
        super().__init__(dataset)
        self.analyzer = analyzer

    def transform(self, example) -> tuple:
        word_ids = torch.tensor([self.dataset.words.string_vocab.index(word) for word in example["words"]], dtype=torch.long)
        tag_ids = torch.tensor([self.dataset.tags.string_vocab.index(tag) for tag in example["tags"]], dtype=torch.long)

        analyzer_ids = []
        for word in example["words"]:
            analyses = self.analyzer.get(word)
            if analyses:
                first_pair = analyses[0]
                tag_chars = [self.dataset.tags.char_vocab.index(c) for c in first_pair.tag]
            else:
                tag_chars = [MorphoDataset.PAD] * 15
            analyzer_ids.append(tag_chars)

        analyzer_ids = torch.tensor(analyzer_ids, dtype=torch.long)
        return (word_ids, analyzer_ids), tag_ids

    def collate(self, batch) -> tuple:
        batch_inputs, tag_ids = zip(*batch)
        word_ids, analyzer_ids = zip(*batch_inputs)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        analyzer_ids = torch.nn.utils.rnn.pad_sequence(analyzer_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        return (word_ids, analyzer_ids), tag_ids


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
    train = TrainableDataset(morpho.train, analyzer=analyses).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, analyzer=analyses).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test, analyzer=analyses).dataloader(batch_size=args.batch_size)

    model = TaggerModel(args, morpho.train)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=len(morpho.train.tags.string_vocab), ignore_index=MorphoDataset.PAD)},
        logdir=args.logdir,
    )
    
    model.fit(train, dev=dev, epochs=args.epochs)


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set. The following code assumes you use the same
        # output structure as in `tagger_we`, i.e., that for each sentence, the predictions are
        # a Numpy vector of shape `[num_tags, sentence_len_or_more]`, where `sentence_len_or_more`
        # is the length of the corresponding batch. (FYI, if you instead used the `packed` variant,
        # the prediction for each sentence is a vector of shape `[exactly_sentence_len, num_tags]`.)
        predictions = model.predict(test, data_with_labels=True)
        for predicted_tags, words in zip(predictions, morpho.test.words.strings):
            for predicted_tag in predicted_tags[:, :len(words)].argmax(axis=0):
                print(morpho.train.tags.string_vocab.string(predicted_tag), file=predictions_file)
            print(file=predictions_file)
    
if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
