#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchmetrics

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--max_samples", default=None, type=int, help="Limit training samples for quick testing.")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=12, type=int)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()
        # TODO: Define the model.
        self.blank_idx = CommonVoiceCs.PAD

        self.rnn = nn.GRU(
            input_size=CommonVoiceCs.MFCC_DIM,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=True,
            batch_first=False
        )
        self.fc = nn.Linear(args.hidden_dim * 2, len(CommonVoiceCs.LETTER_NAMES))
        self.metrics = {"edit_distance": CommonVoiceCs.EditDistanceMetric()}

    def forward(self, audio_input, input_lengths) -> torch.Tensor:
        audio_input = audio_input.permute(2, 0, 1)
        packed = rnn_utils.pack_padded_sequence(audio_input, input_lengths.cpu(), enforce_sorted=False)
        outputs, _ = self.rnn(packed)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs)
        logits = self.fc(outputs)
        return F.log_softmax(logits, dim=2)

    def compute_loss(self, log_probs, y_batch, input_padded, input_lengths) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        target_padded, target_lengths = y_batch
        targets = torch.cat([target_padded[i, :target_lengths[i]] for i in range(target_lengths.size(0))])
        loss_fn = nn.CTCLoss(blank=self.blank_idx, zero_infinity=True)
        if log_probs.device.type == 'mps':
            loss = loss_fn(log_probs.cpu(), targets, input_lengths.cpu(), target_lengths.cpu())
            return loss.to(log_probs.device)
        return loss_fn(log_probs, targets, input_lengths, target_lengths)

    def ctc_decoding(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        #   - note that you need to provide `blank_token` and `sil_token` arguments
        #     and they must be valid tokens. For `blank_token`, you need to specify
        #     the token whose index corresponds to the blank token index;
        #     for `sil_token`, you can use also the blank token index (by default,
        #     `sil_token` has ho effect on the decoding apart from being added as the
        #     first and the last token of the predictions unless it is a blank token).
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.
        predictions = []
        argmax = log_probs.argmax(dim=2)
        T, B = argmax.shape
        for b in range(B):
            prediction_sequence = argmax[:, b].tolist()
            collapsed_sequence = []
            prev_token = None
            for idx in prediction_sequence:
                if idx != self.blank_idx and idx != prev_token:
                    collapsed_sequence.append(idx)
                prev_token = idx
            predictions.append(torch.tensor(collapsed_sequence, device=log_probs.device))
        return predictions

    def compute_metrics(self, log_probs, y_batch, input_padded, input_lengths) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
        target_padded, target_lengths = y_batch
        if not self.training:
            predictions = self.ctc_decoding(log_probs, input_lengths)
            pred_str = ["".join(CommonVoiceCs.LETTER_NAMES[i] for i in seq.tolist()) for seq in predictions]
            gold_str = [
                "".join(CommonVoiceCs.LETTER_NAMES[i] for i in target_padded[j, :target_lengths[j]].tolist())
                for j in range(len(predictions))
            ]
            self.metrics["edit_distance"].update(pred_str, gold_str)
        return {k: m.compute() for k, m in self.metrics.items()}

    def predict_step(self, batch, as_numpy=True):
        (input_padded, input_lengths), _ = batch
        with torch.no_grad():
            log_probs = self.forward(input_padded, input_lengths)
            predictions = self.ctc_decoding(log_probs, input_lengths)
        if as_numpy:
            return [p.cpu().numpy() for p in predictions]
        return predictions


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # Note that while the `CommonVoiceCs.LETTER_NAMES` do not explicitly contain a blank token,
        # the [PAD] token can be employed as a blank token.
        mfcc = example["mfccs"]
        transcript = [CommonVoiceCs.LETTER_NAMES.index(c) for c in example["sentence"]]
        return mfcc, torch.tensor(transcript, dtype=torch.long)

    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        input_list, target_list = zip(*batch)
        input_lengths = torch.tensor([x.shape[0] for x in input_list], dtype=torch.long)
        input_padded = rnn_utils.pad_sequence(input_list, batch_first=True).permute(0, 2, 1)
        target_padded = rnn_utils.pad_sequence(target_list, batch_first=True, padding_value=CommonVoiceCs.PAD)
        target_lengths = torch.tensor([y.size(0) for y in target_list], dtype=torch.long)
        return (input_padded, input_lengths), (target_padded, target_lengths)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                for k, v in sorted(vars(args).items())
            )
        )
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the data.
    common_voice = CommonVoiceCs()

    train_dataset = common_voice.train
    if args.max_samples is not None:
        import torch.utils.data as data_utils
        train_dataset = data_utils.Subset(train_dataset, list(range(min(args.max_samples, len(train_dataset)))))

    train_loader = TrainableDataset(train_dataset).dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = TrainableDataset(common_voice.dev).dataloader(batch_size=args.batch_size)
    test_loader = TrainableDataset(common_voice.test).dataloader(batch_size=args.batch_size)

    # TODO: Create the model and train it
    model = Model(args, common_voice.train).to(device)
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss=None,
        metrics=model.metrics,
        logdir=args.logdir
    )

    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    predictions = []
    for batch in test_loader:
        (input_padded, input_lengths), (target_padded, target_lengths) = batch
        input_padded = input_padded.to(device)
        input_lengths = input_lengths.to(device)
        batch_device = ((input_padded, input_lengths), (target_padded, target_lengths))
        batch_predictions = model.predict_step(batch_device, as_numpy=True)
        predictions.extend(batch_predictions)

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as f:
        # TODO: Predict the CommonVoice sentences.
        for sequence in predictions:
            f.write("".join(CommonVoiceCs.LETTER_NAMES[i] for i in sequence.tolist()) + "\n")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
