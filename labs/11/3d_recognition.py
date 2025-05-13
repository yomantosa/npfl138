#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch

import npfl138
npfl138.require_version("2425.11")
from npfl138.datasets.modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class Model3D(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=128 * 2 * 2 * 2, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=512, out_features=ModelNet.LABELS),
        )

    def forward(self, input_volume: torch.Tensor) -> torch.Tensor:
        output_logits = self.feature_extractor(input_volume)
        return output_logits


class ModelNetDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, resolution=20, training=True):
        self.training = training
        formatted_data = []

        for sample_index in range(len(dataset)):
            original_voxel = dataset.data['grids'][sample_index]
            class_label = dataset.data['labels'][sample_index]

            resized_voxel = np.resize(original_voxel, (resolution, resolution, resolution))
            voxel_tensor = torch.tensor(resized_voxel, dtype=torch.float32).unsqueeze(0)

            formatted_data.append((voxel_tensor, class_label))

        self._dataset = formatted_data


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

    # Load the data.
    modelnet = ModelNet(args.modelnet)

    train = torch.utils.data.DataLoader(ModelNetDataset(modelnet.train, resolution=20, training=True), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(ModelNetDataset(modelnet.dev, resolution=20, training=True), batch_size=args.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(ModelNetDataset(modelnet.test, resolution=20, training=False), batch_size=args.batch_size, shuffle=False)

    
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # TODO: Create the model and train it
    model = Model3D(args).to(device)
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss=torch.nn.CrossEntropyLoss(),
        logdir=args.logdir,
    )
    
    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(grid, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
