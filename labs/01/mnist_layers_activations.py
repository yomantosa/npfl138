#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
from npfl138 import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--activation", default="none", choices=["none", "relu", "tanh", "sigmoid"], help="Activation.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--hidden_layers", default=1, type=int, help="Number of layers.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


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

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model.
    model = torch.nn.Sequential()
    
    # TODO: Finish the model. Namely:
    # - start by adding the `torch.nn.Flatten()` layer;
    # - then add `args.hidden_layers` number of fully connected hidden layers
    #   `torch.nn.Linear()`, each with `args.hidden_layer_size` neurons and followed by
    #   a specified `args.activation`, allowing "none", "relu", "tanh", "sigmoid";
    # - finally, add an output fully connected layer with `MNIST.LABELS` units.
    
    model_layer = [torch.nn.Flatten()]
    input_size = MNIST.H * MNIST.W
    
    for i in range(args.hidden_layers):
        model_layer.append(torch.nn.Linear(input_size, args.hidden_layer_size))
        
        if args.activation == "none":
            model_layer.append(torch.nn.Identity())
        elif args.activation == "relu":
            model_layer.append(torch.nn.ReLU())
        elif args.activation == "tanh":
            model_layer.append(torch.nn.Tanh())
        elif args.activation == "sigmoid":
            model_layer.append(torch.nn.Sigmoid())
        
        input_size = args.hidden_layer_size
            
    model_layer.append(torch.nn.Linear(input_size, MNIST.LABELS))
    
    model = torch.nn.Sequential(*model_layer)
    # Create the TrainableModule and configure it for training.
    model = npfl138.TrainableModule(model)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )

    # Train the model.
    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
