#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
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

class ResidualBlock(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)
    
class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer **without bias** and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default padding of 0 (the "valid" padding).
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # To implement the residual connections, you can use various approaches, for example:
        # - you can create a specialized `torch.nn.Module` subclass representing a residual
        #   connection that gets the inside layers as an argument, and implement its forward call.
        #   This allows you to have the whole network in a single `torch.nn.Sequential`.
        # - you could represent the model module as a `torch.nn.ModuleList` of `torch.nn.Sequential`s,
        #   each representing one user-specified layer, keep track of the positions of residual
        #   connections, and manually perform them in the forward pass.
        #
        # It might be difficult to compute the number of features after the `F` layer. You can
        # nevertheless use the `torch.nn.LazyLinear`, `torch.nn.LazyConv2d`, and `torch.nn.LazyBatch2d`
        # layers, which do not require the number of input features to be specified in the constructor.
        # During `__init__`, these layers do not allocate their parameters, and only do so when
        # they are first called on a tensor, at which point the number of input features is known.
        # During this first call they also change themselves to the corresponding `torch.nn.Linear` etc.
        # This first call might be the first training batch, or you can call the model on a dummy input
        #   self.eval()(torch.zeros(1, MNIST.C, MNIST.H, MNIST.W))
        # where the `self.eval()` is necessary to avoid the BatchNorms to update their running statistics.

        # TODO: Finally, add the final Linear output layer with `MNIST.LABELS` units.
        
        super().__init__() 
        
        def parse_padding(p_str):
            if p_str == "valid":
                return 0
            elif p_str == "same":
                return "same"
            else:
                return int(p_str)

        tokens = []
        current = ""
        depth = 0
        for c in args.cnn:
            if c == "[":
                depth += 1
                current += c
            elif c == "]":
                depth -= 1
                current += c
            elif c == "," and depth == 0:
                tokens.append(current)
                current = ""
            else:
                current += c
        if current:
            tokens.append(current)

        layers = []

        for token in tokens:
            if token.startswith("C-"):
                _, f, k, s, p = token.split("-")
                layers.append(torch.nn.Sequential(
                    torch.nn.LazyConv2d(int(f), int(k), int(s), parse_padding(p)),
                    torch.nn.ReLU()
                ))

            elif token.startswith("CB-"):
                _, f, k, s, p = token.split("-")
                layers.append(torch.nn.Sequential(
                    torch.nn.LazyConv2d(int(f), int(k), int(s), parse_padding(p), bias=False),
                    torch.nn.BatchNorm2d(int(f)),
                    torch.nn.ReLU()
                ))

            elif token.startswith("M-"):
                _, k, s = token.split("-")
                layers.append(torch.nn.MaxPool2d(kernel_size=int(k), stride=int(s)))

            elif token.startswith("R-[") and token.endswith("]"):
                inner_spec = token[3:-1]
                inner_tokens = []
                inner = ""
                d = 0
                for c in inner_spec:
                    if c == "[":
                        d += 1
                        inner += c
                    elif c == "]":
                        d -= 1
                        inner += c
                    elif c == "," and d == 0:
                        inner_tokens.append(inner)
                        inner = ""
                    else:
                        inner += c
                if inner:
                    inner_tokens.append(inner)

                inner_layers = []
                for t in inner_tokens:
                    if t.startswith("C-"):
                        _, f, k, s, p = t.split("-")
                        inner_layers.append(torch.nn.Sequential(
                            torch.nn.LazyConv2d(int(f), int(k), int(s), parse_padding(p)),
                            torch.nn.ReLU()
                        ))
                    elif t.startswith("CB-"):
                        _, f, k, s, p = t.split("-")
                        inner_layers.append(torch.nn.Sequential(
                            torch.nn.LazyConv2d(int(f), int(k), int(s), parse_padding(p), bias=False),
                            torch.nn.BatchNorm2d(int(f)),
                            torch.nn.ReLU()
                        ))
                layers.append(ResidualBlock(inner_layers))

            elif token == "F":
                layers.append(torch.nn.Flatten())

            elif token.startswith("H-"):
                _, size = token.split("-")
                layers.append(torch.nn.Sequential(
                    torch.nn.LazyLinear(int(size)),
                    torch.nn.ReLU()
                ))

            elif token.startswith("D-"):
                _, rate = token.split("-")
                layers.append(torch.nn.Dropout(float(rate)))

        layers.append(torch.nn.LazyLinear(MNIST.LABELS))
        self.model = torch.nn.Sequential(*layers)

        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, MNIST.C, MNIST.H, MNIST.W)
            self.model(dummy_input)

    def forward(self, x):
        return self.model(x)


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

    # Create the model and train it
    model = Model(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
