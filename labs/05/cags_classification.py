#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchmetrics
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=11, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset, fn):
        super().__init__(dataset)
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self._fn = fn

    def transform(self, example):
        image = self._fn(example["image"])
        label = example["label"]
        return image.to(self.device), label.to(self.device)

class Model(npfl138.TrainableModule):
    def __init__(self, backbone, num_classes):
        super(Model, self).__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Linear(backbone.num_features, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, inputs, labels=None):
        features = self.backbone(inputs)
        logits = self.classifier(features)

        if self.training or labels is not None:
            return logits
        else:
            return torch.argmax(logits, dim=1)

    def train(self, training=True):
      
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False 

        self.classifier.train(training)
        return self
    
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

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1280]`.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    model = Model(efficientnetv2_b0, CAGS.LABELS)

    train = torch.utils.data.DataLoader(Dataset(cags.train, preprocessing), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(cags.dev, preprocessing), batch_size=args.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(Dataset(cags.test, preprocessing), batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.0005)
    model.configure(
        optimizer=optimizer,
        loss=model.loss_fn,
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS).to(model.device)},
        logdir=args.logdir
    )
    model.fit(train, dev=dev, epochs=args.epochs)
    
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
