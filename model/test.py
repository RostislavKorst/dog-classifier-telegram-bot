import argparse
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from model.config import CLASS_NAMES
from model.utils import get_model, get_data_loader, calculate_metrics


def test(weights_path: Optional[str]) -> None:
    """
    Testing classifier and visualizing predictions.

    :param weights_path: path to saved weights or None.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_loader = get_data_loader(is_train=False)
    class_names = test_loader.dataset.classes
    class_names = [CLASS_NAMES[class_names[i]] for i in range(len(class_names))]
    testing_samples_num = len(test_loader.dataset)

    model = get_model(weights_path)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()

    model.eval()
    running_loss, running_corrects = 0.0, 0

    all_labels, all_predictions = [], []
    for inputs, labels in tqdm(test_loader, desc='Testing. Batch'):
        all_labels.append(labels.numpy())
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_func(outputs, labels)
        all_predictions.append(preds.cpu().numpy())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / testing_samples_num
    epoch_acc = running_corrects.double() / testing_samples_num
    print('Testing: loss = {:.4f}, accuracy = {:.4f}.\n'.format(epoch_loss, epoch_acc))
    report = calculate_metrics(all_labels, all_predictions, class_names)
    print(report)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('Script for evaluating imagewoof classifier and visualizing predictions')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generator\'s seed. If seed < 0, seed will be None.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    if args.weights is None:
        print('WARNING: using randomly initialized model for evaluating.')

    test(args.weights)
