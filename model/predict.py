import io

import numpy as np
import torch
from PIL import Image

from model.config import TEST_DATA_TRANSFORMS, CLASS_NAMES, WEIGHTS_PATH
from model.utils import get_model

class_names = list(CLASS_NAMES.values())


def get_prediction_by(file) -> (str, int):
    """
    Get prediction class and its probability obtained by our model
    """
    img_pil = Image.open(io.BytesIO(file.read()))
    img_array = np.array(img_pil)
    img = TEST_DATA_TRANSFORMS(img_array)[None, :, :, :]

    model = get_model(weights_path=WEIGHTS_PATH)
    model.eval()
    outputs = model(img)

    _, preds = torch.max(outputs, 1)
    outputs_softmax = torch.nn.Softmax(-1).forward(outputs)
    pred_class_index = preds.cpu().numpy()[0]
    pred_class = class_names[pred_class_index]
    pred_value = outputs_softmax.detach().numpy()[0, pred_class_index]

    return pred_class, pred_value
