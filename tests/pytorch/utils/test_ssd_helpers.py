import os

import pytest
import torch

from sparseml.pytorch.utils import get_default_boxes_300


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_default_box_representations():
    default_boxes = get_default_boxes_300()
    xywh_boxes = default_boxes.as_xywh()
    ltrb_boxes = default_boxes.as_ltrb()

    x = 0.5 * (ltrb_boxes[:, 2] + ltrb_boxes[:, 0])
    y = 0.5 * (ltrb_boxes[:, 3] + ltrb_boxes[:, 1])
    w = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
    h = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]

    assert torch.max(torch.abs(xywh_boxes[:, 0] - x)) < 1e-4
    assert torch.max(torch.abs(xywh_boxes[:, 1] - y)) < 1e-4
    assert torch.max(torch.abs(xywh_boxes[:, 2] - w)) < 1e-4
    assert torch.max(torch.abs(xywh_boxes[:, 3] - h)) < 1e-4


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_default_box_encode_decode():
    default_boxes = get_default_boxes_300()
    boxes = torch.FloatTensor(1, 4).uniform_(0.0, 1.0).sort()[0]  # random ltrb box
    labels = torch.Tensor([1])
    enc_boxes, enc_labels = default_boxes.encode_image_box_labels(boxes, labels)

    # create scores to simulate model output
    scores = torch.zeros(1, 2, enc_boxes.size(1))
    scores[:, 0, :] = 100
    scores[:, 0, enc_labels == 1] = 0
    scores[:, 1, enc_labels == 1] = 100
    dec_boxes, dec_labels, _ = default_boxes.decode_output_batch(
        enc_boxes.unsqueeze(0), scores
    )[0]
    assert dec_labels.size(0) == 1
    assert dec_labels.item() == 1
    assert torch.max(torch.abs(boxes - dec_boxes)) < 1e-6
