# TODO: Incorporate this script into testing pipeline

import numpy as np
import torch

from loss import BinaryCrossEntropyLossWrapper, CrossEntropyLossWrapper, label_smoothing

torch.manual_seed(420)

# Test BinaryCrossEntropyLossWrapper with standard arguments.
input = torch.FloatTensor([[-1.6977, 0.6374],
                           [0.0781, -0.4140],
                           [1.5172, 0.0473]])
target = torch.FloatTensor([[0.8435, -0.2261],
                            [0.0345, -0.3422],
                            [0.6827, -0.8155]])

BCE = BinaryCrossEntropyLossWrapper(loss_params={'alpha': 0.0})
expected_solution = 0.8896  # calculated manually
solution = BCE(data=target, pred=input)['__loss__'].item()
assert np.isclose(expected_solution, solution, rtol=1e-04)

# Test label_smoothing for 2 classes
alpha = 0.5
target = torch.LongTensor([[1, 0],
                           [0, 1],
                           [1, 0]])
expected_target = torch.FloatTensor([[0.75, 0.25], [0.25, 0.75], [0.75, 0.25]])  # calculated manually
assert torch.equal(label_smoothing(target, alpha), expected_target)

# Test CrossEntropyLossWrapper where target expresses class indices and no smoothing (Scenario 1)
input = torch.FloatTensor(([[-1.6977, 0.6374, 0.0781, -0.4140],
                            [1.5172, 0.0473, 0.8435, -0.2261],
                            [0.0345, -0.3422, 0.6827, -0.8155]]))
target = torch.LongTensor([1, 2, 0])
CE = CrossEntropyLossWrapper(loss_params={'alpha': 0.0})
solution = CE(data=target, pred=input)['__loss__'].item()
expected_solution = 1.1393  # calculated manually
assert np.isclose(expected_solution, solution, rtol=1e-04)

# Test CrossEntropyLossWrapper where target expresses class indices with smoothing (Scenario 2)
CE = CrossEntropyLossWrapper(loss_params={'alpha': 0.5})
solution = CE(data=target, pred=input)['__loss__'].item()
expected_solution = 1.3775  # calculated using pytorch==1.10.0 function
assert np.isclose(expected_solution, solution, rtol=1e-04)

# Test CrossEntropyLossWrapper where target expresses class probability (Scenario 3) w/o smoothing
input = torch.FloatTensor(([[-1.6977, 0.6374, 0.0781, -0.4140],
                            [1.5172, 0.0473, 0.8435, -0.2261],
                            [0.0345, -0.3422, 0.6827, -0.8155]]))
target = torch.FloatTensor([[0.7058, 0.2871, 0.2633, 0.4042],
                            [0.2391, 0.5550, 0.9059, 0.5682],
                            [0.8020, 0.0656, 0.1067, 0.4335]])

CE = CrossEntropyLossWrapper(loss_params={'alpha': 0.0})
solution = CE(data=target, pred=input)['__loss__'].item()
expected_solution = 3.1869  # calculated using pytorch==1.10.0 function
assert np.isclose(expected_solution, solution, rtol=1e-04)

# Test CrossEntropyLossWrapper where target expresses class probability (Scenario 3) w smoothing
CE = CrossEntropyLossWrapper(loss_params={'alpha': 0.5})
solution = CE(data=target, pred=input)['__loss__'].item()
expected_solution = 2.4013
assert np.isclose(expected_solution, solution, rtol=1e-04)
