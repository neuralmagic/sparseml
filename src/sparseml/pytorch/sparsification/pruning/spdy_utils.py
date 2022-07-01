import os
import torch
import logging
import numpy as np
import torch.nn as nn


from torch import Tensor
from torch.utils.data import DataLoader


__all__ = [
    "WeightDatabase",
    "SPDY"
]

    
_LOGGER = logging.getLogger(__name__)


class WeightDatabase:

    def __init__(
        self,
        store_on_drive: bool = False,
        store_dir: str = './weight_database'
    ):
        # set initial values
        self.size = 0
        self.store_on_drive = store_on_drive
        self.store_dir = store_dir
        self.weight_traces = {}
        if store_on_drive:
            # make directory if doesn't exist
            os.makedirs(store_dir, exist_ok=True)

    def __setitem__(self, layer_name, weight_traces: Tensor):
        # offload to CPU for memory saving
        if self.store_on_drive:
            for i in range(weight_traces.shape[0]): 
                torch.save(weight_traces[i], f'{self.store_dir}/{layer_name}_weight_traces_{i}.pth')
            self.weight_traces[layer_name] = True
        # store in RAM
        else:
            self.weight_traces[layer_name] = weight_traces

    def get(self, layer_name: str, sp_idx: int):
        if self.store_on_drive:
            return torch.load(f'{self.store_dir}/{layer_name}_weight_traces_{sp_idx}.pth')
        else:
            return self.weight_traces[layer_name][sp_idx]

    def keys(self):
        return self.weight_traces.keys()

    def __len__(self):
        return len(self.weight_traces)

    
class SPDY:

    def __init__(
        self,
        model : nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        weight_database: dict,
        errs: dict,
        budgets: dict,
        target_budget_frac: float,
        num_buckets: int = 10000,
        num_rand_inits: int = 100,
        resample_perc: float = 0.1, 
        patience: int = 100,
        device: str = 'cpu',
        verbose: bool = False,
        save_profile: bool = False,
        save_profile_path: str = './spdy_profile.npy',
    ):
        self.model = model
        self.loader = loader
        self.loss_fn = loss_fn
        self.num_buckets = num_buckets
        self.target_budget_frac = target_budget_frac
        self.weight_database = weight_database
        self.num_rand_inits = num_rand_inits
        self.resample_perc = resample_perc
        self.patience = patience
        self.device = device
        self.verbose = verbose
        self.save_profile = save_profile
        self.save_profile_path = save_profile_path
        # stack dict of errs to 2d array
        self.errs = np.stack([v for _, v in errs.items()])
        # stack dict of budgets to 2d array
        self.budgets = np.stack([v for _, v in budgets.items()])
        # get buckets
        max_budget = self.budgets[:, 0].sum()
        target_bugdet = int(target_budget_frac * max_budget)
        bucket_size = target_bugdet / num_buckets
        # renormalize cost 
        self.budgets = (self.budgets / bucket_size).astype(int)
    
    def get_costs(self, coefs: np.ndarray) -> np.ndarray:
        return np.stack([
            [self.errs[i][j] * coefs[i] for j in range(self.errs.shape[1])] \
            for i in range(self.errs.shape[0])
        ])


    @torch.no_grad()
    def set_weights(self, solution: np.ndarray):
        for layer_id, layer_name in enumerate(self.weight_database.keys()):
            layer = self.model.get_submodule(layer_name)
            layer.weight.data = self.weight_database.get(layer_name, solution[layer_id])


    @torch.no_grad()
    def get_loss(self):
        total_loss = 0
        num_collected = 0
        for inputs, targets in self.loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            total_loss += len(inputs) * self.loss_fn(self.model(inputs), targets).item()
            num_collected += len(inputs)
        return total_loss / num_collected


    def evaluate(self, coefs: np.ndarray):
        # generate costs
        costs = self.get_costs(coefs)
        # find solution to DP problem
        solution = self.solve(costs)
        # construct model
        self.set_weights(solution)
        return solution, self.get_loss()
        

    def solve(self, costs: np.ndarray) -> list:
        sp_levels = costs.shape[1]
        Ds = np.full((len(costs), self.num_buckets + 1), float('inf'))
        Ps = np.full((len(costs), self.num_buckets + 1), -1)

        for i in range(sp_levels):
            if costs[0][i] < Ds[0][self.budgets[0][i]]:
                Ds[0][self.budgets[0][i]] = costs[0][i]
                Ps[0][self.budgets[0][i]] = i

        for module_id in range(1, len(Ds)):
            for sparsity in range(sp_levels):
                budget = self.budgets[module_id][sparsity]
                score = costs[module_id][sparsity]
                if budget == 0:
                    tmp = Ds[module_id - 1] + score
                    better = tmp < Ds[module_id]
                    if np.sum(better):
                        Ds[module_id][better] = tmp[better]
                        Ps[module_id][better] = sparsity
                    continue
                if budget > self.num_buckets:
                    continue
                tmp = Ds[module_id - 1][:-budget] + score
                better = tmp < Ds[module_id][budget:]
                if np.sum(better):
                    Ds[module_id][budget:][better] = tmp[better]
                    Ps[module_id][budget:][better] = sparsity

        score = np.min(Ds[-1, :])
        budget = np.argmin(Ds[-1, :])
        
        solution = []
        for module_id in range(len(Ds) - 1, -1, -1):
            solution.append(Ps[module_id][budget])
            budget -= self.budgets[module_id][solution[-1]]
        solution.reverse()

        return solution

    
    def search(self):
        num_layers = len(self.weight_database)
        num_evalutations = 0
        if self.verbose:
            _LOGGER.info('Finding init.')
        # init values
        best_coefs = None
        best_score = float('inf')
        best_solution = None
        for _ in range(self.num_rand_inits):
            coefs = np.random.uniform(0, 1, size=num_layers)
            #print(coefs.mean(), '+-', coefs.std())
            solution, score = self.evaluate(coefs)
            num_evalutations += 1
            if self.verbose:
                _LOGGER.info(f'Evaluation {num_evalutations} {score:.4f} (best {best_score:.4f})')
            if score < best_score:
                best_score = score
                best_coefs = coefs
                best_solution = solution

        if self.verbose:
            _LOGGER.info('Running local search.')
        for resamplings in range(int(self.resample_perc * num_layers), 0, -1):
            if self.verbose:
                _LOGGER.info(f'Trying {resamplings} resamplings ...')
            improved = True
            while improved: 
                improved = False
                for _ in range(self.patience):
                    coefs = best_coefs.copy()
                    for _ in range(resamplings):
                        coefs[np.random.randint(0, num_layers - 1)] = np.random.uniform(0, 1)
                    solution, score = self.evaluate(coefs)
                    num_evalutations += 1
                    if self.verbose:
                        _LOGGER.info(f'Evaluation {num_evalutations} {score:.4f} (best {best_score:.4f})')
                    if score < best_score:
                        best_score = score
                        best_coefs = coefs
                        best_solution = solution
                        improved = True
                        break
        # save best coefs
        self.best_coefs = coefs
        # save best solution
        self.best_solution = best_solution
        # save best solution
        if self.save_profile:
            torch.save({
                'coefs': torch.as_tensor(best_coefs),
                'solution': torch.as_tensor(best_solution) 
            })
