import os
import time
import math
import torch
import numpy as np
import torch.nn as nn


from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


__all__ = [
    "safe_cholesky_inv",
    "OBSHandle",
    "WeightDatabase",
    "SPDY"
]


def safe_cholesky_inv(X: Tensor, rel_damp: float = 1e-2):
    try:
        return torch.cholesky_inverse(torch.linalg.cholesky(X))
    except RuntimeError:
        reg = (rel_damp * torch.diag(X).mean()) * torch.eye(X.shape[0], device=X.device)
        return torch.cholesky_inverse(torch.linalg.cholesky(X + reg))


class OBSHandle:

    def __init__(
        self, 
        layer: Module,
        num_samples: int,
        dim_batch_size: int,
        rel_damp: float = 0.0,
        dtype_H = torch.float32,
        verbose: bool = False,
        layer_name: str = ''
    ) -> None:
        assert isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        self.layer = layer
        self.num_samples = num_samples
        self.dim_batch_size = dim_batch_size
        self.rel_damp = rel_damp
        self.verbose = verbose
        self.layer_name = layer_name
        # get weight
        W = layer.weight
        self.device = W.device
        # convert weight to the matrix form (d_out, d_in)
        self.dim_out  = W.shape[0]
        self.dim_in   = np.prod(W.shape[1:])
        self.H = torch.zeros((self.dim_in, self.dim_in), device=self.device, dtype=dtype_H)
        # init weight handle
        self.W =  None
        # init the loss evolution
        self.losses = None
        # init weight traces
        self.traces = None


    def update_H(self, inp: Tensor) -> None:
        # unfold inp if needed
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            inp = inp.view(-1, inp.shape[-1])
        self.H += (2 / self.num_samples) * inp.T @ inp


    def prepare(self) -> None:
        self.W = self.layer.weight.data.clone()
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.W = self.W.flatten(1)
        # if the entire input is 0 -> channel is dead and doesn't contribute
        dead = torch.diag(self.H) == 0
        self.H[dead, dead] = 1
        self.W[:, dead] = 0
        # prepare losses
        self.losses = torch.zeros((self.dim_out, self.dim_in + 1), device=self.device)
        # prepare traces
        self.traces = torch.zeros((self.dim_in + 1, self.dim_out, self.dim_in), device='cpu')


    def prepare_batch(self, i_start, i_end) -> tuple[Tensor]:
        W_batch = self.W[i_start:i_end, :]
        mask_batch = torch.zeros_like(W_batch).bool()
        return W_batch, mask_batch


    def prepare_batch_sparse(self, W_batch, mask_batch): 
        min_zeros = torch.sum((W_batch == 0), dim=1).min().item()
        # temporary hessian
        H_inv_batch = torch.empty((W_batch.shape[0], *self.H.shape), device=self.device)
        for i in range(W_batch.shape[0]):
            zero_ids = (W_batch[i] == 0)
            H_cur = self.H.clone()
            H_cur[zero_ids, :] = 0
            H_cur[:, zero_ids] = 0
            H_cur[zero_ids, zero_ids] = 1
            # invert
            H_inv_batch[i] = safe_cholesky_inv(H_cur)
            mask_batch[i, torch.nonzero(zero_ids, as_tuple=True)[0][:min_zeros]] = True

        return H_inv_batch, min_zeros


    def prepare_losses_and_traces(self) -> None:
        # prepare all
        self.prepare()

        _start = time.perf_counter()

        for i_start in range(0, self.dim_out, self.dim_batch_size):
            i_end = min(i_start + self.dim_batch_size, self.dim_out)
            batch_size = i_end - i_start
            batch_ids = torch.arange(batch_size, device=self.device)
            # prepare batch 
            W_batch, mask_batch = self.prepare_batch(i_start, i_end)
            H_inv_batch, min_nnz = self.prepare_batch_sparse(W_batch, mask_batch) 
            # init weight traces
            trace = torch.zeros((self.dim_in + 1, i_end - i_start, self.dim_in), device=self.device)
            trace[:(min_nnz + 1), :, :] = W_batch      

            for zeros in range(min_nnz + 1, self.dim_in + 1):
                _diag = torch.diagonal(H_inv_batch, dim1=1, dim2=2)
                scores = (W_batch ** 2) / _diag
                scores[mask_batch] = float('inf')
                pruned_id = torch.argmin(scores, 1)
                self.losses[i_start: i_end, zeros] = scores[batch_ids, pruned_id]
                row = H_inv_batch[batch_ids, pruned_id, :]
                d = _diag[batch_ids, pruned_id]
                W_batch -= row * (W_batch[batch_ids, pruned_id] / d).unsqueeze(1)
                mask_batch[batch_ids, pruned_id] = True
                W_batch[mask_batch] = 0
                trace[zeros, :, :] = W_batch
                # do not update on the last iteration
                if zeros == self.dim_in:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                H_inv_batch -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))

            self.losses[i_start: i_end, :] /= 2
            self.traces[:, i_start: i_end, :] = trace.cpu()

            torch.cuda.synchronize()

        _end = time.perf_counter()

        if self.verbose:
            print(f'[{self.layer_name}] Preparation of losses and traces took {(_end - _start):.2f} s')
            

    def get_pruning_database(self, sparsities: np.ndarray) -> Tensor:
        losses = self.losses[:, 1:].reshape(-1)
        order = torch.argsort(losses)
        Ws = torch.zeros((len(sparsities), self.dim_out, self.dim_in), device=self.device)
        cum_losses = [0] * len(sparsities)

        for i in range(self.dim_out):
            for j, sparsity in enumerate(sparsities):
                count = int(math.ceil(self.dim_out * self.dim_in * sparsity))
                perrow = torch.sum(
                    torch.div(order[:count], self.dim_in, rounding_mode='trunc') == i
                ).item()
                cum_losses[j] += torch.sum(self.losses[i, :(perrow + 1)]).item()
                Ws[j, i, :] = self.traces[perrow, i, :].to(self.device)
        
        if self.verbose:
            for sparsity, cum_loss in zip(sparsities, cum_losses):
                print(f'Sparsity: {sparsity:.3f} / Loss: {cum_loss:.4f}')

        return Ws


    def free(self) -> None:
        del self.H
        del self.W
        del self.losses
        del self.traces
        torch.cuda.empty_cache()


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
        save_profile_path: str = './best_coefs.npy',
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
            print('Finding init.')
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
                print(f'Evaluation {num_evalutations} {score:.4f} (best {best_score:.4f})')
            if score < best_score:
                best_score = score
                best_coefs = coefs
                best_solution = solution

        if self.verbose:
            print('Running local search.')
        for resamplings in range(int(self.resample_perc * num_layers), 0, -1):
            if self.verbose:
                print(f'Trying {resamplings} resamplings ...')
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
                        print(f'Evaluation {num_evalutations} {score:.4f} (best {best_score:.4f})')
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
