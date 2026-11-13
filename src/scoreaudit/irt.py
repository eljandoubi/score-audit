"""A graded response model for the trait scores, fitted by marginal maximum likelihood.

Each trait is an item with an ordered score. A writer has one latent proficiency, drawn from a
standard normal. The chance of reaching score k or more on item j is a logistic function of
the proficiency, with a discrimination a_j and a threshold b_jk for each step.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GradedResponse:
    a: np.ndarray  # discrimination of each item, shape (items,)
    b: np.ndarray  # thresholds, shape (items, categories - 1), increasing along the last axis
    loglik: float
    nodes: np.ndarray
    weights: np.ndarray


def recode(scores, low: float = 2.0, high: float = 4.5, step: float = 0.5) -> np.ndarray:
    """Categories 0, 1, 2... from half-point scores. Scores beyond `low` and `high` share the end
    category, since the few essays there are too few to place a threshold."""
    levels = np.rint((np.asarray(scores, dtype=float) - low) / step).astype(int)
    return np.clip(levels, 0, int(round((high - low) / step)))


def _nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return nodes, weights / np.sqrt(2 * np.pi)


def _category_probs(a, b, nodes):
    """Probability of each category for each item at each node, shape (items, nodes, categories)."""
    reach = torch.sigmoid(a[:, None, None] * (nodes[None, :, None] - b[:, None, :]))
    ones = torch.ones(reach.shape[0], reach.shape[1], 1, dtype=reach.dtype)
    zeros = torch.zeros_like(ones)
    return torch.cat([ones, reach], dim=2) - torch.cat([reach, zeros], dim=2)


def _params(raw_a, raw_first, raw_gaps):
    a = torch.nn.functional.softplus(raw_a)
    steps = torch.cat([raw_first[:, None], torch.nn.functional.softplus(raw_gaps)], dim=1)
    return a, torch.cumsum(steps, dim=1)


def _log_lik_by_node(responses, a, b, nodes):
    """log of the chance of each writer's answers at each node, shape (writers, nodes)."""
    log_p = torch.log(_category_probs(a, b, nodes).clamp_min(1e-12)).permute(0, 2, 1)
    items = torch.arange(responses.shape[1])[None, :]
    return log_p[items, responses].sum(dim=1)


def fit(responses: np.ndarray, n_nodes: int = 41, iterations: int = 150) -> GradedResponse:
    """Fit the model to an (writers x items) table of categories 0, 1, 2..."""
    x = torch.as_tensor(np.asarray(responses), dtype=torch.long)
    n_items, n_categories = x.shape[1], int(x.max()) + 1
    nodes_np, weights_np = _nodes(n_nodes)
    nodes = torch.as_tensor(nodes_np, dtype=torch.float64)
    log_w = torch.log(torch.as_tensor(weights_np, dtype=torch.float64))

    # start the thresholds at the logit of the share that reaches each category
    start = torch.zeros(n_items, n_categories - 1, dtype=torch.float64)
    for j in range(n_items):
        for k in range(1, n_categories):
            share = float((x[:, j] >= k).double().mean().clamp(0.005, 0.995))
            start[j, k - 1] = -np.log(share / (1 - share))
    start, _ = torch.sort(start, dim=1)
    gaps = (start[:, 1:] - start[:, :-1]).clamp_min(0.05)
    raw_gaps = torch.log(torch.expm1(gaps)).clone().requires_grad_(True)
    raw_first = start[:, 0].clone().requires_grad_(True)
    raw_a = torch.full((n_items,), 0.5413, dtype=torch.float64, requires_grad=True)  # a = 1

    optimizer = torch.optim.LBFGS(
        [raw_a, raw_first, raw_gaps], lr=1.0, max_iter=iterations, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        a, b = _params(raw_a, raw_first, raw_gaps)
        marginal = torch.logsumexp(_log_lik_by_node(x, a, b, nodes) + log_w, dim=1)
        loss = -marginal.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        a, b = _params(raw_a, raw_first, raw_gaps)
        marginal = torch.logsumexp(_log_lik_by_node(x, a, b, nodes) + log_w, dim=1)
    return GradedResponse(a.numpy(), b.numpy(), float(marginal.sum()), nodes_np, weights_np)


def proficiency(model: GradedResponse, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected a posteriori proficiency of each writer, and the standard deviation around it."""
    x = torch.as_tensor(np.asarray(responses), dtype=torch.long)
    nodes = torch.as_tensor(model.nodes, dtype=torch.float64)
    a = torch.as_tensor(model.a, dtype=torch.float64)
    b = torch.as_tensor(model.b, dtype=torch.float64)
    log_w = torch.log(torch.as_tensor(model.weights, dtype=torch.float64))
    with torch.no_grad():
        posterior = torch.softmax(_log_lik_by_node(x, a, b, nodes) + log_w, dim=1)
    mean = posterior @ nodes
    var = posterior @ nodes**2 - mean**2
    return mean.numpy(), np.sqrt(np.maximum(var.numpy(), 0))


def reliability(theta: np.ndarray, sd: np.ndarray) -> float:
    """Share of the variance of the proficiency estimates that is signal, not estimation error."""
    signal = theta.var(ddof=1)
    return float(signal / (signal + (sd**2).mean()))


def cronbach_alpha(table: np.ndarray) -> float:
    """Internal consistency of the items of an (writers x items) table."""
    x = np.asarray(table, dtype=float)
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1)))
