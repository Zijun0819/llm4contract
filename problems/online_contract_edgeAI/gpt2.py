import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linprog

def agent_solver_v2(v: np.ndarray, content: list[dict]) -> np.ndarray:
    """
    Infer a valid agent setting matrix (actions x [12 probabilities + 1 cost]) that explains
    all historical interaction logs (accept/reject) in an online contract design problem.

    Improvements over v1:
    - Prioritize strict IR satisfaction with minimal cost increments before addressing IC.
    - Use adaptive DBSCAN eps based on robust median cosine distance scaling.
    - Assign noise points precisely by cosine distance to cluster centers.
    - Normalize probability vectors robustly with numerical safeguards.
    - Enforce tighter IC margins per action, iteratively restore IR.
    - Ensure all costs remain non-negative.
    - Handle edge cases gracefully (no accepted contracts, all noise, etc.).

    Args:
        v: np.ndarray shape (12,), principal's reward vector for 12 outcomes.
        content: list of dicts with keys:
            - 'Contract': list or array of 12 payments,
            - 'Principal Utility': float,
            - 'Agent Action': int (1 accept, -1 reject).

    Returns:
        agent_setting: np.ndarray shape (n_actions, 13),
            each row: 12 outcome probabilities (sum to 1) + 1 non-negative cost.
    """
    m_outcomes = v.shape[0]
    logs_df = pd.DataFrame(content)

    def infer_p_from_w(w: np.ndarray) -> np.ndarray | None:
        # Maximize p @ w <=> minimize -p @ w with constraints sum p=1, p>=0
        c = -w
        A_eq = np.ones((1, m_outcomes))
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0)] * m_outcomes
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            p = res.x
            p = np.clip(p, 0, None)
            s = p.sum()
            if s > 0:
                p /= s
            else:
                p = np.ones(m_outcomes) / m_outcomes
            return p
        return None

    accepted_logs = logs_df[logs_df['Agent Action'] == 1]
    if accepted_logs.empty:
        # No accepted contracts: uniform distribution, zero cost
        uniform_p = np.ones(m_outcomes) / m_outcomes
        return np.hstack([uniform_p[np.newaxis, :], np.zeros((1, 1))])

    accepted_ps = []
    accepted_ws = []
    for _, row in accepted_logs.iterrows():
        w = np.array(row['Contract'], dtype=np.float64)
        p = infer_p_from_w(w)
        if p is not None:
            accepted_ps.append(p)
            accepted_ws.append(w)
    if not accepted_ps:
        # LP failed for all accepted contracts: fallback uniform zero cost
        uniform_p = np.ones(m_outcomes) / m_outcomes
        return np.hstack([uniform_p[np.newaxis, :], np.zeros((1, 1))])

    accepted_ps = np.vstack(accepted_ps)
    accepted_ws = np.vstack(accepted_ws)

    # Compute pairwise cosine distances robustly
    cos_dist = cosine_distances(accepted_ps)
    if len(accepted_ps) > 1:
        tri_upper = cos_dist[np.triu_indices(len(accepted_ps), k=1)]
        median_dist = np.median(tri_upper)
        # Adaptive eps: scale median_dist, clamp between 0.04 and 0.16 for robustness
        eps = max(0.04, min(0.16, median_dist * 1.3))
    else:
        eps = 0.1  # default eps for single point

    dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(cos_dist)

    unique_labels = [lab for lab in set(labels) if lab >= 0]
    if not unique_labels:
        # All noise, assign all to single cluster 0
        unique_labels = [0]
        labels = np.zeros(len(accepted_ps), dtype=int)

    # Assign noise points (-1) to nearest cluster center by cosine distance
    noise_idx = np.where(labels == -1)[0]
    if noise_idx.size > 0 and unique_labels:
        cluster_centers = []
        for lab in unique_labels:
            cluster_points = accepted_ps[labels == lab]
            center = cluster_points.mean(axis=0)
            center = np.clip(center, 0, None)
            s = center.sum()
            center = center / s if s > 0 else np.ones(m_outcomes) / m_outcomes
            cluster_centers.append(center)
        cluster_centers = np.vstack(cluster_centers)
        for ni in noise_idx:
            dists = cosine_distances(accepted_ps[ni:ni+1], cluster_centers).flatten()
            nearest_lab = unique_labels[np.argmin(dists)]
            labels[ni] = nearest_lab
    elif noise_idx.size > 0:
        # No clusters exist, assign all noise to cluster 0
        labels[noise_idx] = 0
        unique_labels = [0]

    # Recompute unique_labels after noise assignment
    unique_labels = sorted(set(labels))

    cluster_ps = []
    cluster_costs = []
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        p_cluster = accepted_ps[idxs].mean(axis=0)
        p_cluster = np.clip(p_cluster, 0, None)
        s = p_cluster.sum()
        p_cluster = p_cluster / s if s > 0 else np.ones(m_outcomes) / m_outcomes
        cluster_ps.append(p_cluster)

        w_cluster = accepted_ws[idxs]
        utilities = w_cluster @ p_cluster
        cost_a = utilities.min()
        cost_a = max(cost_a, 0.0)
        cluster_costs.append(cost_a)

    cluster_ps = np.vstack(cluster_ps)
    cluster_costs = np.array(cluster_costs)

    reject_logs = logs_df[logs_df['Agent Action'] == -1]
    reject_ws = None
    if not reject_logs.empty:
        reject_ws = np.vstack(reject_logs['Contract'].values)

    margin_ic = 1e-11  # tighter margin for IC
    margin_ir = 1e-11  # tighter margin for IR
    max_iter = 30

    for _ in range(max_iter):
        prev_costs = cluster_costs.copy()

        # Step 1: Enforce IR strictly with minimal increments
        for i, lab in enumerate(unique_labels):
            idxs = np.where(labels == lab)[0]
            w_cluster = accepted_ws[idxs]
            p_a = cluster_ps[i]
            min_expected_payment = (w_cluster @ p_a).min()
            # Tighten cost to min_expected_payment - margin_ir if possible
            desired_cost = min_expected_payment - margin_ir
            if cluster_costs[i] > desired_cost:
                cluster_costs[i] = max(desired_cost, 0.0)

        # Step 2: Enforce IC: cost >= max expected utility on rejects + margin
        if reject_ws is not None:
            # Compute utilities matrix: actions x reject contracts
            utilities = cluster_ps @ reject_ws.T - cluster_costs[:, np.newaxis]
            max_util_per_reject = utilities.max(axis=0)  # max over actions for each reject contract
            violation = max_util_per_reject.clip(min=0)
            if violation.size > 0 and violation.max() > -margin_ic:
                increment = violation.max() + margin_ic
                cluster_costs += increment

        # Ensure costs non-negative
        cluster_costs = np.clip(cluster_costs, 0, None)

        # If costs changed less than margin, consider converged
        if np.allclose(cluster_costs, prev_costs, atol=margin_ic*5):
            break

    # Final normalization and sanity fixes
    for i in range(len(cluster_ps)):
        p = cluster_ps[i]
        p = np.clip(p, 0, None)
        s = p.sum()
        cluster_ps[i] = p / s if s > 0 else np.ones(m_outcomes) / m_outcomes
        cluster_costs[i] = max(cluster_costs[i], 0.0)

    agent_setting = np.hstack([cluster_ps, cluster_costs[:, np.newaxis]])
    return agent_setting
