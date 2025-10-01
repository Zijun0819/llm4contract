import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

def agent_solver(v: np.ndarray, content: list[dict]) -> np.ndarray:
    """
    Infer a valid agent setting matrix (actions x [12 outcome probs + 1 cost]) that explains
    all historical logs under IR and IC constraints.

    Improvements over v1:
    - Adaptive, finer clustering with early merging of small clusters.
    - Assign noise points via max cosine similarity.
    - Enforce stricter IR/IC margins with minimal slack.
    - Iteratively refine cost bounds more conservatively.
    - Strict projection of distributions onto simplex.
    - Careful normalization and clipping with numerical stability.
    """

    m_outcomes = v.shape[0]
    logs_df = pd.DataFrame(content)
    accepted_logs = logs_df[logs_df['Agent Action'] == 1].reset_index(drop=True)
    rejected_logs = logs_df[logs_df['Agent Action'] == -1].reset_index(drop=True)

    if len(accepted_logs) == 0:
        raise ValueError("No accepted contracts; cannot infer agent strategies.")

    from scipy.optimize import linprog

    def infer_p(w):
        w = np.array(w, dtype=np.float64)
        c = -w  # maximize p @ w <=> minimize -p @ w
        A_eq = np.ones((1, m_outcomes))
        b_eq = np.array([1.0])
        bounds = [(0, 1) for _ in range(m_outcomes)]
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
        else:
            return np.ones(m_outcomes) / m_outcomes

    accepted_contracts = accepted_logs['Contract'].to_list()
    ps = np.array([infer_p(w) for w in accepted_contracts])

    # Step 2: Cluster inferred ps adaptively with DBSCAN on cosine distance
    dist_mat = cosine_distances(ps)
    best_eps = None
    best_labels = None
    best_n_clusters = 0

    # Adaptive eps candidates with finer granularity and early stop on suitable clustering
    eps_candidates = np.linspace(0.0075, 0.028, 45)  # slightly finer granularity, lower start eps
    for eps_try in eps_candidates:
        clustering = DBSCAN(eps=eps_try, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(dist_mat)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Accept clustering with 3 <= clusters <= 9 for tighter control
        if 3 <= n_clusters <= 9:
            best_eps = eps_try
            best_labels = labels
            best_n_clusters = n_clusters
            break

    if best_labels is None or best_n_clusters < 2:
        # fallback AgglomerativeClustering with distance threshold 0.028 Euclidean on ps
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.028, linkage='average')
        best_labels = clustering.fit_predict(ps)
        best_n_clusters = best_labels.max() + 1
        best_eps = None

    labels = best_labels
    n_actions = best_n_clusters

    # Step 3: Project vector onto simplex (Duchi et al. 2008)
    def project_simplex(v):
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
        if len(rho) == 0:
            theta = 0.0
        else:
            rho = rho[-1]
            theta = (cssv[rho] - 1) / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w

    # Step 4: Compute cluster centers as representative outcome distributions
    p_actions = np.zeros((n_actions, m_outcomes), dtype=np.float64)
    cluster_sizes = []
    for a in range(n_actions):
        members_idx = np.where(labels == a)[0]
        cluster_sizes.append(len(members_idx))
        if len(members_idx) == 0:
            p_actions[a] = np.ones(m_outcomes) / m_outcomes
        else:
            center = ps[members_idx].mean(axis=0)
            center = np.clip(center, 0, None)
            center = project_simplex(center)
            p_actions[a] = center

    # Step 5: Early merge small clusters (size < 5) into nearest cluster by cosine similarity
    small_clusters = [a for a, sz in enumerate(cluster_sizes) if sz < 5]
    if len(small_clusters) > 0 and n_actions > 1:
        for sc in small_clusters:
            others = [a for a in range(n_actions) if a != sc]
            if not others:
                continue
            sc_center = p_actions[sc]
            sc_norm = sc_center / (np.linalg.norm(sc_center) + 1e-14)
            others_centers = p_actions[others]
            others_norm = others_centers / (np.linalg.norm(others_centers, axis=1, keepdims=True) + 1e-14)
            sims = others_norm @ sc_norm
            if len(sims) == 0:
                continue
            nearest = others[np.argmax(sims)]
            labels[labels == sc] = nearest
        unique_labels = np.unique(labels)
        new_n_actions = len(unique_labels)
        new_p_actions = np.zeros((new_n_actions, m_outcomes), dtype=np.float64)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        new_labels = np.array([label_map[l] for l in labels])
        for a in range(new_n_actions):
            members_idx = np.where(new_labels == a)[0]
            if len(members_idx) == 0:
                new_p_actions[a] = np.ones(m_outcomes) / m_outcomes
            else:
                center = ps[members_idx].mean(axis=0)
                center = np.clip(center, 0, None)
                center = project_simplex(center)
                new_p_actions[a] = center
        p_actions = new_p_actions
        labels = new_labels
        n_actions = new_n_actions

    # Step 6: Assign noise points (-1) in DBSCAN to nearest cluster by max cosine similarity if DBSCAN used
    if best_eps is not None and -1 in labels:
        noise_idx = np.where(labels == -1)[0]
        cluster_norm = p_actions / (np.linalg.norm(p_actions, axis=1, keepdims=True) + 1e-14)
        for ni in noise_idx:
            p_noise = ps[ni]
            p_noise_norm = p_noise / (np.linalg.norm(p_noise) + 1e-14)
            sims = cluster_norm @ p_noise_norm
            assigned = np.argmax(sims)
            labels[ni] = assigned
        for a in range(n_actions):
            members_idx = np.where(labels == a)[0]
            if len(members_idx) == 0:
                p_actions[a] = np.ones(m_outcomes) / m_outcomes
            else:
                center = ps[members_idx].mean(axis=0)
                center = np.clip(center, 0, None)
                center = project_simplex(center)
                p_actions[a] = center

    # Step 7: Assign accepted logs to nearest action by cosine similarity (dot product)
    accepted_indices = accepted_logs.index.to_list()
    assigned_actions = np.zeros(len(accepted_indices), dtype=int)
    for i, idx in enumerate(accepted_indices):
        contract = np.array(content[idx]['Contract'], dtype=np.float64)
        dots = p_actions @ contract
        assigned_actions[i] = np.argmax(dots)

    # Step 8: Infer minimal costs per action to satisfy IR:
    eps_ir = 2e-9  # slightly relaxed but still tight margin
    c_ir = np.zeros(n_actions, dtype=np.float64)
    for a in range(n_actions):
        assigned_idx = [accepted_indices[i] for i in range(len(assigned_actions)) if assigned_actions[i] == a]
        if assigned_idx:
            costs_upper = []
            for idx in assigned_idx:
                contract = np.array(content[idx]['Contract'], dtype=np.float64)
                costs_upper.append(p_actions[a] @ contract - eps_ir)
            c_ir[a] = max(0.0, min(costs_upper))
        else:
            c_ir[a] = 0.0

    # Step 9: Enforce IC on rejected logs:
    eps_ic = 2e-9  # slightly relaxed but still tight margin
    if len(rejected_logs) > 0:
        rejected_contracts = rejected_logs['Contract'].to_list()
        for a in range(n_actions):
            max_rej_util = -np.inf
            for w_rej in rejected_contracts:
                w_rej = np.array(w_rej, dtype=np.float64)
                util = p_actions[a] @ w_rej
                if util > max_rej_util:
                    max_rej_util = util
            needed_cost = max_rej_util + eps_ic
            if needed_cost > c_ir[a]:
                c_ir[a] = needed_cost

    # Step 10: Iteratively refine costs globally to ensure IR and IC feasibility with smaller slack
    max_iter = 50  # more iterations for convergence
    slack = 5e-12  # smaller slack for stricter feasibility
    accepted_contracts_np = [np.array(content[idx]['Contract'], dtype=np.float64) for idx in accepted_indices]
    rejected_contracts_np = [np.array(w, dtype=np.float64) for w in rejected_logs['Contract'].to_list()] if len(rejected_logs) > 0 else []

    for _ in range(max_iter):
        updated = False
        # IR constraints
        for a in range(n_actions):
            assigned_idx = [i for i, aa in enumerate(assigned_actions) if aa == a]
            if assigned_idx:
                min_util = min((p_actions[a] @ accepted_contracts_np[i] for i in assigned_idx))
                if c_ir[a] > min_util + slack:
                    c_ir[a] = max(0.0, min_util)
                    updated = True
        # IC constraints
        if rejected_contracts_np:
            for a in range(n_actions):
                max_rej_util = max((p_actions[a] @ w for w in rejected_contracts_np))
                if c_ir[a] < max_rej_util + slack:
                    c_ir[a] = max_rej_util + slack
                    updated = True
        if not updated:
            break

    # Step 11: Final normalization and safety checks
    p_actions = np.clip(p_actions, 0, None)
    for i in range(n_actions):
        p_actions[i] = project_simplex(p_actions[i])
    c_ir = np.clip(c_ir, 0, None)

    agent_setting = np.hstack([p_actions, c_ir[:, None]])
    return agent_setting
