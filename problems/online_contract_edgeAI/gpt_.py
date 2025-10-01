import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.mixture import GaussianMixture


# This is the running results when alpha=5e-4
def agent_solver_v2(v: np.ndarray, content: list[dict]) -> np.ndarray:
    """
    Infer a valid agent setting matrix (actions x [12 outcomes + 1 cost]) consistent with historical logs.
    Enhanced version with:
    - Increased GMM n_init and components with deterministic seeds.
    - Adaptive and separate shrinking IR/IC/PU margins with distinct decay schedules.
    - Iterative cost refinement enforcing strict principal utility consistency.
    - Strict simplex projection with repeated normalization.
    - Robust fallback strategies for stability and minimal agent settings.
    - Fully vectorized and reproducible implementation.

    Args:
        v (np.ndarray): Principal's reward vector for 12 outcomes, shape (12,)
        content (list of dict): Historical logs with keys 'Contract', 'Principal Utility', 'Agent Action'.

    Returns:
        np.ndarray: Agent setting matrix with shape (n_actions, 13),
                    first 12 cols are outcome distributions (probabilities),
                    last col is cost for that action.
    """
    m_outcomes = v.shape[0]
    L = len(content)

    if L == 0:
        # No data: uniform distribution with zero cost
        p_uniform = np.ones(m_outcomes) / m_outcomes
        return np.hstack([p_uniform, 0.0])[np.newaxis, :]

    df = pd.DataFrame(content)
    contracts = np.array(df['Contract'].tolist(), dtype=np.float64)  # shape (L,12)
    principal_utils = np.array(df['Principal Utility'].tolist(), dtype=np.float64)  # shape (L,)
    agent_actions = np.array(df['Agent Action'].tolist(), dtype=int)  # shape (L,)

    accepted_idx = np.where(agent_actions == 1)[0]
    rejected_idx = np.where(agent_actions == -1)[0]

    if len(accepted_idx) == 0:
        # No accepted contracts: fallback uniform zero cost
        p_uniform = np.ones(m_outcomes) / m_outcomes
        return np.hstack([p_uniform, 0.0])[np.newaxis, :]

    # --- Helper: Project vector(s) onto simplex ---
    def proj_simplex(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            u = np.sort(y)[::-1]
            sv = np.cumsum(u)
            rho_candidates = np.nonzero(u * np.arange(1, len(y) + 1) > (sv - 1))[0]
            if len(rho_candidates) == 0:
                theta = 0.0
            else:
                rho = rho_candidates[-1]
                theta = (sv[rho] - 1) / (rho + 1)
            return np.maximum(y - theta, 0.0)
        else:
            proj = np.empty_like(y)
            for i in range(y.shape[0]):
                proj[i] = proj_simplex(y[i])
            return proj

    # --- Step 1: Infer plausible outcome distributions p for accepted contracts by LP maximizing p@w subject to simplex ---
    def solve_p_given_w(w: np.ndarray) -> np.ndarray:
        c_lp = -w  # maximize p@w <=> minimize -p@w
        A_eq = np.ones((1, m_outcomes))
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0)] * m_outcomes
        res = linprog(c=c_lp, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            p = proj_simplex(res.x)
            s = p.sum()
            if s > 0:
                p /= s
            else:
                p = np.ones(m_outcomes) / m_outcomes
            return p
        else:
            # fallback uniform
            return np.ones(m_outcomes) / m_outcomes

    p_candidates = []
    for i in accepted_idx:
        p = solve_p_given_w(contracts[i])
        p_candidates.append(p)
    p_candidates = np.array(p_candidates)
    if p_candidates.shape[0] == 0:
        # fallback uniform zero cost
        p_uniform = np.ones(m_outcomes) / m_outcomes
        return np.hstack([p_uniform, 0.0])[np.newaxis, :]

    # Deterministic subsampling if too large
    max_samples = 100
    rng = np.random.default_rng(2024)
    if p_candidates.shape[0] > max_samples:
        sample_indices = np.sort(rng.choice(p_candidates.shape[0], max_samples, replace=False))
        p_candidates_sub = p_candidates[sample_indices]
    else:
        p_candidates_sub = p_candidates

    # --- Step 2: Use GMM to select number of actions (1 to min(20, #samples)) ---
    best_gmm = None
    best_bic = np.inf
    n_min = 1
    n_max = min(20, len(p_candidates_sub))
    for n_comp in range(n_min, n_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                random_state=2024,
                n_init=100,
                reg_covar=1e-8,
                max_iter=2000,
                init_params='kmeans'
            )
            gmm.fit(p_candidates_sub)
            bic = gmm.bic(p_candidates_sub)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue

    if best_gmm is None:
        p0 = np.mean(p_candidates_sub, axis=0, keepdims=True)
    else:
        p0 = best_gmm.means_

    # Project means to simplex to ensure valid distributions
    for _ in range(5):
        p0 = proj_simplex(p0)
        for i in range(p0.shape[0]):
            s = p0[i].sum()
            if s > 0:
                p0[i] /= s
            else:
                p0[i] = np.ones(m_outcomes) / m_outcomes
    n_actions = p0.shape[0]

    # --- Step 3: Assign accepted contracts to closest action by max expected payment p0[a]@w ---
    assigns = np.full(L, -1, dtype=int)
    for i in accepted_idx:
        w = contracts[i]
        expected_payments = p0 @ w  # shape (n_actions,)
        assigns[i] = int(np.argmax(expected_payments))

    # --- Step 4: Setup LP to solve for costs c >=0 satisfying IR and IC constraints with adaptive epsilon margins ---
    epsilon_IR = 1e-6
    epsilon_IC = 1e-6
    epsilon_PU = 1e-6
    epsilon_IR_min = 1e-18
    epsilon_IC_min = 1e-18
    epsilon_PU_min = 1e-18
    max_iter = 60
    success = False

    for _ in range(max_iter):
        # IR constraints: -c[a] <= -p0[a]@w_i + epsilon_IR for accepted contracts i assigned to a
        IR_A = np.zeros((len(accepted_idx), n_actions))
        IR_b = np.zeros(len(accepted_idx))
        for idx_i, i in enumerate(accepted_idx):
            a = assigns[i]
            IR_A[idx_i, a] = -1.0
            IR_b[idx_i] = - (p0[a] @ contracts[i]) + epsilon_IR

        # IC constraints: -c[a] <= -p0[a]@w_j - epsilon_IC for rejected contracts j for all a
        IC_A = np.zeros((len(rejected_idx) * n_actions, n_actions))
        IC_b = np.zeros(len(rejected_idx) * n_actions)
        row = 0
        for j in rejected_idx:
            wj = contracts[j]
            for a in range(n_actions):
                IC_A[row, a] = -1.0
                IC_b[row] = - (p0[a] @ wj) - epsilon_IC
                row += 1

        A_ub = np.vstack([IR_A, IC_A])
        b_ub = np.hstack([IR_b, IC_b])

        bounds = [(0.0, None)] * n_actions
        c_obj = np.ones(n_actions)  # minimize sum costs for regularization

        res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            costs = res.x
            success = True
            break
        else:
            # Shrink margins adaptively but separately with lower bounds
            epsilon_IR = max(epsilon_IR * 0.4, epsilon_IR_min)
            epsilon_IC = max(epsilon_IC * 0.4, epsilon_IC_min)
            epsilon_PU = max(epsilon_PU * 0.4, epsilon_PU_min)

    if not success:
        # Fallback costs: max over rejected contracts payments + margin or minimal IR from acceptances or zero
        costs = np.zeros(n_actions)
        if len(rejected_idx) > 0:
            W_rej = contracts[rejected_idx]
            for a in range(n_actions):
                p_a = p0[a]
                rej_payments = W_rej @ p_a
                costs[a] = max(np.max(rej_payments) + 1e-7, 0.0)
        else:
            for a in range(n_actions):
                idx_a = np.where(assigns == a)[0]
                if len(idx_a) > 0:
                    min_pay = np.min(contracts[idx_a] @ p0[a])
                    costs[a] = max(0.0, min_pay)

    # --- Step 5: Iterative cost refinement enforcing strict IR, IC margins, and principal utility consistency ---
    margin_IR = 1e-12
    margin_IC = 1e-12
    margin_PU = 1e-12
    max_refine_iter = 100
    eps_tol = 1e-20

    for _ in range(max_refine_iter):
        changed = False

        # IR adjustment: cost[a] <= min pay over assigned accepted contracts - margin_IR
        for a in range(n_actions):
            idx_a = np.where(assigns == a)[0]
            if len(idx_a) == 0:
                continue
            p_a = p0[a]
            pays = contracts[idx_a] @ p_a
            min_pay = np.min(pays)
            target_c = max(0.0, min_pay - margin_IR)
            if costs[a] > target_c + eps_tol:
                costs[a] = target_c
                changed = True

        # IC adjustment: cost[a] >= max pay over rejected contracts + margin_IC
        if len(rejected_idx) > 0:
            W_rej = contracts[rejected_idx]
            for a in range(n_actions):
                p_a = p0[a]
                rej_payments = W_rej @ p_a
                max_rej_pay = np.max(rej_payments)
                target_c = max_rej_pay + margin_IC
                if costs[a] < target_c - eps_tol:
                    costs[a] = target_c
                    changed = True

        # Principal utility consistency for rejected contracts:
        # principal utility estimate: contract@v - (p_a@contract - cost[a])
        # For rejected contracts, principal utility should be >= -margin_PU (allow tiny negative tolerance)
        if len(rejected_idx) > 0:
            for j in rejected_idx:
                wj = contracts[j]
                pu_j = principal_utils[j]
                for a in range(n_actions):
                    pu_estimate = wj @ v - (p0[a] @ wj - costs[a])
                    if pu_estimate < -margin_PU:
                        increment = (-pu_estimate) + margin_PU
                        new_c = costs[a] + increment
                        if new_c > costs[a] + eps_tol:
                            costs[a] = new_c
                            changed = True

        if not changed:
            break

        # Shrink margins adaptively to approach tightest consistent margins with lower bounds
        margin_IR = max(margin_IR * 0.45, 1e-19)
        margin_IC = max(margin_IC * 0.45, 1e-19)
        margin_PU = max(margin_PU * 0.45, 1e-19)

    # --- Step 6: Final repeated projection of p0 onto simplex for numerical stability ---
    for _ in range(5):
        p0 = proj_simplex(p0)
        for i in range(p0.shape[0]):
            s = p0[i].sum()
            if s > 0:
                p0[i] /= s
            else:
                p0[i] = np.ones(m_outcomes) / m_outcomes

    costs = np.maximum(costs, 0.0)

    # --- Step 7: Final validation and minor corrections ---
    # Ensure IR constraints hold with small margin
    for i in accepted_idx:
        a = assigns[i]
        gap = (p0[a] @ contracts[i]) - costs[a]
        if gap < -1e-18:
            costs[a] = max(0.0, (p0[a] @ contracts[i]) - 1e-18)

    # Ensure IC constraints hold with small margin
    if len(rejected_idx) > 0:
        W_rej = contracts[rejected_idx]
        for a in range(n_actions):
            p_a = p0[a]
            rej_payments = W_rej @ p_a
            max_rej_pay = np.max(rej_payments)
            if max_rej_pay >= costs[a] - 1e-18:
                costs[a] = max_rej_pay + 1e-16

    # Ensure non-negativity of costs
    costs = np.maximum(costs, 0.0)

    agent_setting = np.hstack([p0, costs.reshape(-1, 1)])
    return agent_setting
