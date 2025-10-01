import math
import itertools
import os
import random
import time

import numpy as np
import pandas as pd
from utils.utils import contract_oracle_solver, save_results


# ------------------------------------------------------------------
# 2)  UCB multi-armed bandit for contract selection
# ------------------------------------------------------------------
class UCBContractBandit:
    def __init__(self, contracts, alpha1, r_s, contract_setting):
        self.contracts = contracts
        self.K = len(contracts)
        self.alpha1 = alpha1
        self.r_s = r_s
        self.contract_setting = contract_setting
        self.counts = np.zeros(self.K, dtype=int)
        self.means = np.zeros(self.K, dtype=float)
        self.t = 0

    def _ucb(self, i):
        if self.counts[i] == 0:
            return float("inf")  # explore every arm once
        return self.means[i] + math.sqrt(2 * math.log(self.t) / self.counts[i])

    def select_arm(self):
        self.t += 1
        return int(np.argmax([self._ucb(i) for i in range(self.K)]))

    def update(self, idx, reward):
        self.counts[idx] += 1
        n = self.counts[idx]
        self.means[idx] += (reward - self.means[idx]) / n

    def run(self, T, simulate_round):
        for _ in range(T):
            arm = self.select_arm()
            contract = self.contracts[arm]
            a_t, q_t, _ = simulate_round(contract, self.contract_setting)
            p, _, v, _ = self.contract_setting
            reward = np.dot(p[a_t], v - contract)
            self.update(arm, reward)
        return int(np.argmax(self.means))


def simulate_round(contract, cs: tuple):
    """
    Agent chooses a = argmax_a [ r_b(a) – C_p(a) ].
    Quality q(a) = quality_score[a].
    """
    p, c, v, quality_score = cs
    agent_utilities = np.dot(p, contract) - c
    a_star = np.argmax(agent_utilities)
    q_index = np.random.choice(len(p[a_star]), p=p[a_star])
    q = quality_score[q_index]
    principal_util = np.dot(p[a_star], v - contract)
    agent_util = np.max(agent_utilities)
    return a_star, q, (principal_util, agent_util)


# ------------------------------------------------------------------
# 1)  Generate candidate contracts via Cartesian product
# ------------------------------------------------------------------
def generate_candidate_contracts(max_v: float, k: int, levels: int = 10):
    """
    Split (0, max_v] into `levels` equally spaced points.
    Take the Cartesian product over the k coordinates to obtain candidate
    length-k contracts.  Total arms = levels**k.
    """
    grid = np.linspace(0, max_v, num=levels)
    return [np.array(p) for p in itertools.product(grid, repeat=k)]


def create_sys_model(cfg, m_outcomes, n_actions):
    # ------------------------------------------------------------------
    # principal-agent environment
    # ------------------------------------------------------------------
    quality_range = np.linspace(0.9, 1.8, m_outcomes + 1)
    quality_score = np.array([np.mean([quality_range[i], quality_range[i + 1]]) for i in range(len(quality_range) - 1)])
    assert (m_outcomes == quality_score.shape[0])
    v = np.log(1 + cfg.problem.alpha_1 * quality_score) - cfg.problem.reward_subs
    model_infer_cost = np.array([8e-6, 9.5e-6, 1.2e-5, 1.36e-5, 1.31e-5, 1.61e-5, 1.58e-5])
    _model_infer_cost = np.array([item - 8e-6 for item in model_infer_cost])
    c = _model_infer_cost[:n_actions]
    assert (n_actions == c.shape[0])

    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "../", f'model_perf_dist_{m_outcomes}.csv')
    p_df = pd.read_csv(csv_path, header=None)
    p = p_df.iloc[1:n_actions + 1, 1:].astype(float).values
    contract_setting = (p, c, v, quality_score)

    return contract_setting


def run_bandit(cfg, split_level: int = 3):
    print("================>Enter into the bandit algorithm<================")

    # Generate the training dataset
    n_actions = cfg.problem.n_actions
    m_outcomes = cfg.problem.m_outcomes
    # Generate the evaluation dataset
    n_actions_l = cfg.problem.n_actions_l
    m_outcomes_l = cfg.problem.m_outcomes_l
    horizon = 300  # learning horizon T
    alpha1 = cfg.problem.alpha_1
    r_s = cfg.problem.reward_subs
    n_instances = 3

    contract_to_optimal = []
    bonus_prin_u = []
    bonus_agent_u = []
    infer_time_l = []
    for m in m_outcomes_l:
        row_cto = []
        row_bpu = []
        row_bau = []
        row_itl = []
        for n in n_actions_l:
            cto = []
            bpu = []
            bau = []
            itl = []
            for i in range(n_instances):
                contract_setting = create_sys_model(cfg, m, n)
                p, c, v, _ = contract_setting
                max_v = v.max()

                # ------------------------------------------------------------------
                # Build candidate arm set and learn
                # ------------------------------------------------------------------
                time_start = time.time()
                contracts = generate_candidate_contracts(max_v, m, levels=split_level)
                random.shuffle(contracts)
                print(f"Generated {len(contracts)} candidate contracts (arms).")

                bandit = UCBContractBandit(contracts, alpha1, r_s, contract_setting)
                best_idx = bandit.run(horizon, simulate_round)
                inferred_contract = contracts[best_idx]
                time_end = time.time()
                infer_time = time_end - time_start
                wo_bonus_contract = np.zeros_like(inferred_contract)
                _, _, (principal_u, agent_u) = simulate_round(inferred_contract, contract_setting)
                _, _, (wob_principal_u, wob_agent_u) = simulate_round(wo_bonus_contract, contract_setting)
                oracle_best_prin_u, _ = contract_oracle_solver(p, c, v)
                bonus_improve_prin_u = round((principal_u - wob_principal_u) / wob_principal_u, 4)
                bonus_improve_agent_u = round(agent_u / 4e-5, 4)
                percentage_to_optimal = round(principal_u / oracle_best_prin_u, 4)
                cto.append(percentage_to_optimal)
                bpu.append(bonus_improve_prin_u)
                bau.append(bonus_improve_agent_u)
                itl.append(infer_time)
            row_cto.append(round(np.mean(cto), 4))
            row_bpu.append(round(np.mean(bpu), 4))
            row_bau.append(round(np.mean(bau), 4))
            row_itl.append(round(np.mean(itl), 4))
        contract_to_optimal.append(row_cto)
        bonus_prin_u.append(row_bpu)
        bonus_agent_u.append(row_bau)
        infer_time_l.append(row_itl)

    save_results("edgeAI_to_optimal_bandit.csv", contract_to_optimal)
    save_results("edgeAI_bonus_prin_bandit.csv", bonus_prin_u)
    save_results("edgeAI_bonus_agent_bandit.csv", bonus_agent_u)
    save_results("edgeAI_inference_time_bandit.csv", infer_time_l)

