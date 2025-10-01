import numpy as np
import os
import pandas as pd
from omegaconf import DictConfig
from itertools import product
import pickle


def generate_data(cfg: DictConfig, save_path: str, random_seed: int, n_actions: int, m_outcomes: int):
    np.random.seed(42+random_seed)
    # Step 1: Sample outcome values from [0, 10]
    v = np.sort(np.random.uniform(0, 10, size=m_outcomes))

    # Step 2: Softmax-normalized outcome distributions
    raw_outcome_logits = np.random.randn(n_actions, m_outcomes)
    p = np.exp(raw_outcome_logits) / np.sum(np.exp(raw_outcome_logits), axis=1, keepdims=True)

    # Step 3: Cost model
    expected_value = np.dot(p, v)
    cr = cfg.problem.alpha_p * expected_value
    ci = np.random.uniform(0, 1, size=n_actions)
    c = (1 - cfg.problem.beta_p) * cr + cfg.problem.beta_p * ci

    # Create DataFrame
    agent_setting = pd.DataFrame({
        "Expected Value": expected_value,
        "Final Cost": c
    })

    agent_pmf = pd.DataFrame(p, columns=[f"p{j}" for j in range(m_outcomes)])

    # Combine and filter required columns
    agent_setting = agent_setting.join(agent_pmf)[[f"p{j}" for j in range(m_outcomes)] + ["Final Cost"]]
    agent_setting["Expected Value"] = expected_value

    # Sort by expected value
    agent_setting = agent_setting.sort_values(by="Expected Value", ascending=True).drop(
        columns="Expected Value").reset_index(drop=True)

    # Display
    print(agent_setting.head())

    # Step 4: Generate valid contract–utility–action tuples
    contracts = []
    utilities = []
    actions = []
    a_utilities = []

    while len(contracts) < cfg.problem.num_contracts:
        w = np.random.uniform(0, 10, size=m_outcomes)  # random contract

        agent_utilities = np.dot(p, w) - c
        a_star = np.argmax(agent_utilities)
        a_utility = np.max(agent_utilities)

        principal_util = np.dot(p[a_star], v - w)

        if principal_util >= 0:  # enforce utility constraint
            contracts.append(w)
            utilities.append(principal_util) if agent_utilities[a_star] >= 0 else utilities.append(0)
            actions.append(1) if agent_utilities[a_star] >= 0 else actions.append(-1)
            a_utilities.append(a_utility)

    # Step 5: Assemble result DataFrame
    contract_logs = pd.DataFrame({
        "Contract": contracts,
        "Principal Utility": utilities,
        "Agent Action": actions
    })

    with open(save_path, "wb") as f:
        pickle.dump({"agent_setting": agent_setting, "contract_logs": contract_logs, "v": v}, f)


def generate_instances(cfg):
    basepath = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    n_instances = cfg.n_instances

    # for i in range(n_instances):
    #     generate_data(cfg, save_path=os.path.join(basepath, f"train_0{i + 1}.pkl"), random_seed=i,
    #                   n_actions=cfg.problem.n_actions, m_outcomes=cfg.problem.m_outcomes)

    data_configs = [[x, y] for x, y in product(cfg.problem.n_actions_l, cfg.problem.m_outcomes_l)]

    for n_actions, m_outcomes in data_configs:
        for i in range(n_instances):
            generate_data(cfg, save_path=os.path.join(basepath, f"eval{n_actions}_{m_outcomes}_0{i + 1}.pkl"),
                          random_seed=i, n_actions=n_actions, m_outcomes=m_outcomes)



