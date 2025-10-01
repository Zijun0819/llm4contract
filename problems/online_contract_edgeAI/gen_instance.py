import numpy as np
import os
import pandas as pd
from omegaconf import DictConfig
import pickle


def generate_data(cfg: DictConfig, save_path: str, n_actions: int, m_outcomes: int, seed: int = 1):
    '''
    The utility function of the principal and edge agents for processing one image are
    principal: $ln(1+\alpha_1 q(a))-r_s - r_b(a), in which v = ln(1+\alpha_1 q(a))-r_s, w = r_b(a)
    edge agent: r_b(a) - C_p(a) + r_s - C_t
    r_s = 1.6e-4, \alpha_1=0.0005, q(a)~[1.2, 1.8], C_t=1.2e-4
    The optimization variable is r_b(a), the bonus for different AIGC service quality
    :param seed: Aid for random seed
    :param cfg: configurations
    :param save_path: data save path
    :param n_actions: action set size of online learning edge AI problem, we consider it as 6, diffusion steps range
    from 5 to 10
    :param m_outcomes: the size of outcome space, equal to the bonus reward size, we assume it as 12, quality score of
    the diffusion model under varying diffusion steps is in the range of [1.2, 1.8, 0.05]
    :return:
    '''
    # The random seed we select 42, 50, 66, 99 as seeds for robustness assessment
    np.random.seed(99 + seed)
    # Step 1: Define outcome values as per the utility model of the principal
    quality_range = np.linspace(0.9, 1.8, m_outcomes + 1)
    quality_score = np.array([np.mean([quality_range[i], quality_range[i + 1]]) for i in range(len(quality_range) - 1)])
    assert (m_outcomes == quality_score.shape[0])
    v = np.log(1 + cfg.problem.alpha_1 * quality_score) - cfg.problem.reward_subs

    # Step 2: Obtain the outcome distributions
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, f'model_perf_dist_{m_outcomes}.csv')
    p_df = pd.read_csv(csv_path, header=None)
    p = p_df.iloc[1:n_actions + 1, 1:].astype(float).values

    # Step 3: Cost model, only consider the energy cost of the aigc model inference phase
    model_infer_cost = np.array([8e-6, 9.5e-6, 1.2e-5, 1.36e-5, 1.31e-5, 1.61e-5, 1.58e-5])
    _model_infer_cost = np.array([item - 8e-6 for item in model_infer_cost])
    c = _model_infer_cost[:n_actions]
    assert (n_actions == c.shape[0])

    # Create DataFrame
    agent_setting = pd.DataFrame({
        **{f"p{j}": [row[j] for row in p] for j in range(m_outcomes)},
        "Final Cost": c
    })

    # Display
    print(agent_setting.head())

    # Step 4: Generate valid contract–utility–action tuples
    contracts = []
    utilities = []
    actions = []

    while len(contracts) < cfg.problem.num_contracts:
        w = np.random.uniform(0, np.max(v), size=m_outcomes)  # random contract

        agent_utilities = np.dot(p, w) - c
        a_star = np.argmax(agent_utilities)

        principal_util = np.dot(p[a_star], v - w)

        if principal_util >= 0:  # enforce utility constraint
            contracts.append(w)
            utilities.append(principal_util) if agent_utilities[a_star] >= 0 else utilities.append(0)
            actions.append(1) if agent_utilities[a_star] >= 0 else actions.append(-1)

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
    n_instances = 3

    # Generate the training dataset
    default_n_action = cfg.problem.n_actions
    default_m_outcome = cfg.problem.m_outcomes
    for n in range(n_instances):
        generate_data(cfg, save_path=os.path.join(basepath, f"train_0{n+1}.pkl"),
                      n_actions=default_n_action, m_outcomes=default_m_outcome, seed=n)

    # Generate the evaluation dataset
    n_actions_l = cfg.problem.n_actions_l
    m_outcomes_l = cfg.problem.m_outcomes_l
    for n_actions in n_actions_l:
        for m_outcomes in m_outcomes_l:
            for n in range(n_instances):
                generate_data(cfg, save_path=os.path.join(basepath, f"edgeAI_{n_actions}_{m_outcomes}_0{n+1}.pkl"),
                              n_actions=n_actions, m_outcomes=m_outcomes, seed=n)
