import pickle

from dotenv import load_dotenv
import os
import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path
from utils.utils import init_client, print_hyperlink
from llmreevo import LLMReEvo
from problems.online_contract_edgeAI.gen_instance import generate_instances
import subprocess
from problems.online_contract_edgeAI.benchmarks.bandit_solution import run_bandit
from problems.online_contract_edgeAI.benchmarks.delu_solution import run_delu
from problems.online_contract_edgeAI.benchmarks.llm_solution import run_llm
load_dotenv()
ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")

    # Generate dataset05 used for llm solver training and validation, uncomment it at the first run
    # generate_instances(cfg=cfg)

    # # initialize the LLM clients for solving the problem iteratively
    # client = init_client(cfg)
    # # optional clients for operators (ReEvo)
    # long_ref_llm = hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
    # short_ref_llm = hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
    # crossover_llm = hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
    # mutation_llm = hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None
    #
    # # ============== Training LLM agents is start from here ==============
    # # Load data from dataset
    # data_path = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/dataset/train_01.pkl"
    # with open(data_path, "rb") as f:
    #     data = pickle.load(f)
    # contract_logs = data["contract_logs"]
    # v = data["v"]
    # llmReEvo = LLMReEvo(cfg, ROOT_DIR, contract_logs=contract_logs, v=v, generator_llm=client)
    #
    # best_code_overall, best_code_path_overall = llmReEvo.evolve()
    # logging.info(f"Best Code Overall: {best_code_overall}")
    # best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
    # logging.info(f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}")
    #
    # with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w') as file:
    #     file.writelines(best_code_overall + '\n')

    # ============== Use LLM agents for inference is start from here ==============
    # run_llm(cfg)
    # run_bandit(cfg, split_level=3)
    run_delu(cfg, N_samples=300, split_level=3)


if __name__ == "__main__":
    main()
