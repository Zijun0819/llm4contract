import csv
import logging
import re
import inspect
import hydra
import os
import numpy as np
from pandas import DataFrame
from scipy.optimize import linprog


def init_client(cfg):
    global client
    if cfg.get("model", None):  # for compatibility
        model: str = cfg.get("model")
        temperature: float = cfg.get("temperature", 1.0)
        if model.startswith("gpt"):
            from utils.llm_clients.openai import OpenAIClient
            client = OpenAIClient(model, temperature)
    else:
        client = hydra.utils.instantiate(cfg.llm_client)
    return client


def file_to_string(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        return file.read()


def print_hyperlink(path, text=None):
    """Print hyperlink to file or folder for convenient navigation"""
    # Format: \033]8;;file:///path/to/file\033\\text\033]8;;\033\\
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"


def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logging.warning(
                    f"Iteration {iter_num}: Code Run {response_id} execution error! (see {print_hyperlink(stdout_filepath, 'stdout')}))"
                )
            else:
                logging.info(
                    f"Iteration {iter_num}: Code Run {response_id} successful! (see {print_hyperlink(stdout_filepath, 'stdout')})"
                )
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if start is None and (line.strip().startswith('import') or line.strip().startswith('from')):
                start = i
            if 'return' in line:
                end = i
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end + 1])

    if code_string is None:
        return None
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_func_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def extract_contract_elements(agent_setting: DataFrame) -> tuple:
    p = np.array(agent_setting[[f'p{i}' for i in range(agent_setting.shape[1]-1)]])
    c = np.array(agent_setting['Final Cost'])

    return p, c


def save_results(save_pth, data):
    with open(save_pth, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows(data)  # write all rows


def contract_oracle_solver(p, c, v):
    # --------------------------------------------------------------------
    # assume you already have:
    #   p         : np.ndarray, shape (n_actions, m_outcomes)
    #   costs     : np.ndarray, shape (n_actions,)     # your c vector
    #   v         : np.ndarray, shape (m_outcomes,)
    #   n_actions = p.shape[0]
    # --------------------------------------------------------------------
    n_actions = p.shape[0]
    epsilon = 1e-8  # small slack to enforce strict IC/IR
    tol = 1e-6  # tolerance for post‐solve check

    best_princ_util = -np.inf
    best_contract = None
    best_action = None

    for a_star in range(n_actions):
        # 1) objective: minimize p[a_star]·w
        c_lp = p[a_star].copy()

        # 2) build IC constraints in one go
        #    (p[a] - p[a_star])·w <= costs[a] - costs[a_star] - ε
        IC_A = p - p[a_star]  # shape (n_actions, m_outcomes)
        IC_b = c - c[a_star] - epsilon
        mask = np.arange(n_actions) != a_star
        A_ub = IC_A[mask]
        b_ub = IC_b[mask]

        # 3) IR constraint: -p[a_star]·w <= -costs[a_star] - ε
        A_ub = np.vstack([A_ub, -p[a_star]])
        b_ub = np.append(b_ub, -c[a_star] - epsilon)

        # 4) bounds: 0 ≤ w_i ≤ v_i
        bounds = [(0, v_i) for v_i in v]

        # 5) solve with Highs
        res = linprog(
            c=c_lp,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
        )
        if not res.success:
            continue

        w_opt = res.x

        # 6) verify IC with tolerance
        agent_utils = p.dot(w_opt) - c
        max_util = agent_utils.max()
        if agent_utils[a_star] + tol < max_util:
            # still violating IC
            print(
                f"IC violation for a*={a_star}: best at {np.argmax(agent_utils)} (Δ={max_util - agent_utils[a_star]:.2e})")
            continue

        # 7) compute principal’s utility
        princ_util = p[a_star].dot(v - w_opt)
        if princ_util > best_princ_util:
            best_princ_util = princ_util
            best_contract = w_opt.copy()
            best_action = a_star

    print("→ Best principal utility:", best_princ_util)
    # print("→ Best action index:   ", best_action)
    # print("→ Best contract w:      ", best_contract)
    # print("→ Best agent utility:   ", (p.dot(best_contract) - c).max())

    return best_princ_util, best_contract
