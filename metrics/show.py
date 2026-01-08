import pickle
import numpy as np
import argparse
import sys
from pathlib import Path

# Add project root to sys.path to import logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from logger import logger
from rich.table import Table
from rich.console import Console


def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument("--path", type=str, default=None)
    args = parent_parser.parse_args()
    return args


def get_ism(results):
    ism_result = np.array(results)
    # print(ism_result)
    ism_score = ism_result[ism_result != -1].mean()
    fdfr_rate = (ism_result == -1).mean()
    return ism_score, fdfr_rate


# CLIP LIQE ISM FDFR


def print_results(path):
    with open(path, "rb") as f:
        results = pickle.load(f)
    if results["ism"].shape[0] != 800:
        logger.warning(
            f"Warning: {results['ism'].shape[0]} results found (expected 800)"
        )

    ism_score, fdfr_rate = get_ism(results["ism"])
    metric_threshold = 0.1318359375
    metric = "clip_iqac"

    table = Table(title="Evaluation Results")
    table.add_column("CLIP-IQAC", justify="center", style="cyan")
    table.add_column("LIQE", justify="center", style="magenta")
    table.add_column("ISM", justify="center", style="green")
    table.add_column("FDFR", justify="center", style="yellow")
    table.add_column("PSR", justify="center", style="blue")

    psr = sum(results[metric] < metric_threshold) / len(results[metric])

    table.add_row(
        f"{results['clip_iqac'].mean():.4f}",
        f"{results['liqe'].mean():.4f}",
        f"{ism_score:.4f}",
        f"{fdfr_rate:.4f}",
        f"{psr:.4f}",
    )

    console = Console()
    console.print(table)


args = get_args()
path = args.path
if path:
    print_results(path)
else:
    logger.error("No path provided")
