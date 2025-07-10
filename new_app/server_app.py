"""new-app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from new_app.task import HybridModel, get_weights
import torch
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json


# 🔑 Aggregation with saving to disk
def aggregate_fit_metrics(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in results)
    weighted_loss = sum(num_examples * metrics["train_loss"] for num_examples, metrics in results)
    avg_train_loss = weighted_loss / total_examples

    # ✅ Create folder
    Path("metrics").mkdir(exist_ok=True)

    # ✅ Save each client's loss curve
    for i, (num_examples, metrics) in enumerate(results):
        curve = metrics.get("loss_curve")
        accuracy_curve = metrics.get("accuracy_curve")
        if curve:
            with open(f"metrics/client_{i}_loss_curve.jsonl", "a") as f:
                json.dump(curve, f)
                f.write("\n")
        if accuracy_curve:
            with open(f"metrics/client_{i}_accuracy_curve.jsonl", "a") as f:
                json.dump(accuracy_curve, f)
                f.write("\n")

    return {"avg_train_loss": avg_train_loss}


def aggregate_evaluate_metrics(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in results)
    avg_metrics = {}
    metric_keys = results[0][1].keys()

    for key in metric_keys:
        weighted_sum = sum(num_examples * metrics[key] for num_examples, metrics in results)
        avg_metrics[f"avg_{key}"] = weighted_sum / total_examples

    return avg_metrics


def server_fn(context: Context):
    # Read config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    config = [3, 3, 0.0002, 0.3707, 7, 3, 3, 5, 1, 1, 8]
    model = HybridModel(config, 23)
    model.load_state_dict(
        torch.load(
            "models/model_30000_30000_00002_03707_70000_30000_30000_50000_10000_10000_80000.pt",
            weights_only=True,
        )
    )
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # ✅ Add custom `on_fit_config_fn` to pass `partition-id` + `final_round`
    def on_fit_config_fn(server_round: int):
        return {
            "final_round": server_round == num_rounds,
        }

    def on_evaluate_config_fn(server_round: int):
        return {
            "final_round": server_round == num_rounds,
        }

    # ✅ Attach partition id using node config (each client has its id already)
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
