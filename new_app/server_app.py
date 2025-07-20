"""new-app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from new_app.task import (
    HybridModel,
    get_weights,
    decompress_model_weights,
    cluster_model_weights,
    self_compress,
)
import torch
import pickle
import numpy as np  # ✅ Required for packing clustered bytes
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json

# ✅ Compression starts after N rounds
COMPRESSION_START_ROUND = 20

class CustomFedAvg(FedAvg):
    def __init__(self, on_aggregate_fit_fn=None, on_round_end_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.on_aggregate_fit_fn = on_aggregate_fit_fn
        self.on_round_end_fn = on_round_end_fn
    
    def aggregate_fit(self, server_round, results, failures):
        if self.on_aggregate_fit_fn is not None:
            results, failures = self.on_aggregate_fit_fn(server_round, results, failures)
        return super().aggregate_fit(server_round, results, failures)

    def __call__(self, server_round, client_manager):
        fit_ins, evaluate_ins, parameters, history = super().__call__(server_round, client_manager)
        if self.on_round_end_fn is not None:
            new_parameters = self.on_round_end_fn(server_round, parameters, history)
            return fit_ins, evaluate_ins, new_parameters, history
        return fit_ins, evaluate_ins, parameters, history
    
def aggregate_fit_metrics(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in results)
    weighted_loss = sum(num_examples * metrics["train_loss"] for num_examples, metrics in results)
    avg_train_loss = weighted_loss / total_examples

    Path("metrics").mkdir(exist_ok=True)
    for i, (_, metrics) in enumerate(results):
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
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    config = [3, 3, 0.0002, 0.3707, 7, 3, 3, 5, 1, 1, 8]
    model = HybridModel(config, 23)
    model.load_state_dict(
        torch.load(
            "models/model_30000_30000_00002_03707_70000_30000_30000_50000_10000_10000_80000.pt",
            weights_only=True,
        )
    )
    model.to("cpu")

    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    def on_fit_config_fn(server_round: int):
        return {
            "final_round": server_round == num_rounds,
            "use_compression": server_round >= COMPRESSION_START_ROUND,
        }

    def on_evaluate_config_fn(server_round: int):
        return {
            "final_round": server_round == num_rounds,
            "use_compression": False,
        }

    def on_aggregate_fit_fn(server_round: int, weights_results, failures):
        new_results = []
        for fit_res in weights_results:
            params = parameters_to_ndarrays(fit_res.parameters)
            if isinstance(params, list) and len(params) == 1 and isinstance(params[0], np.ndarray):
                # ✅ Received clustered: decompress
                clustered = pickle.loads(params[0].tobytes())
                nds = decompress_model_weights(clustered)
            else:
                nds = params
                
            new_fit_res = FitRes(
                parameters=ndarrays_to_parameters(nds),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics
                )
            
            new_results.append(new_fit_res)
        return new_results, failures

    def on_round_end_fn(server_round: int, parameters, _):
        new_weights = parameters_to_ndarrays(parameters)
        model = HybridModel(config, 23)
        model.load_state_dict(
            dict(zip(model.state_dict().keys(), [torch.tensor(v) for v in new_weights]))
        )
        model.to("cpu")

        if server_round >= COMPRESSION_START_ROUND:
            print(f"🔑 [Server] Clustering + Distillation at round {server_round} ...")

            # ✅ Self-compress: cluster, decompress, distill
            student = self_compress(
                model,
                trainloader=None,   # Optional: add server distillation data
                device="cpu",
                do_distill=True,
                num_clusters=50,
                epochs=1,
            )

            clustered = cluster_model_weights(student, num_clusters=50)

            # ✅ Serialize and pack as ndarray
            clustered_bytes = pickle.dumps(clustered)
            clustered_ndarray = np.frombuffer(clustered_bytes, dtype=np.uint8)

            parameters = ndarrays_to_parameters([clustered_ndarray])

            torch.save(student.state_dict(), f"models/compressed_round_{server_round}.pt")

            return parameters
        
        return ndarrays_to_parameters(new_weights)

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        on_aggregate_fit_fn=on_aggregate_fit_fn,
        on_round_end_fn=on_round_end_fn,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
