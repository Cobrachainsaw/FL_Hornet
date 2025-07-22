from logging import config
import os
import sys
import torch
import pickle
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from new_app.task import (
    HybridModel,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
    cluster_model_weights,
    decompress_model_weights,
)
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import time
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
mp.set_executable(sys.executable)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        print(f"[Client {partition_id}] 🚀 Initializing FlowerClient")
        print(f"[ClientFn] ✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"[ClientFn] 🧪 Python executable: {sys.executable}")
        print(f"[ClientFn] 🧪 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[ClientFn] 🧪 torch version: {torch.__version__}")
        print(f"[ClientFn] 🧪 torch.cuda.version: {torch.version.cuda}")
        print(f"[ClientFn] 🧪 torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
        print(f"[ClientFn] 🧪 torch.cuda.device_count(): {torch.cuda.device_count()}")    
        print(f"[Client {partition_id}] 🧠 torch.cuda.is_available(): {torch.cuda.is_available()}")
        
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Client {partition_id}] 🧠 Using device: {self.device}")
        self.net.to(self.device)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] ⚙️ fit() started")

        if isinstance(parameters, list):
            params_ndarrays = parameters
        else:
            params_ndarrays = parameters_to_ndarrays(parameters)

        if len(params_ndarrays) == 1 and params_ndarrays[0].dtype == np.uint8:
            print(f"[Client {self.partition_id}] 🔓 Decompressing clustered weights...")
            try:
                clustered = pickle.loads(params_ndarrays[0].tobytes())
                params_ndarrays = decompress_model_weights(clustered)
            except Exception as e:
                print(f"[Client {self.partition_id}] ❌ Decompression failed: {e}")
                raise

        set_weights(self.net, params_ndarrays)

        print(f"[Client {self.partition_id}] 🏋️ Training started")
        start = time.time()
        try:
            epoch_losses, epoch_accuracies = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
                partition_id=self.partition_id,
            )
        except Exception as e:
            print(f"[Client {self.partition_id}] ❌ Exception during training: {e}")
            raise
        end = time.time()
        print(f"[Client {self.partition_id}] ✅ Training finished in {end - start:.2f}s")

        updated_weights = get_weights(self.net)

        # Step 1: Cluster/compress weights
        try:
            clustered = cluster_model_weights(updated_weights)
            compressed = pickle.dumps(clustered)
            compressed_ndarray = np.frombuffer(compressed, dtype=np.uint8)
            print(f"[Client {self.partition_id}] 📦 Compressed weights: {compressed_ndarray.shape}")
        except Exception as e:
            print(f"[Client {self.partition_id}] ❌ Compression failed: {e}")
            raise

        # Step 2: Return compressed weights
        return (
            [compressed_ndarray],  # Must be a list of one ndarray (uint8)
            len(self.trainloader.dataset),
            {
                "train_loss": epoch_losses[-1],
                "loss_curve": epoch_losses,
                "accuracy_curve": epoch_accuracies,
            },
        )   

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] 🧪 evaluate() started")

        # Decode parameters
        if isinstance(parameters, list):
            params_ndarrays = parameters
        else:
            params_ndarrays = parameters_to_ndarrays(parameters)

        # Decompress if necessary
        if len(params_ndarrays) == 1 and params_ndarrays[0].dtype == np.uint8:
            print(f"[Client {self.partition_id}] 🔓 Detected compressed weights, decompressing...")
            try:
                clustered = pickle.loads(params_ndarrays[0].tobytes())
                params_ndarrays = decompress_model_weights(clustered)
                print(f"[Client {self.partition_id}] ✅ Decompression successful")
            except Exception as e:
                print(f"[Client {self.partition_id}] ❌ Decompression failed: {e}")
                raise

        # Set weights
        try:
            set_weights(self.net, params_ndarrays)
            print(f"[Client {self.partition_id}] 🧠 Model weights updated")
        except Exception as e:
            print(f"[Client {self.partition_id}] ❌ Failed to set model weights: {e}")
            raise

        # Perform evaluation
        print(f"[Client {self.partition_id}] 🧪 Evaluating model...")
        try:
            val_loss, val_acc, val_prec, val_rec, val_f1 = test(
                self.net,
                self.valloader,
                self.device,
                partition_id=self.partition_id,
            )
        except Exception as e:
            print(f"[Client {self.partition_id}] ❌ Evaluation failed: {e}")
            raise

        # Format results
        results = {"accuracy": val_acc}
        if config.get("final_round", False):
            results.update({
                "precision": val_prec,
                "recall": val_rec,
                "f1": val_f1,
            })

        print(f"[Client {self.partition_id}] ✅ Evaluation done: loss={val_loss:.4f}, accuracy={val_acc:.4f}")
        return val_loss, len(self.valloader.dataset), results



def client_fn(context: Context):
    print(f"[ClientFn] ✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"[ClientFn] ✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"[ClientFn] 🧪 Python executable: {sys.executable}")
    print(f"[ClientFn] 🧪 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[ClientFn] 🧪 torch version: {torch.__version__}")
    print(f"[ClientFn] 🧪 torch.cuda.version: {torch.version.cuda}")
    print(f"[ClientFn] 🧪 torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
    print(f"[ClientFn] 🧪 torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"[ClientFn] 🧬 Creating model and loading data...")
    config = [3, 3, 0.0002, 0.3707, 7, 3, 3, 5, 1, 1, 8]
    net = HybridModel(config, 23)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    print(f"[ClientFn] 🧪 Loading data for partition {partition_id}/{num_partitions}")
    try:
        trainloader, valloader = load_data(partition_id, num_partitions)
        print(f"[ClientFn] ✅ Data loaded: {len(trainloader.dataset)} train samples, {len(valloader.dataset)} val samples")
    except Exception as e:
        print(f"[ClientFn] ❌ Failed to load data: {e}")
        raise

    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()


app = ClientApp(client_fn)
