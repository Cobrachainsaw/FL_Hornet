import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from new_app.task import HybridModel, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        epoch_losses, epoch_accuracies = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            partition_id=config["partition-id"],  # 👈 pass to train for saving local metrics
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": epoch_losses[-1],
                "loss_curve": epoch_losses,
                "accuracy_curve": epoch_accuracies,
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        val_loss, val_acc, val_prec, val_rec, val_f1 = test(
            self.net,
            self.valloader,
            self.device,
            partition_id=config["partition-id"],  # 👈 pass to test for saving local val metrics
        )
        results = {
            "accuracy": val_acc,
        }
        if config.get("final_round", False):
            results.update({
                "precision": val_prec,
                "recall": val_rec,
                "f1": val_f1,
            })
        return val_loss, len(self.valloader.dataset), results


def client_fn(context: Context):
    config = [3, 3, 0.0002, 0.3707, 7, 3, 3, 5, 1, 1, 8]
    net = HybridModel(config, 23)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()


app = ClientApp(
    client_fn,
)
