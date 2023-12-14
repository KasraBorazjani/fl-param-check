import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os
from copy import deepcopy
import matplotlib.pyplot as plt

from dataset import create_client_ds
from plots import draw_bar_plot
from utils import setup_result_path
from model import simpleModel
import networkx as nx




class MNIST_client():
    """
    MNIST_client class for federated learning on the MNIST dataset.

    Args:
        original_ds (torch.utils.data.Dataset): The original MNIST dataset.
        client_id (int): Identifier for the client.
        label_idx_list (List[int]): List of label indices to include in the client's dataset.
        args (argparse.Namespace): Command-line arguments.
        neighbors: List of neighboring clients.

    Attributes:
        client_id (int): Identifier for the client.
        dataset (torch.utils.data.Dataset): Client-specific dataset.
        dataloader (torch.utils.data.DataLoader): DataLoader for the client's dataset.
        model (simpleModel): Client's model.
        updated_model (simpleModel): Client's model placeholder after aggregation.
        history (Dict[str, List[float]]): Training and validation accuracy and loss history.
        sgd_per_round (int): Number of stochastic gradient descent steps per federated round.
        device (str): Device (e.g., 'cpu' or 'cuda') for model training.
        criterion (torch.nn.Module): Negative log-likelihood loss criterion.
        neighbors: List of neighboring clients.
        init_lr (float): Initial learning rate for stochastic gradient descent.

    Methods:
        __init__: Initializes the MNIST_client object.
        train_one_round: Performs one round of federated learning on the client's dataset.
        validate_model: Validates the model on a given validation loader.

    Example:
        client = MNIST_client(original_ds, client_id, label_idx_list, args, neighbors)
        client.train_one_round()
        client.validate_model(val_loader)
    """
    
    def __init__(self, original_ds, client_id, label_idx_list, args, neighbors):

        """
        Initializes the MNIST_client object.

        Args:
            original_ds (torch.utils.data.Dataset): The original MNIST dataset.
            client_id (int): Identifier for the client.
            label_idx_list (List[int]): List of label indices to include in the client's dataset.
            args (argparse.Namespace): Command-line arguments.
            neighbors: List of neighboring clients.
        """

        self.client_id = client_id
        self.dataset, label_idx_list, self.label_set = create_client_ds(original_ds, label_idx_list, args.total_ds_len, args.primary_label_fraction, args.num_secondaries)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
        self.model = simpleModel()
        self.model.load_state_dict(torch.load(os.path.join(args.saved_model_path, 'initial_model_weights.pth')))
        self.updated_model = simpleModel()
        self.updated_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, 'initial_model_weights.pth')))
        self.history = {"train_acc":[], "train_loss":[], "val_acc":[], "val_loss":[]}
        self.sgd_per_round = args.sgd_per_round
        self.device = args.device
        self.model.to(self.device)
        self.updated_model.to(self.device)
        self.criterion = nn.NLLLoss()
        self.neighbors = neighbors
        self.init_lr = args.init_lr

    
    def train_one_round(self):

        """
        Performs one round of local training on the client's dataset.
        Updates the model based on the local dataset using stochastic gradient descent.
        """

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.5)
        self.model.train()
        train_correct = 0
        train_total = 0
        running_loss = 0
        for idx, (data, labels) in enumerate(self.dataloader):

            if idx > self.sgd_per_round:
                break

            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data.view(data.shape[0], -1))
            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(output, dim=1)
            train_correct += torch.sum(pred==labels).item()
            train_total += labels.size(0)
        
        self.history['train_loss'].append(running_loss/len(self.dataloader))
        self.history['train_acc'].append(100*train_correct/train_total)
    
    def validate_model(self, val_loader):

        """
        Validates the model on a given validation loader.

        Args:
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        """

        self.model.eval()
        
        with torch.no_grad():
            running_loss = 0
            correct = 0
            total = 0
            for idx, (data, labels) in enumerate(val_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                output = self.model(data.view(data.shape[0], -1))
                loss = self.criterion(output, labels)

                running_loss += loss.item()
                _, pred = torch.max(output, dim=1)
                correct += torch.sum(pred==labels).item()
                total += labels.size(0)
            
            self.history['val_loss'].append(running_loss/len(val_loader))
            self.history['val_acc'].append(100*correct/total)
        




class Network():
    """
    Network class for federated learning on the MNIST dataset.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Attributes:
        train_set (torchvision.datasets.MNIST): Training dataset.
        test_set (torchvision.datasets.MNIST): Testing dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        label_idxs (List[np.array]): List of label indices for each digit (0-9).
        network_graph (networkx.Graph): Graph representing the network structure.
        num_clients (int): Number of clients in the network.
        clients (List[MNIST_client]): List of MNIST_client objects.
        num_fed_rounds (int): Number of federated learning rounds.
        model_inertia (float): Model inertia parameter.
        result_path (str): Path to store results.
    
    Methods:
        __init__: Initializes the Network object.
        aggregate_models: Aggregates models of clients based on model inertia and neighbors.
        run: Runs the federated learning process.
        plot_summaries: Plots summaries of accuracy and loss across models.
        plot_label_distribution: Plots the distribution of primary and secondary labels across clients.

    Example:
        network = Network(args)
        network.run()
    """
    
    def __init__(self, args) -> None:
        """
        Initializes the Network object.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """

        # Transforms to be applied on each data sample before being passed to the model (Transforming to tensor and normalizing channel values)
        mnist_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        # Loading the training dataset and validation dataset of MNIST
        self.train_set = torchvision.datasets.MNIST(args.data_path, download=True, train=True, transform=mnist_transforms)
        self.test_set = torchvision.datasets.MNIST(args.data_path, download=True, train=False, transform=mnist_transforms)
        self.val_loader = DataLoader(self.test_set, batch_size=1)

        # Counting the index of the datapoints for each label
        self.label_idxs = [np.array([], dtype=int) for i in range(10)]
        for i, datapoint in enumerate(self.train_set):
            self.label_idxs[datapoint[1]] = np.append(self.label_idxs[datapoint[1]], int(i))

        
        # Initializing the global network
        self.network_graph = nx.random_regular_graph(d=args.neighbors_per_client, n=args.num_clients, seed=args.random_seed)

        # Initializing the clients
        self.num_clients = args.num_clients
        self.clients = []
        for i in range(self.num_clients):
            client_neighbors = list(nx.neighbors(self.network_graph, i))
            self.clients.append(MNIST_client(self.train_set, i, self.label_idxs, args, client_neighbors))

        
        # Initializing the runtime variables
        self.num_fed_rounds = args.num_fed_rounds
        self.model_inertia = args.model_inertia
        self.result_path = setup_result_path(args)
        self.plot_label_distribution()
        

    # Aggregation function
    def aggregate_models(self):
        """
        Aggregates models of clients based on model inertia and neighbors.
        """
        with torch.no_grad():
            for client in self.clients:

                aggregated_model = deepcopy(client.model.state_dict())

                for key in aggregated_model.keys():

                    # Initializing to the client's weights * model_inertia
                    aggregated_model[key] = self.model_inertia * client.model.state_dict()[key]

                    # Aggregating with neighbor weights with an effect of (1-model_inertia)
                    for neighbor_id in client.neighbors:
                        aggregated_model[key] += ((1 - self.model_inertia)/(len(client.neighbors))) * self.clients[neighbor_id].model.state_dict()[key]
                
                # Updating the client's updated model with the aggregated weights
                client.updated_model.load_state_dict(aggregated_model)
    
    
    def run(self):
        """
        Runs the federated learning process.
        """

        for fed_round in range(self.num_fed_rounds):

            for client in self.clients:
                
                # Loading the updated model (aggregated model) into the model that will be used for training
                client.model.load_state_dict(client.updated_model.state_dict())

                # Validating the performance of the aggregated model
                client.validate_model(self.val_loader)
                
                # Training the client for one round
                client.train_one_round()
            
            # Aggregating the client models
            self.aggregate_models()

            print(f"fed round {fed_round} done!")

        # Plotting the summary of the training
        self.plot_summaries()
    

    def plot_summaries(self):
        """
        Plots summaries of accuracy and loss across models.
        """

        overall_acc_mean = []
        overall_loss_mean = []
        overall_acc_std = []
        overall_loss_std = []

        for step in range(self.num_fed_rounds):

            acc_list = []
            loss_list = []

            for client in self.clients:
                acc_list.append(client.history['val_acc'][step])
                loss_list.append(client.history['val_loss'][step])

            overall_acc_mean.append(np.mean(acc_list))
            overall_loss_mean.append(np.mean(loss_list))
            overall_acc_std.append(np.std(acc_list))
            overall_loss_std.append(np.std(loss_list))
        
        overall_acc_mean = np.asarray(overall_acc_mean)
        overall_acc_std = np.asarray(overall_acc_std)
        overall_loss_mean = np.asarray(overall_loss_mean)
        overall_loss_std = np.asarray(overall_loss_std)

        np.savetxt(os.path.join(self.result_path, 'overall_acc_mean.csv'), overall_acc_mean, delimiter=",")
        np.savetxt(os.path.join(self.result_path, 'overall_acc_std.csv'), overall_acc_mean, delimiter=",")
        np.savetxt(os.path.join(self.result_path, 'overall_loss_mean.csv'), overall_acc_mean, delimiter=",")
        np.savetxt(os.path.join(self.result_path, 'overall_loss_std.csv'), overall_acc_mean, delimiter=",")
        
        
        x = [i for i in range(len(overall_acc_mean))]
        plt.figure()
        plt.plot(x, overall_acc_mean)
        plt.fill_between(x, overall_acc_mean-overall_acc_std, overall_acc_mean+overall_acc_std, alpha=0.4)
        plt.xlabel('Global Aggregation Rounds')
        plt.ylabel('Prediction Accuracy (%)')
        plt.title('Mean Accuracy of the Participating Clients in Each Global Aggregation Round')
        plt.savefig(os.path.join(self.result_path, 'acc_result.png'))

        plt.figure()
        plt.plot(x, overall_loss_mean)
        plt.fill_between(x, overall_loss_mean-overall_loss_std, overall_loss_mean+overall_loss_std, alpha=0.4)
        plt.xlabel('Global Aggregation Rounds')
        plt.ylabel('Negative Log Likelihood Loss')
        plt.title('Mean Loss of the Participating Clients in Each Global Aggregation Round')
        plt.savefig(os.path.join(self.result_path, 'loss_result.png'))
    

    def plot_label_distribution(self):
        """
        Plots the distribution of primary and secondary labels across clients.
        """
        
        label_sets = []
        for client in self.clients:
            label_sets.append(client.label_set)
        label_sets = np.asarray(label_sets)
        primary_labels = label_sets[:, 0]
        secondary_labels = label_sets[:, 1]
        primary_stats = np.unique(primary_labels, return_counts=True)
        secondary_stats = np.unique(secondary_labels, return_counts=True)
        draw_bar_plot([primary_stats, secondary_stats], ['Primary Labels', 'Secondary Labels'], ['blue', 'orange'], 'Label Value',
                                                        'Number of Present Labels', 'distribution of labels across existing clients',
                                                        f'label_distribution_{self.num_clients}.png', self.result_path)
        
        
            

            
