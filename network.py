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
    def __init__(self, original_ds, client_id, label_idx_list, args, neighbors):
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
    
    def __init__(self, args) -> None:
        '''
        Initializing the network parameters given arguments 'args'
        See parse_args.py for further information on the available options.
        '''

        mnist_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        self.train_set = torchvision.datasets.MNIST(args.data_path, download=True, train=True, transform=mnist_transforms)
        self.test_set = torchvision.datasets.MNIST(args.data_path, download=True, train=False, transform=mnist_transforms)
        self.val_loader = DataLoader(self.test_set, batch_size=1)

        self.label_idxs = [np.array([], dtype=int) for i in range(10)]
        for i, datapoint in enumerate(self.train_set):
            self.label_idxs[datapoint[1]] = np.append(self.label_idxs[datapoint[1]], int(i))


        # # Initializing the global model
        # self.global_model = simpleModel()
        # self.global_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, 'initial_model_weights.pth')))
        
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
        

    def aggregate_models(self):
        with torch.no_grad():
            for client in self.clients:
                # if client.client_id == 1:
                #     print(f"Client {client.client_id}")
                aggregated_model = deepcopy(client.model.state_dict())
                for key in aggregated_model.keys():
                    # if client.client_id == 1:
                    #     print(f"key value: {aggregated_model[key][0]}")
                    aggregated_model[key] = self.model_inertia * client.model.state_dict()[key]
                    for neighbor_id in client.neighbors:
                        # if client.client_id == 1:
                        #     print(f"neighbor: {neighbor_id}")
                        #     print(f"key value: {self.clients[neighbor_id].model.state_dict()[key][0]}")
                        aggregated_model[key] += ((1 - self.model_inertia)/(len(client.neighbors))) * self.clients[neighbor_id].model.state_dict()[key]
                
                client.updated_model.load_state_dict(aggregated_model)
    
    
    def run(self):
        for fed_round in range(self.num_fed_rounds):
            
            # print(f"fed round {fed_round}")

            for client in self.clients:
                # print(f"training client {client.client_id}")
                client.model.load_state_dict(client.updated_model.state_dict())
                # print(f"aggregated model copied")
                client.validate_model(self.val_loader)
                # print(f"validation accuracy gathered: {client.history['val_acc'][-1]}")
                client.train_one_round()
                # print(f"client trained")
            
            # print("aggregating models")
            self.aggregate_models()
            print(f"fed round {fed_round} done!")

        # print("plotting summaries")
        self.plot_summaries()
    

    def plot_summaries(self):

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
        label_sets = []
        for client in self.clients:
            label_sets.append(client.label_set)
            # print(len(client.dataset))
        label_sets = np.asarray(label_sets)
        primary_labels = label_sets[:, 0]
        secondary_labels = label_sets[:, 1]
        primary_stats = np.unique(primary_labels, return_counts=True)
        secondary_stats = np.unique(secondary_labels, return_counts=True)
        draw_bar_plot([primary_stats, secondary_stats], ['Primary Labels', 'Secondary Labels'], ['blue', 'orange'], 'Label Value',
                                                        'Number of Present Labels', 'distribution of labels across existing clients',
                                                        f'label_distribution_{self.num_clients}.png')
        
        
            

            
