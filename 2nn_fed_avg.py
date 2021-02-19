""" Fedavg with N number of clients IID"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import copy
import random
from random import randrange
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import pickle
import time

class Client:
    def __init__(self, model, criterion, optimizer, dataset):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        # flatten image input
        x = x.view(-1,784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

# Returns an array with N datasets
def split_to_n_datasets(dataset, n, args):
    dataset_size = len(dataset)//n
    # Seed for reproducible results
    datasets = torch.utils.data.random_split(dataset, [dataset_size for _ in range(n)], generator=torch.Generator().manual_seed(args.seed))
    return datasets

def global_model_to_clients(global_model, clients):
    global_model_sd = global_model.state_dict()
    for client in clients:
        client.model.load_state_dict(global_model_sd)
    return clients


def create_n_clients(n, global_model, datasets, learning_rate, device):
    """
        Returns an array with N Client objects.
    """
    clients = []
    criterion_train = nn.CrossEntropyLoss() #move this
    for i in range(n):
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # the indexes in dataset are the same for client, thus we can do dataset[i].
        client = Client(model, criterion_train, optimizer, datasets[i])
        clients.append(client)
    # Use the same inztialized weights from global model
    global_model_to_clients(global_model, clients)
    return clients

def train_client(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if args.dry_run:
            break

def models_same(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        return True



def update_global_model(sampled_clients, device):
    """
        Update global model by averaging the clients weights and biases.
        CAUTION: this function currently assumes clients got equally amount of data!
    """
    global_model = copy.deepcopy(sampled_clients[0].model)
    global_model_sd = global_model.state_dict()
    nr_clients = len(sampled_clients)

    with torch.no_grad():
        for layer in global_model_sd.keys():
            for client in sampled_clients:
                global_model_sd[layer] += client.model.state_dict()[layer]
            # Average
            global_model_sd[layer] = torch.div(global_model_sd[layer], nr_clients)
    global_model.load_state_dict(global_model_sd)
    return global_model

def compute_metrics(preds, targets):
    preds = preds.cpu()
    targets = targets.cpu()
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
    acc = accuracy_score(targets, preds)
    cr = classification_report(targets, preds, digits=5, output_dict=True)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': cr
    }

def test(model, device, test_loader, criterion, args):
    model.eval()
    test_loss = 0
    correct = 0

    preds = torch.empty(0).to(device)
    targets = torch.empty(0).to(device)           

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds = torch.cat((preds, pred), 0)
            targets = torch.cat((targets, target), 0)
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = test_loss
    
    return metrics

def federated_learning(C):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Federated Learning')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',         ### Changed
        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='learning rate (default: 1.0)')                                    ### Changed
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,        
        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
        help='For Saving the current Model')
    parser.add_argument('--save-metrics', action='store_true', default=True,
        help='For Saving the client metrics')
    args, unknown = parser.parse_known_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #For reproducible results
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    

    ### PARAMETERS
    nr_clients = 100 ### NUMBER OF CLIENTS/ENTITIES
    #C = 100  ### number of clients in each round, [1, nr_clients]
    client_epochs = 1   # Number of client epochs
    communication_rounds = 200 # Number of maximum communication rounds
    test_every_x_round = 1


    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    splitted_datasets = split_to_n_datasets(dataset, nr_clients, args)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    criterion_test = nn.CrossEntropyLoss(reduction='sum')

    global_model = Net().to(device)

    t0 = time.time()

    ### Create n models
    clients = create_n_clients(nr_clients, global_model, splitted_datasets, args.lr, device)

    ### MAIN LOOPS
    metrics_per_test=[]
    for round in range(1, communication_rounds + 1):
        print('Train Round: {}'.format(round))
        sampled_clients = random.sample(clients, C)
        for client in sampled_clients:
            
            for client_epoch in range(1, client_epochs + 1):
                train_loader = torch.utils.data.DataLoader(client.dataset,**train_kwargs)
                train_client(args, client.model, device, train_loader, client.criterion, client.optimizer, client_epoch)

        # Update global model
        global_model = update_global_model(sampled_clients, device)
        clients = global_model_to_clients(global_model, clients)
        if round % test_every_x_round == 0:
            metrics_per_test.append(test(global_model, device, test_loader, criterion_test, args))

        if args.dry_run:
            break

    t1 = time.time()
    total_time = t1-t0
    print("Total experimentation time (s): {}".format(total_time))

    
    dir_str = '{}_{}_{}_{}_{}/'.format(nr_clients, C, client_epochs, communication_rounds, test_every_x_round)
    path = './Results/Fedavg/' + dir_str

    if args.save_model:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), path + "mnist_2nn_fedavg.pt")

    if args.save_metrics:
        metrics = [metrics_per_test, total_time]
        # Directory = (nr_clients)_(C)_(client_epochs)_(communication_rounds)_(test_every_x_round)

        #create dir if nonexistent
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + 'metrics', 'wb') as metrics_file:
            pickle.dump(metrics, metrics_file)

federated_learning(100)

#272