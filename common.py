"""
    Common classes and functions for the different experiments
"""

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report

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

def models_same(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

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

def global_model_to_clients(global_model, clients):
    global_model_sd = global_model.state_dict()
    for client in clients:
        client.model.load_state_dict(global_model_sd)
    return clients

def create_n_clients(n, global_model, datasets, criterion, learning_rate, device):
    """
        Returns an array with N Client objects.
    """
    clients = []
    for i in range(n):
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # the indexes in dataset are the same for client, thus we can do dataset[i].
        client = Client(model, criterion, optimizer, datasets[i])
        clients.append(client)
    # Use the same inztialized weights from global model
    global_model_to_clients(global_model, clients)
    return clients

def train_client(args, model, device, train_loader, criterion, optimizer):
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

# Returns an array with N datasets
def split_to_n_datasets(dataset, n, args):
    dataset_size = len(dataset)//n
    # Seed for reproducible results
    datasets = torch.utils.data.random_split(dataset, [dataset_size for _ in range(n)], generator=torch.Generator().manual_seed(args.seed))
    return datasets

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')