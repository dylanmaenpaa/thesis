""" Fedavg with N number of clients IID"""

import argparse
import torch
import copy
import random
from random import randrange
from torchvision import datasets, transforms
from pathlib import Path
import pickle
import time

# Import common classes and functions
from common import *


def update_global_model(sampled_clients, device):
    """
        Update global model by averaging the clients weights and biases.
    """
    global_model = copy.deepcopy(sampled_clients[0].model)
    global_model_sd = global_model.state_dict()

    total_dataset_len = 0
    for client in sampled_clients:
        total_dataset_len += len(client.dataset)

    first_client_dataset_len = len(sampled_clients[0].dataset)

    with torch.no_grad():
        for layer in global_model_sd.keys():
            
            global_model_sd[layer] = global_model_sd[layer] * (first_client_dataset_len / total_dataset_len)
            for client in sampled_clients[1:]:
                client_dataset_len = len(client.dataset)
                global_model_sd[layer] += client.model.state_dict()[layer] * (client_dataset_len/total_dataset_len)

    global_model.load_state_dict(global_model_sd)
    return global_model




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

def federated_learning():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Federated Learning')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument('--comm-rounds', type=int, default=200, metavar='N',
        help='input batch size for testing (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='learning rate (default: 0.1)')                                    
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,        
        help='quickly check a single pass')
    parser.add_argument('--non-iid', action='store_true', default=False,        
        help='run with non-iid data, default is with iid data')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
        help='For Saving the current Model')
    parser.add_argument('--save-metrics', action='store_true', default=True,
        help='For Saving the client metrics')
    parser.add_argument('--clients', type=int, default=1, 
        help='the number of clients to communicate with during communication round (default 1)')
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
    C = args.clients
    client_epochs = 1   # Number of client epochs
    communication_rounds = args.comm_rounds # Number of maximum communication rounds
    test_every_x_round = 1

    dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    if args.non_iid:
        splitted_datasets = split_to_n_datasets_noniid(dataset, nr_clients)
    else:
        splitted_datasets = split_to_n_datasets(dataset, nr_clients, args)
    
    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    criterion_train = nn.CrossEntropyLoss()

    global_model = Net().to(device)

    t0 = time.time()

    ### Create n models
    clients = create_n_clients(nr_clients, global_model, splitted_datasets, criterion_train, args.lr, device)

    ### MAIN LOOPS
    metrics_per_test=[]
    for round in range(1, communication_rounds + 1):
        print('Train Round: {}'.format(round))
        sampled_clients = random.sample(clients, C)
        for client in sampled_clients:
            
            for client_epoch in range(1, client_epochs + 1):
                train_loader = torch.utils.data.DataLoader(client.dataset,**train_kwargs)
                train_client(args, client.model, device, train_loader, client.criterion, client.optimizer)

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

    # Directory string: (nr_clients)_(C)_(client_epochs)_(communication_rounds)_(test_every_x_round)
    dir_str = '{}_{}_{}_{}_{}/'.format(nr_clients, C, client_epochs, communication_rounds, test_every_x_round)
    if args.non_iid:
        path = './Results/Fedavg_non_iid/' + dir_str
    else:
        path = './Results/Fedavg_iid/' + dir_str

    if args.save_model:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), path + "mnist_2nn_fedavg.pt")

    if args.save_metrics:
        metrics = [metrics_per_test, total_time]
        
        #create dir if nonexistent
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + 'metrics', 'wb') as metrics_file:
            pickle.dump(metrics, metrics_file)

if __name__ == "__main__":
   federated_learning()
