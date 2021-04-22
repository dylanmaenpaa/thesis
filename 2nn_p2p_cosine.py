import argparse
import torch
import copy
import random
from random import randrange
from torchvision import datasets, transforms
from pathlib import Path
from numpy import dot
from numpy.linalg import norm
import pickle
import time

# Import common classes and functions
from common import *
    
def calc_avg_clients_metrics(clients_metrics):
    avg_acc = 0
    avg_prec = 0
    avg_rec = 0
    avg_fscore = 0
    avg_loss = 0

    for client_metrics in clients_metrics:
        avg_acc += client_metrics["accuracy"]
        avg_prec += client_metrics["precision"]
        avg_rec += client_metrics["recall"]
        avg_fscore += client_metrics["f1"]
        avg_loss += client_metrics["loss"]
    
    nr_clients = len(clients_metrics)
    avg_acc /= nr_clients
    avg_prec /= nr_clients
    avg_rec /= nr_clients
    avg_fscore /= nr_clients
    avg_loss /= nr_clients

    return {
        'average_accuracy': avg_acc,
        'average_precision': avg_prec,
        'average_recall': avg_rec,
        'average_fscore': avg_fscore,
        'average_loss': avg_loss,
    }

def get_fscores(class_report):
    digs = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
    fscore_array = []
    for dig in digs:
        fscore_array.append(class_report[dig]['f1-score'])
    return fscore_array

def fscore_cosine_similarity_neighbors(client_idx, clients_metrics):
    client_fscores = get_fscores(clients_metrics[client_idx]['classification_report'])
    diffs = []
    
    for idx, neighbor_metrics in enumerate(clients_metrics):
        if idx != client_idx:
            neighbor_fscore = get_fscores(neighbor_metrics['classification_report'])
            cosine_similarity = dot(client_fscores, neighbor_fscore) / (norm(client_fscores) * norm(neighbor_fscore))
            diffs.append((idx, cosine_similarity))
    # Shuffle before sort to avoid the initial neighbors in the array to be chosen
    # more often
    random.shuffle(diffs)
    diffs = sorted(diffs, key=lambda x: x[1])
    if client_idx == 0:
        print(diffs)
    idxs = [diff[0] for diff in diffs]
    return idxs



def update_models(clients, graph, clients_metrics, device, C):
    """
        Update client models by aggregating and averaging all clients models with neighbors.
    """
    new_clients = copy.deepcopy(clients)

    # Choose C neighbors that have communicated most different!
    neighbors_diffs = []
    for client_idx, client in enumerate(new_clients):
        neighbors_diffs.append(fscore_cosine_similarity_neighbors(client_idx, clients_metrics))

    for client_idx, client in enumerate(new_clients):
        total_dataset_len = len(client.dataset)
        client_model_sd = client.model.state_dict()

        #neighbor_idxs = random.sample(graph[client_idx], C)
        neighbor_idxs = neighbors_diffs[client_idx][:C]
        # Calc total dataset length for the client and neighbors
        for neighbor_idx in neighbor_idxs:
            neighbor = clients[neighbor_idx]
            client_dataset_len = len(neighbor.dataset)
            total_dataset_len += client_dataset_len

        client_dataset_len = len(client.dataset)
        for layer in client_model_sd.keys():
            
            client_model_sd[layer] = client_model_sd[layer] * (client_dataset_len / total_dataset_len) 
            for neighbor_idx in neighbor_idxs:
                neighbor = clients[neighbor_idx]
                neighbor_dataset_len = len(neighbor.dataset)
                client_model_sd[layer] += neighbor.model.state_dict()[layer]*(neighbor_dataset_len/total_dataset_len)
                #print(client_model_sd[layer])
            # Average
        client.model.load_state_dict(client_model_sd)
    return new_clients

# Run test set on every client 
# Returns: each clients metrics and averages over all clients
def test(clients, device, test_loader, criterion, args):
    clients_metrics = []
    for i, client in enumerate(clients):
        model = client.model
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
                preds = torch.cat((preds, pred), 0)
                targets = torch.cat((targets, target), 0)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if args.dry_run:
                    break

        test_loss /= len(test_loader.dataset)

        metrics = compute_metrics(preds, targets)
        metrics["loss"] = test_loss
        clients_metrics.append(metrics)
    
    return clients_metrics

def decentralized_learning():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Federated Learning')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument('--comm-rounds', type=int, default=200, metavar='N',
        help='input batch size for testing (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='learning rate (default: 1.0)')                                    ### Changed
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,        
        help='quickly check a single pass')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
        help='local client epochs (default 1')        
    parser.add_argument('--non-iid', action='store_true', default=False,        
        help='run with non-iid data, default is with iid data')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--save-models', action='store_true', default=True,
        help='For Saving all client models')
    parser.add_argument('--save-metrics', action='store_true', default=True,
        help='For Saving the client metrics')
    parser.add_argument('--neighbor-clients', type=int, default=1, 
        help='the number of neighboring clients to communicate with (default 1)')
    args, unknown = parser.parse_known_args()

    ### PARAMETERS
    nr_clients = 100 # number of clients/entities
    C = args.neighbor_clients  # number of neighbors to communicate with in each round for each clients. range: [0, max_nr_neighbours]
    client_epochs = args.epochs   # Number of client epochs
    communication_rounds = args.comm_rounds  # Number of maximum communication rounds
    test_every_x_round = 10

    graph = [] # A list showing connections, e.g. for client 0 neighbors are at index 0. Unidirectional.
    for i in range(nr_clients):
        neighbors = []
        for j in range(nr_clients):
            if j != i:
                neighbors.append(j)
        graph.append(neighbors)


    nghbrs_idx_comm = [[] for _ in range(nr_clients)] # A list containing the idxs whom each client has communicated with. E.g. client 0 neighbors are at index 0.


    #For reproducible results
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
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

    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    _, test_set = torch.utils.data.random_split(dataset2, [9000, 1000])
    test_loader_mini = torch.utils.data.DataLoader(test_set, **test_kwargs)

    if args.non_iid:
        splitted_datasets = split_to_n_datasets_noniid(dataset, nr_clients)
    else:
        splitted_datasets = split_to_n_datasets(dataset, nr_clients, args)

    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    criterion_train = nn.CrossEntropyLoss()

    # All models are initialized with the same weights at start from a "global_model"
    global_model = Net().to(device)

    ### Create n models
    clients = create_n_clients(nr_clients, global_model, splitted_datasets, criterion_train, args.lr, device)

    clients_metrics_per_test = []
    
    # init
    clients_metrics = test(clients, device, test_loader_mini, criterion_test, args)

    ### MAIN LOOPS
    t0 = time.time()
    for round in range(1, communication_rounds+1):
        print('Train Round: {}'.format(round))
        for client in clients:
            for client_epoch in range(1, client_epochs + 1):
                train_loader = torch.utils.data.DataLoader(client.dataset,**train_kwargs)
                train_client(args, client.model, device, train_loader, client.criterion, client.optimizer)
        # Only test every xth communication round due to computationally heavy (time consuming).
        
        # Update clients model
        clients = update_models(clients, graph, clients_metrics, device, C)

        if round % test_every_x_round == 0:
            clients_metrics = test(clients, device, test_loader, criterion_test, args)
            print(calc_avg_clients_metrics(clients_metrics))
            clients_metrics_per_test.append(clients_metrics)
        else:
            clients_metrics = test(clients, device, test_loader_mini, criterion_test, args)
            print(calc_avg_clients_metrics(clients_metrics))

        if args.dry_run:
            break
    

    t1 = time.time()
    total_time = t1-t0
    print("Total experimentation time (s): {}".format(total_time))

    # Directory string: (nr_clients)_(C)_(client_epochs)_(communication_rounds)_(test_every_x_round)
    dir_str = '{}_{}_{}_{}_{}/'.format(nr_clients, C, client_epochs, communication_rounds, test_every_x_round)

    if args.non_iid:
        print(dir_str)
        path = './Results/P2P_cosine_non_iid/' + dir_str
    else:
        path ='./Results/P2P_cosine_iid/' + dir_str

    if args.save_models:
        #create dirs if nonexistent
        Path(path).mkdir(parents=True, exist_ok=True)
        path_to_models = path + 'models'
        Path(path_to_models).mkdir(parents=True, exist_ok=True)

        for i, client in enumerate(clients):
            torch.save(client.model.state_dict(), "{}/mnist_2nn_client{}.pt".format(path_to_models, i))

    if args.save_metrics:
        metrics = [clients_metrics_per_test, graph, total_time]
        # Directory = (nr_clients)_(C)_(client_epochs)_(communication_rounds)_(test_every_x_round)

        #create dir if nonexistent
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + 'metrics', 'wb') as metrics_file:
            pickle.dump(metrics, metrics_file)


if __name__ == "__main__":
    decentralized_learning()
