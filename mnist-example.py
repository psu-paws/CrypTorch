import torch
import torchvision

from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cryptorch.pass_manager import TunerConfig
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(F.avg_pool2d(self.conv1(x), 2))
        x = self.act(F.avg_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.act(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

batch_size_test = 250

def main():
    n_epochs = 190
    batch_size_train = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    use_gpu = False # change to true to run on GPU
    world_size = 1 # change to 2 to run simulated MPC
    num_test_samples = 1000 # change how many sample out of the test set to use.
    
    # optional: load configuration file
    """
    from cryptorch.system_params import load_config
    load_config("config.yml")
    """

    random_seed = 1
    torch.manual_seed(random_seed)
    train_dataset = torchvision.datasets.MNIST(
        'data/MNIST/', train=True, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ]))
    
    search_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [0.1, 0.9])
    
    print(f"{len(search_dataset)=}")
    print(f"{len(train_dataset)=}")
    
    search_loader = torch.utils.data.DataLoader(
        search_dataset,
        batch_size=batch_size_test
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train, shuffle=True
    )

    test_dataset = torchvision.datasets.MNIST('data/MNIST/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    
    indices = torch.randperm(len(test_dataset))[:num_test_samples]
    
    test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"{len(test_dataset)=}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False)
    
    _, (examples, _) = next(enumerate(test_loader))
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    
    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        if epoch % 10 == 0:
            torch.save(network.state_dict(), f'model-{epoch}.pth')
            torch.save(optimizer.state_dict(), f'optimizer-{epoch}.pth')
    
    def test(network):
        # network.eval()
        test_loss = 0
        correct = 0
        loss_func = CrossEntropyLoss()
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += loss_func(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    #########################
    # training
    """
    test(network)
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test(network)
    """
    ###########################
    # OR load trained model
    
    weights = torch.load("model-190.pth", weights_only=True)
    network.load_state_dict(weights)
    network = network.eval()
    test(network)
    
    ## export network
    
    # Get export IR
    exported_program = torch.export.export(network, args=(examples,))
    
    # Get module
    module = exported_program.module()
    
    # Merge batchnorm
    from cryptorch.helpers import fuse
    fuse(module)
    
    print("=========================== Original model ===============================")
    print(module)
    
    # Test 1: To test the original module, uncomment below.
    """
    test(module)
    return
    """
    
    # sets secrets
    from cryptorch.pass_manager import CrypTorchPassManager, set_secret, get_all_param_names
    # Input x is a secret and owned by party 0
    secret_tensors = {"x": [0]}
    # If you uncomment below, all the params becomes a secret owned by party 1
    """
    for p in get_all_param_names(module):
       secret_tensors[p] = [1 % world_size]
    """
    
    set_secret(
        module, 
        secret_tensors=secret_tensors,
        output_owner=0,
        num_parties=world_size
    )

    # tuning
    from cryptorch.passes import rewriting_passes, TunableGeluPass, GeluPass
    
    # example pass
    """
    from cryptorch.passes import BasePass
    class OurGeluPass(BasePass):

        def get_patterns(self):
            return [lambda x: torch.ops.aten.gelu.default(x)]

        def get_replacement(self):
            return self.Replacement()
        
        class Replacement(nn.Module):
            def forward(self, x):
                a0 = -0.03161286950386981
                a1 = 0.2597658446632754
                a2 = 0.11594076368157785

                pos_x = x.abs()
                relu_x = (pos_x + x) / 2
                cond = pos_x <= 2.2
                
                gelu_p0 = pos_x * pos_x * a2 + pos_x * a1 + a0 + 0.5 * x

                return torch.where(cond, gelu_p0, relu_x)
    """
    
    # rewrite only
    pm = CrypTorchPassManager(module,
            rewriting_passes + [
                GeluPass("bolt3"),
                # OurGeluPass(),
            ]
    )
    
    # Autotuning
    """
    pm = CrypTorchPassManager(module, 
            rewriting_passes + [
                TunableGeluPass(),
            ],
            tuner_config=TunerConfig(
                tuner="greedy_binary",
                search_inputs=search_loader,
                input_preprocess_func=lambda entry: ((entry[0], ), entry[1]),
                objective_func=(CrossEntropyLoss()),
                objective_threshold=0.05
            )
        )
    """
    
    print("Passes in use:")
    pm.print_passes()
    
    module = pm.run()

    print("=========================== Approximated model ===============================")
    print(module)

    # Test 2: To test the approximated module, uncomment below.
    """
    test(module)
    return
    """
    
    # Test 3: Tune hummingbird
    """
    from cryptorch.hummingbird_tuning import hummingbird_tuning
    hummingbird_tuning(module, search_loader, lambda entry: (entry[0],))
    """
    
    from cryptorch.utils import FakeDataset, launch_mpc_processes
    
    launch_mpc_processes(module, run_mpc, world_size=world_size, additional_arguments=[(test_dataset, use_gpu) if rank == 0 else (FakeDataset(test_dataset), use_gpu) for rank in range(world_size)])
    
def run_mpc(module, rank, world_size, dataset, use_gpu):
    from cryptorch.mpc_runtime.runtime import init_runtime, get_comm_stats
    from cryptorch.mpc_runtime.cryptenpp_runtime import CrypTenPPRuntime
    from cryptorch.lowering import lower
    import contextlib
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_test, shuffle=False)
    

    init_runtime(CrypTenPPRuntime(), rank)
    lower(module, rank=rank)

    if rank == 0:
        print("=========================== Lowered model ===============================")
        print(module)
    
    if use_gpu:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"
        
    module.to(device)
        
    test_loss = 0
    correct = 0
    batches = 0
    samples = 0

    loss_func = CrossEntropyLoss()

    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        if use_gpu:
            # enables custom int kernels for GPU
            # these kernels implement operations such as matmul on the GPU for integers when torch only have float kernels.
            from cryptorch.helpers import enable_custom_int_kernels
            stack.enter_context(enable_custom_int_kernels())
        for data, target in tqdm(dataloader, total=len(dataloader)) if rank == 0 else dataloader:
            data = data.to(device)
            target = target.to(device)

            output = module(data)
            #output = intp.run(data)
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            batches += 1
            samples += data.shape[0]
            test_loss /= samples
            
        if rank == 0:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(dataloader.dataset),
                100. * correct / len(dataloader.dataset))
            )
                
        stats = get_comm_stats()
        print(f'RANK {rank}: Comm Stats: rounds {stats["rounds"]}, bytes {stats["bytes"]}, time {stats["time"]}')
        print(f'RANK {rank}: Comm Stats: rounds per batch {stats["rounds"] / batches}, bytes per sample {stats["bytes"] / samples}')
            
    
if __name__ == "__main__":
    main()

