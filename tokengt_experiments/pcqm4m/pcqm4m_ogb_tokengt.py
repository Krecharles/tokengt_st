# Code adapted from OGB.
# https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m-v2
# and from
# https://github.com/pyg-team/pytorch_geometric/blob/b8c0d82d3e8a66063a9fe33ec31c8bb654c1fdc3/examples/multi_gpu/pcqm4m_ogb.py#L4

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.transforms.add_orthornormal_node_identifiers import AddOrthonormalNodeIdentifiers
from tqdm.auto import tqdm

from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.io import fs
import wandb
import torch.optim.lr_scheduler as lr_scheduler

from ogb.lsc import PCQM4Mv2Evaluator, PygPCQM4Mv2Dataset

from ogb.utils import smiles2graph

from tokengt_experiments.pcqm4m.pcqm4m_models import GATGraphRegression, TokenGTGraphRegression, GCNGraphRegression


def ogb_from_smiles_wrapper(smiles, *args, **kwargs):
    """Returns `torch_geometric.data.Data` object from smiles while
    `ogb.utils.smiles2graph` returns a dict of np arrays.
    """
    data_dict = smiles2graph(smiles, *args, **kwargs)
    return Data(
        x=torch.from_numpy(data_dict['node_feat']),
        edge_index=torch.from_numpy(data_dict['edge_index']),
        edge_attr=torch.from_numpy(data_dict['edge_feat']),
        smiles=smiles,
    )


def train(model, rank, device, loader, optimizer):
    model.train()
    reg_criterion = torch.nn.L1Loss()
    loss_accum = 0.0

    for step, batch in enumerate(
            tqdm(loader, desc="Training", disable=(rank > 0))):
        batch = batch.to(device)
        pred = model(batch).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []
    for batch in tqdm(loader, desc="Testing"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred


def run(rank, dataset, args):

    wandb.init(
        entity="krecharles-university-of-oxford",
        project="PCQM4M_TokenGT",
        config=vars(args),
        # mode="disabled"
    )

    num_devices = args.num_devices
    device = torch.device(
        "cuda:" + str(rank)) if num_devices > 0 else torch.device("cpu")

    if num_devices > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=num_devices)

    if args.on_disk_dataset:
        train_idx = torch.arange(len(dataset.indices()))
    else:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]

    if num_devices > 1:
        num_splits = math.ceil(train_idx.size(0) / num_devices)
        train_idx = train_idx.split(num_splits)[rank]

    train_dataset = dataset[train_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if rank == 0:
        transform = AddOrthonormalNodeIdentifiers(args.D_P, args.use_laplacian) 
        root_f = f'data/pcqm4m_{args.D_P}_{"lap" if args.use_laplacian else "ort"}'
        if args.on_disk_dataset:
            valid_dataset = PCQM4Mv2(root=root_f, split="val",
                                     from_smiles=ogb_from_smiles_wrapper,
                                     transform=transform)
            test_dev_dataset = PCQM4Mv2(
                root=root_f, split="test",
                from_smiles=ogb_from_smiles_wrapper,
                transform=transform)
            test_challenge_dataset = PCQM4Mv2(
                root=root_f, split="holdout",
                from_smiles=ogb_from_smiles_wrapper,
                transform=transform)
            valid_dataset = valid_dataset[:int(max(1024, len(valid_dataset)*args.dataset_fraction))]
            test_dev_dataset = test_dev_dataset
            test_challenge_dataset = test_challenge_dataset
        else:
            valid_dataset = dataset[split_idx["valid"]]
            test_dev_dataset = dataset[split_idx["test-dev"]]
            test_challenge_dataset = dataset[split_idx["test-challenge"]]

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.save_test_dir != '':
            testdev_loader = DataLoader(
                test_dev_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            testchallenge_loader = DataLoader(
                test_challenge_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

        if args.checkpoint_dir != '':
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        evaluator = PCQM4Mv2Evaluator()

    if args.model == 'token_gt':
        model = TokenGTGraphRegression(
        d_p=args.D_P,
        d=args.head_dim*args.num_heads,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        include_graph_token=args.include_graph_token,
        is_laplacian_node_ids=args.use_laplacian,
        dropout=args.dropout_ratio,
        device=device,
    )
    elif args.model == 'gcn':
        model = GCNGraphRegression(
            hidden_channels=args.hidden_channels, 
            batch_norm=True,
            num_layers=args.num_encoder_layers,
            dropout=args.dropout_ratio,
            device=device,
        )
    elif args.model == 'gat':
        model = GATGraphRegression(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_encoder_layers,
            heads=args.num_heads,
            dropout=args.dropout_ratio,
            device=device,
        )

    print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.log({"number of parameters": sum(p.numel() for p in model.parameters())})

    if num_devices > 0:
        model = model.to(rank)
    if num_devices > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_mae = 1000
    # Scheduler: linear warmup to args.lr, then linear decay to 0
    scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=args.warmup_epochs)
    scheduler2 = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.0, total_iters=args.epochs-args.warmup_epochs)

    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.warmup_epochs])

    current_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if os.path.isfile(checkpoint_path):
        checkpoint = fs.torch_load(checkpoint_path)
        current_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_valid_mae = checkpoint['best_val_mae']
        print(f"Found checkpoint, resume training at epoch {current_epoch}")

    for epoch in range(current_epoch, args.epochs + 1):
        train_mae = train(model, rank, device, train_loader, optimizer)

        if num_devices > 1:
            dist.barrier()

        if rank == 0:
            valid_mae = eval(
                model.module if isinstance(model, DistributedDataParallel) else
                model, device, valid_loader, evaluator)

            print(f"Epoch {epoch:02d}, "
                  f"Train MAE: {train_mae:.4f}, "
                  f"Val MAE: {valid_mae:.4f}")

            wandb.log({
                "train_mae": train_mae,
                "valid_mae": valid_mae,
            }, step=epoch)

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                if args.checkpoint_dir != '':
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_mae': best_valid_mae,
                    }
                    torch.save(checkpoint, checkpoint_path)

                if args.save_test_dir != '':
                    test_model = model.module if isinstance(
                        model, DistributedDataParallel) else model

                    testdev_pred = test(test_model, device, testdev_loader)

                    evaluator.save_test_submission(
                        {'y_pred': testdev_pred.cpu().detach().numpy()},
                        args.save_test_dir,
                        mode='test-dev',
                    )

                    testchallenge_pred = test(test_model, device,
                                              testchallenge_loader)
                    evaluator.save_test_submission(
                        {'y_pred': testchallenge_pred.cpu().detach().numpy()},
                        args.save_test_dir,
                        mode='test-challenge',
                    )

            print(f'Best validation MAE so far: {best_valid_mae}')

        if rank == 0:
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=epoch)
            scheduler.step()
        if num_devices > 1:
            dist.barrier()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TokenGT baselines on pcqm4m',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gat', 'token_gt'])
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of epochs to warmup the learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='',
                        help='directory to save test submission file')
    parser.add_argument('--num_devices', type=int, default='1',
                        help="Number of GPUs, if 0 runs on the CPU")
    parser.add_argument('--on_disk_dataset', action='store_true')

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--use_laplacian', action='store_true')
    parser.add_argument('--D_P', type=int, default=64,
                        help='Positional encoding dimension (e.g., 16 for LapPE, 64 for ORF)')
    parser.add_argument('--head_dim', type=int, default=24,
                        help='Dimension of each attention head')
    parser.add_argument('--num_heads', type=int, default=32,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int,
                        default=12, help='Number of transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=768,
                        help='Dimension of the feedforward network')
    parser.add_argument('--dropout_ratio', type=float, default=0.1)
    parser.add_argument('--include_graph_token', action='store_true')
    parser.add_argument('--dataset_fraction', type=float, default=1)
    parser.add_argument('--hidden_channels', type=int, default=32,
                        help='Number of hidden channels for GCN')

    args = parser.parse_args()

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available(
    ) else 0
    if args.num_devices > available_gpus:
        if available_gpus == 0:
            print("No GPUs available, running w/ CPU...")
        else:
            raise ValueError(f"Cannot train with {args.num_devices} GPUs: "
                             f"available GPUs count {available_gpus}")

    # automatic dataloading and splitting
    transform = AddOrthonormalNodeIdentifiers(
        args.D_P, args.use_laplacian)
    if args.on_disk_dataset:
        root_f = f'data/pcqm4m_{args.D_P}_{"lap" if args.use_laplacian else "ort"}'
        print(f"Saving to {root_f}")
        dataset = PCQM4Mv2(root=root_f, split='train',
                           from_smiles=ogb_from_smiles_wrapper,
                           transform=transform)
    else:
        dataset = PygPCQM4Mv2Dataset(root='data/', transform=transform)

    if args.dataset_fraction < 1:
        dataset = dataset.shuffle()[:int(len(dataset)*args.dataset_fraction)]
        print(f"dataset size: {len(dataset)}")

    if args.num_devices > 1:
        mp.spawn(run, args=(dataset, args), nprocs=args.num_devices, join=True)
    else:
        run(0, dataset, args)
