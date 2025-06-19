# Code adapted from OGB.
# https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m-v2
import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms.add_orthornormal_node_identifiers import AddOrthonormalNodeIdentifiers
from tqdm.auto import tqdm

from torch_geometric.data import Data
from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.io import fs
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.token_gt import TokenGT
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import degree
import wandb
import torch.optim.lr_scheduler as lr_scheduler

try:
    from ogb.lsc import PCQM4Mv2Evaluator, PygPCQM4Mv2Dataset
except ImportError as e:
    raise ImportError(
        "`PygPCQM4Mv2Dataset` requires rdkit (`pip install rdkit`)") from e

from ogb.utils import smiles2graph


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


class TokenGTGraphRegression(torch.nn.Module):
    def __init__(
        self,
        dim_node,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        is_laplacian_node_ids,
        dim_edge,
        dropout,
        device,
    ):
        super().__init__()
        self._token_gt = TokenGT(
            dim_node=dim_node,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dim_edge=dim_edge,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=include_graph_token,
            dropout=dropout,
            device=device,
        )
        self.lm = torch.nn.Linear(d, 1, device=device)

    def forward(self, batch):
        _, graph_emb = self._token_gt(batch.x.float(),
                                      batch.edge_index,
                                      batch.edge_attr.float(),
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids)
        return self.lm(graph_emb)


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
        sync_tensorboard=True,
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
        root_f = f'/mnt/data/pcqm4m_{args.D_P}_{"lap" if args.use_laplacian else "ort"}'
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
        else:
            valid_dataset = dataset[split_idx["valid"]]
            test_dev_dataset = dataset[split_idx["test-dev"]]
            # test_challenge_dataset = dataset[split_idx["test-challenge"]]

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

    model = TokenGTGraphRegression(
        dim_node=train_dataset.num_node_features,
        d_p=args.D_P,
        d=args.head_dim*args.num_heads,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        include_graph_token=args.include_graph_token,
        is_laplacian_node_ids=args.use_laplacian,
        dim_edge=train_dataset.num_edge_features,
        dropout=args.dropout_ratio,
        device=device,
    )

    print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.log({"number of parameters": sum(p.numel() for p in model.parameters())})

    if num_devices > 0:
        model = model.to(rank)
    if num_devices > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-8, weight_decay=args.weight_decay)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000
    # Scheduler: linear warmup to 0.0002, then linear decay to 0
    scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=0.000001, end_factor=args.lr, total_iters=args.warmup_iterations)
    scheduler2 = lr_scheduler.LinearLR(optimizer, start_factor=args.lr, end_factor=0.0, total_iters=args.iterations-args.warmup_iterations)

    # TODO replace 0 with args.warmup_iterations
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[0])

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

            if args.log_dir != '':
                writer.add_scalar('valid/mae', valid_mae, epoch)
                writer.add_scalar('train/mae', train_mae, epoch)

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

        if num_devices > 1:
            dist.barrier()

        if rank == 0:
            scheduler.step()
        if num_devices > 1:
            dist.barrier()
        
        if rank == 0 and args.log_dir != '':
            writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TokenGT baselines on pcqm4m',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('--model', type=str, default='token_gt',
                        choices=['sage', 'gat', 'token_gt'])
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='',
                        help='directory to save test submission file')
    parser.add_argument('--num_devices', type=int, default='1',
                        help="Number of GPUs, if 0 runs on the CPU")
    parser.add_argument('--on_disk_dataset', action='store_true')

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup_iterations', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--use_laplacian', action='store_true')
    parser.add_argument('--D_P', type=int, default=16,
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
        root_f = f'/mnt/data/pcqm4m_{args.D_P}_{"lap" if args.use_laplacian else "ort"}'
        dataset = PCQM4Mv2(root=root_f, split='train',
                           from_smiles=ogb_from_smiles_wrapper,
                           transform=transform)
    else:
        dataset = PygPCQM4Mv2Dataset(root='/mnt/data/', transform=transform)

    # TODO: remove this
    dataset = dataset.shuffle()[:int(len(dataset)*0.02)]
    print(f"dataset size: {len(dataset)}")

    if args.num_devices > 1:
        mp.spawn(run, args=(dataset, args), nprocs=args.num_devices, join=True)
    else:
        run(0, dataset, args)
