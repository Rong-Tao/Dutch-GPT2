import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import os

## Import our own scrips ##

from model import get_model
from dataset import get_loaders
from util import arg, get_optimizer,batch_logger, epoch_logger_saver
from util import criterion, BATCH_SIZE, EPOCH_NUM, TRAIN_VAL_RATIO

## Initialize Distributed Training #####
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return global_rank, world_size

rank, world_size = init_distributed_mode()

if rank == 0:
    result_dir, state_dict_dir, tensor_bd_dir = arg()

# Training
def train(rank, world_size):
    model = get_model.cuda()
    model = DDP(model)
    optimizer, scheduler = get_optimizer(model)
    train_loader, validation_loader = get_loaders(world_size, rank, BATCH_SIZE, TRAIN_VAL_RATIO)

    best_loss = float('inf')

    # Initialize SummaryWriter for rank 0
    if rank == 0:
        writer = SummaryWriter(log_dir=tensor_bd_dir)

    for epoch in range(EPOCH_NUM):
        model.train()
        train_loss = 0.0
        for batch_idx, (input_id, mask, target_id) in enumerate(train_loader):
            input_id, mask, target_id = input_id.cuda(), mask.cuda(), target_id.cuda()
            out = model(input_id, attention_mask=mask)
            logits = out.logits.view(-1, out.logits.size(-1))
            loss = criterion(logits, target_id.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if rank == 0:
                batch_logger(writer, batch_idx, epoch * len(train_loader) + batch_idx, loss.item())

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, (val_input_id, val_mask, val_target_id) in enumerate(validation_loader):
                val_input_id, val_mask, val_target_id = val_input_id.cuda(), val_mask.cuda(), val_target_id.cuda()
                val_outputs = model(val_input_id, attention_mask=val_mask)
                val_logits = val_outputs.logits.view(-1, val_outputs.logits.size(-1))
                val_loss = criterion(val_logits, val_target_id.view(-1))
                validation_loss += val_loss.item()

        validation_loss /= len(validation_loader)
        scheduler.step(validation_loss)
        
        if rank == 0:
            best_loss = epoch_logger_saver(model, writer, epoch, train_loss/len(train_loader), validation_loss, best_loss, state_dict_dir)

    if rank == 0:
        writer.close()

@record
def main():
    train(rank, world_size)

if __name__ == "__main__":
    main()
