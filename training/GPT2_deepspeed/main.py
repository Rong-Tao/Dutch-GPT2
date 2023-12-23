import torch
import deepspeed
from deepspeed import comm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import GPT2TokenizerFast
import os

## Import our own scrips ##

from model import get_model
from dataset import GPT2Dataset
from util import arg, get_optimizer,batch_logger, epoch_logger_saver
from util import criterion, BATCH_SIZE, EPOCH_NUM, TRAIN_VAL_RATIO

## Initialize Distributed Training #####
def init_distributed_mode():
    deepspeed.init_distributed(dist_backend='nccl')
    global_rank = comm.get_global_rank()
    world_size = comm.get_world_size()
    #torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return global_rank, world_size

rank, world_size = init_distributed_mode()

if rank == 0:
    args, result_dir, state_dict_dir, tensor_bd_dir = arg()
else:
    args,_ , _, _ =  arg()

# Training
def train(rank, world_size):
    model_raw = get_model()
    model, optimizer, train_loader, scheduler = deepspeed.initialize(
                                           args = args,
                                           model = model_raw,
                                           model_parameters = model_raw.parameters(),
                                           training_data = GPT2Dataset()
                                        )

    best_loss = float('inf')

    # Initialize SummaryWriter for rank 0
    if rank == 0:
        tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer/bpe-post", max_len=512)
        writer = SummaryWriter(log_dir=tensor_bd_dir)

    for epoch in range(EPOCH_NUM):
        model.train()
        train_loss = 0.0
        for batch_idx, (input_id, mask, target_id) in enumerate(train_loader):
            out = model(input_id, attention_mask=mask, labels=target_id)
            model.backward(out.loss)
            model.step()

            if rank == 0:
                batch_logger(model.module, writer, batch_idx, epoch * len(train_loader) + batch_idx, out.loss.item(), tokenizer)      
                             
            train_loss += out.loss.item()
        
        if rank == 0:
            best_loss = epoch_logger_saver(model, writer, epoch, train_loss/len(train_loader), None, best_loss, state_dict_dir)

    if rank == 0:
        writer.close()

@record
def main():
    train(rank, world_size)

if __name__ == "__main__":
    main()