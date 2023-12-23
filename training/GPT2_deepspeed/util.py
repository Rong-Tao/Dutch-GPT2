import os
import deepspeed
import argparse
import torch
import torch.optim as optim

BATCH_SIZE = 20
EPOCH_NUM = 50
TRAIN_VAL_RATIO = 0.95 # 0-1

criterion = torch.nn.CrossEntropyLoss(ignore_index=1)

def get_optimizer(model):
    optimizer = optim.AdamW(model.module.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_NUM)

    return optimizer, scheduler

def batch_logger(model, writer, batch_idx, step_num, loss, tokenizer):
    writer.add_scalar('Batch Training Loss', loss, step_num)
    generate_text(step_num, model, writer, tokenizer, max_length=100)



def generate_text(step_num, model, writer, tokenizer, max_length=100):
    model.eval()

    input_ids = tokenizer.encode("<s>Kerstmis komt dichterbij en mijn lieve vrienden in Eindhoven", return_tensors="pt").to(model.device)

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=1,
        eos_token_id=2
    )

    if output_sequences.is_cuda:
        output_sequences = output_sequences.cpu()

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    writer.add_text('Generated Text', generated_text, step_num)



def epoch_logger_saver(model, writer, epoch, mean_trainloss, validation_loss, best_loss, state_dict_dir):
    writer.add_scalar('Epoch Training Loss', mean_trainloss, epoch)
    writer.add_scalar('Epoch Validation Loss', validation_loss, epoch)
    if mean_trainloss < best_loss:
        best_loss = mean_trainloss
        model_save_path = os.path.join(state_dict_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
    return best_loss

def arg():
    parser = argparse.ArgumentParser(description='Pass log directories to main script.') 
    parser.add_argument('--output_log', type=str, help='Path to the output log directory.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    result_dir = args.output_log 
    tensor_bd_dir = os.path.join(result_dir, 'tensorboard')
    state_dict_dir = os.path.join(result_dir, 'state_dict') 
    os.makedirs(result_dir, exist_ok=True) 
    os.makedirs(tensor_bd_dir, exist_ok=True)
    os.makedirs(state_dict_dir, exist_ok=True)
    return args, result_dir, state_dict_dir, tensor_bd_dir
