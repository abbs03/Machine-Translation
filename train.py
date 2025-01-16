from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


from tokenizer import Tokenizer
from dataset import BilingiualDataset, causal_mask, special_labels
from transformer import Transformer
from config import get_config


def build_tokenizer(lang, text, path, config):
    tokenizer = Tokenizer(lang=lang)
    if path is None:
        tokenizer.train(text, vocab_size= config[f'{lang}_vocab_size'])    
        tokenizer.save_params('parameters/')
    else:
        tokenizer.load_params(path)
        
    tokenizer.register_special_tokens({
            '<|SOS|>': len(tokenizer.vocab) ,
            '<|EOS|>': len(tokenizer.vocab) + 1 ,
            '<|PAD|>': len(tokenizer.vocab) + 2
        })
    
    return tokenizer

def read_text_path(path):
    with open(path, 'r') as file:
        text = file.readlines()
    return text

def get_ds(config):
    
    src_train_text = read_text_path(config['src_train'])
    tgt_train_text = read_text_path(config['tgt_train'])
    
    src_val_text = read_text_path(config['src_dev'])
    tgt_val_text = read_text_path(config['tgt_dev'])
    
    '''
    Add tokenizer path 
    '''
    src_tok_path = config['src_tokenizer'] if 'src_tokenizer' in config else None
    tgt_tok_path = config['tgt_tokenizer'] if 'tgt_tokenizer' in config else None
    
    src_tokenizer = build_tokenizer(lang='src', text = ''.join(src_train_text), path=src_tok_path, config=config)
    tgt_tokenizer = build_tokenizer(lang='tgt', text = ''.join(tgt_train_text), path= tgt_tok_path, config=config)
    
    train_ds = BilingiualDataset(
        src_txt=src_train_text,
        tgt_txt=tgt_train_text,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        seq_len=config['seq_len']
    )
    val_ds = BilingiualDataset(
        src_txt= src_val_text,
        tgt_txt= tgt_val_text,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        seq_len=config['seq_len'])
    
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer  
    
def get_model(config, tokenizer_src, tokenizer_tgt, model_path=None):

    model = Transformer(
        src_vocab_size = tokenizer_src.get_vocab_size(),
        tgt_vocab_size = tokenizer_tgt.get_vocab_size(),
        seq_len = config['seq_len'],
        embed_dim = config['embed_dim'],
        n_heads = config['n_heads'],
        d_ff = config['d_ff'],
        n_blocks = config['n_blocks'],
        dropout = config['dropout']
    )
    
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    
    model.init_weights()
    
    return model


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src, tokenizer_tgt).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.encode('<|PAD|>', allowed_special=special_labels)[0])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor= config['scheduler_factor'], patience= config['scheduler_patience'])

    train_losses = []
    val_losses = []
    initial_epoch = 0
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for i, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            logits = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            label = batch['label'].to(device)  # (B, seq_len)

            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"train_loss": f" {loss.item():6.3f}"})

            train_losses.append((epoch, loss.item()))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # After last batch, include val_loss and prev_lr in tqdm
            if i == len(train_dataloader) - 1:
                
                val_loss = run_validation(model, val_dataloader, tokenizer_tgt, device)
                scheduler.step(val_loss)
                val_losses.append((epoch, val_loss))
                
                cur_epoch_train_loss = [loss for (ep, loss) in train_losses if ep == epoch]
                
                batch_iterator.set_postfix({
                    "train_loss": f" {np.mean(cur_epoch_train_loss):6.3f}",
                    "val_loss": f" {val_loss:.4f}",
                    "prev_lr": f" {scheduler.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        model_filename =  f"weights/model_{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)
    
    
def greedy_decode(model, src_input, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, greedy=True):
    sos_idx = tokenizer_tgt.encode('<|SOS|>', allowed_special=special_labels)
    eos_idx = tokenizer_tgt.encode('<|EOS|>', allowed_special=special_labels)

    
    encoder_output = model.encoder(src_input, source_mask)
    
    decoder_input = torch.empty(1, 1).fill_(sos_idx[0]).type_as(src_input).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break


        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        logits = model.decoder(decoder_input, decoder_mask, encoder_output, source_mask)
        prob = torch.softmax(logits[:, -1], dim=1)
        
        if greedy:
            _, next_word = torch.max(prob, dim=1)
        else:
            next_word = torch.multinomial(prob, num_samples=1).squeeze(1)
            
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx[0]:
            print('done')
            break

    return decoder_input.squeeze(0)


def run_validation(model, val_loader, tokenizer_tgt, device):
    model.eval()
    
    losses = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.encode('<|PAD|>', allowed_special=special_labels)[0])

    with torch.no_grad():
        for batch in val_loader:
            
            encoder_input = batch["encoder_input"].to(device)  
            decoder_input = batch["decoder_input"].to(device)             
            encoder_mask = batch["encoder_mask"].to(device) 
            decoder_mask = batch["decoder_mask"].to(device) 
    
            logits = model(encoder_input, decoder_input, encoder_mask, decoder_mask) 
            
            B, T, C = logits.shape
            label = batch['label']
            
            loss = loss_fn(logits.view(B*T, C), label.view(B*T))
            losses.append(loss.item())     
            
    model.train()
    return np.mean(losses)




if __name__ == '__main__':
    train(config=get_config())