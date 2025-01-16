
    
def get_config():
    return {
    'src_lang':'en',
    'tgt_lang':'ta',
    
    'src_vocab_size': 8000,
    'tgt_vocab_size': 16000,
    'seq_len': 384,
    'embed_dim':512,
    'n_heads':16,
    'd_ff':512 * 4,
    'n_blocks':6,
    'dropout':0.1,
    
    'lr':1e-4,
    'num_epochs':10,
    'batch_size':8,
    'scheduler_patience': 0, 
    'scheduler_factor': 0.75,
    
    
    'src_train':'train.en.txt',
    'tgt_train':'train.ta.txt',
    'src_dev': 'test.en.txt',
    'tgt_dev': 'test.ta.txt',
    
    
    'src_tokenizer':'tok_params/en_tokenizer.pkl',
    'tgt_tokenizer':'tok_params/ta_tokenizer.pkl',
    }
