from dataclasses import dataclass


@dataclass
class ModelArgs:
    #Hyperparameters
    
    epochs = 5
    block_size = 128
    batch_size = 64
    embeddings_dims = 786
    attn_dropout = 0.1
    no_of_heads = 6 #IMP needs to be thoroughly calculated
    dropout = 0.1
    # epochs = 100
    val_epochs = 2
    max_lr = 2e-4
    no_of_decoder_layers = 6 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    clip = 1.0
    device = 'cuda'
    no_kv_heads = 2
    vocab_size = 50258

