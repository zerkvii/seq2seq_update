import torch
import torch.nn as nn
from tqdm import tqdm
import time
import math
from TransformerT import Seq2SeqTransformer
from loader import get_loader,get_tokenizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

cs_tokenizer,en_tokenizer=get_tokenizer()
train_loader,val_loader,test_loader=get_loader()


src_pad_id = cs_tokenizer.token_to_id('<PAD>')
trg_pad_id = en_tokenizer.token_to_id('<PAD>')
src_eos_id = cs_tokenizer.token_to_id('<EOS>')
src_sos_id = cs_tokenizer.token_to_id('<SOS>')
trg_eos_id = en_tokenizer.token_to_id('<EOS>')
trg_sos_id = en_tokenizer.token_to_id('<SOS>')

input_d = cs_tokenizer.get_vocab_size()
output_d = en_tokenizer.get_vocab_size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = input_d
OUTPUT_DIM = output_d
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# encoder = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS,
#                   ENC_PF_DIM, ENC_DROPOUT, device)
# decoder = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS,
#                   DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# model = Seq2Seq(encoder, decoder, src_pad_id, trg_pad_id, device).to(device)
# encoder_layer=nn.TransformerEncoderLayer(INPUT_DIM,ENC_HEADS,ENC_PF_DIM,ENC_DROPOUT,device=device)
# decoder_layer=nn.TransformerDecoderLayer(OUTPUT_DIM,DEC_HEADS,DEC_PF_DIM,DEC_DROPOUT,device=device)
# enocder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=ENC_LAYERS)
# decoder=nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=DEC_LAYERS)
# model=nn.Transformer(d_model=HID_DIM,nhead=ENC_HEADS,dim_feedforward=ENC_PF_DIM)

model = Seq2SeqTransformer(ENC_LAYERS,DEC_LAYERS,HID_DIM,ENC_HEADS,INPUT_DIM,OUTPUT_DIM,ENC_PF_DIM).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(f'The model has {count_parameters(model)} trainable parameters)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_id)

# def generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
    # tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == src_pad_id).transpose(0, 1)
    tgt_padding_mask = (tgt == trg_pad_id).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0
    pbar = tqdm(iterator)

    for i, batch in enumerate(pbar):

        src = batch[0].to(device).T
        trg = batch[1].to(device).T

        train_trg=trg[:-1,:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, train_trg)

        optimizer.zero_grad()

        output= model(src,train_trg,src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        # output, _ = model(src, trg[:, :-1],)

        # trg = [trg len,batch size]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[1:, :].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        pbar.set_postfix({'loss':loss.item()})

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(tqdm(iterator)):
            
            src = batch[0].to(device).T
            trg = batch[1].to(device).T

            train_trg=trg[:-1,:]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, train_trg)


            output= model(src,train_trg,src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[1:,:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def start_train():

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

def start_test():
    model.load_state_dict(torch.load('tut-model.pt'))

    test_loss = evaluate(model, test_loader, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# function to generate output sequence using greedy algorithm
def greedy_decode(src, src_mask, max_len, start_symbol):

    model.load_state_dict(torch.load('tut-model.pt')) 
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (model.generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == trg_eos_id:
            break
    return ys

def greedy_translate_sentence(sentence,max_len = 50):
    model.load_state_dict(torch.load('tut-model.pt')) 
    model.eval()    
    src_indexes = cs_tokenizer.encode('<SOS> '+sentence+' <EOS>').ids

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # print(src_tensor)
    

    # src_mask = model.make_src_mask(src_tensor)
    src_mask=(src_tensor!=src_pad_id).unsqueeze(1).unsqueeze(2)

    
    with torch.no_grad():
        enc_src = model.encode(src_tensor, src_mask)

    trg_indexes = [trg_sos_id]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        # print(trg_tensor.shape)
        trg_mask = model.make_trg_mask(trg_tensor)
        # print(trg_mask)
        with torch.no_grad():
            output, attention = model.decode(trg_tensor, enc_src, trg_mask, src_mask)
        # print(output.shape)
        pred_token = output.argmax(2)[:,-1].item()
        # print(pred_token)
        trg_indexes.append(pred_token)

        if pred_token == trg_eos_id:
            break
    
    trg_tokens = en_tokenizer.decode(trg_indexes)
    
    return trg_tokens,trg_indexes

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

if __name__=='__main__':
    start_train()