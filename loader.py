from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
    WordPieceTrainer, UnigramTrainer
# a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace
data = load_dataset('wmt16', 'cs-en')
train_data = data['train'][:10000]
val_data = data['validation']
test_data = data['test']


class CustomSet(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data['translation']

    def __getitem__(self, idx):
        return self.raw_data[idx]

    def __len__(self):
        return len(self.raw_data)


train_ds = CustomSet(train_data)
val_ds = CustomSet(val_data)
test_ds = CustomSet(test_data)

unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<UNK>", "<EOS>", "<PAD>", "<SOS>"]  # special tokens


def prepare_tokenizer_trainer(alg):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=spl_tokens)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            unk_token=unk_token, special_tokens=spl_tokens)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=spl_tokens)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(special_tokens=spl_tokens)

    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer


cs_tokenizer, cs_trainer = prepare_tokenizer_trainer('BPE')
en_tokenizer, en_trainer = prepare_tokenizer_trainer('BPE')

cs_lang = [item['cs'] for item in train_ds]
en_lang = [item['en'] for item in train_ds]

cs_tokenizer.train_from_iterator(cs_lang, trainer=cs_trainer)
en_tokenizer.train_from_iterator(en_lang, trainer=en_trainer)

src_pad_id = cs_tokenizer.token_to_id('<PAD>')
trg_pad_id = en_tokenizer.token_to_id('<PAD>')
src_eos_id = cs_tokenizer.token_to_id('<EOS>')
src_sos_id = cs_tokenizer.token_to_id('<SOS>')
trg_eos_id = en_tokenizer.token_to_id('<EOS>')
trg_sos_id = en_tokenizer.token_to_id('<SOS>')

cs_tokenizer.enable_padding(length=50, pad_id=src_pad_id)
cs_tokenizer.enable_truncation(max_length=50)
en_tokenizer.enable_padding(length=50, pad_id=trg_pad_id)
en_tokenizer.enable_truncation(max_length=50)


def collate_fn(batch):
    cs_ls, en_ls = [], []
    for item in batch:
        cs_sent = '<SOS> '+item['cs']+' <EOS>'
        en_sent = '<SOS> '+item['en']+' <EOS>'
        cs_ls.append(torch.LongTensor(cs_tokenizer.encode(cs_sent).ids))
        en_ls.append(torch.LongTensor(en_tokenizer.encode(en_sent).ids))
    return torch.vstack(cs_ls), torch.vstack(en_ls)


def get_loader():
    train_loader = DataLoader(dataset=train_ds, batch_size=64,
                              shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=val_ds, batch_size=64,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_ds, batch_size=64,
                             shuffle=True, collate_fn=collate_fn)
    
    return train_loader,valid_loader,test_loader

def get_tokenizer():
    return cs_tokenizer,en_tokenizer