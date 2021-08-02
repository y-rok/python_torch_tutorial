"""
    위의 예시는 정의된 AG_News를 활용했다면...
    일반적으로 Torch 사용시를 위해 Custom Dataset을 정의하고 DataLoader로 데이터를 불러와보자
    
    Custom Dataset 관련 - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html 
    
    TextDataset -> 데이터 전처리 (tokenization, tensor만듬) / mapy-style dataset
    DataLoader의 collate_fn -> batch 단위의 Tensor 생성 
        maps style의 dataset을 사용했으므로 shuffling 등 가능
"""
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import AG_NEWS
import time
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# AG New 데이터 불러옴
# https://pytorch.org/text/stable/datasets.html#ag-news
# train: 120000 / test: 7600 / class 수 = 4
train_iter = AG_NEWS(root="./data/",split='train') 
print(type(train_iter))
# next(train_iter) # (class, 텍스트)
tokenizer = get_tokenizer('basic_english')

# generator - text를 tokenization하여 yield
def yield_tokens(data_iter):
    for _,text in data_iter:
        yield tokenizer(text)



# 전체 text를 tokenization 하며 vocab 구축 
vocab = build_vocab_from_iterator(yield_tokens(train_iter),specials=["<unk>"]) 
vocab.set_default_index(vocab["<unk>"])

vocab_list = vocab.get_itos()
print("vocab size = %d"%len(vocab_list))

print(vocab(["here","is"]))


class TextDataset(Dataset):
    """
        custom dataset을 위해서 다음 3가지의 method를 구현해야함
        dataset은 크게 2종류로 만들 수 있음 (iterable style, map style)
        여기서는 map style로 만들어봄
    """
    def __init__(self, root,split, transform=None):
        
        train_iter = AG_NEWS(root=root,split=split) 
        self.train_data = [x for x in train_iter]
        self.transform = transform
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx):
        
        print(idx)
        print("\n")
        """ transform method를 통해 전처리 수행 후 tensor로 바꾸어서 Return """
        return self.transform(self.train_data[idx])

def transform(data):
    """
        tokenization 수행 후 tensor로 바꿈
    """
#     print("transform")
    get_token_id_list = vocab(tokenizer(data[1]))
    get_label_id = data[0]-1
    
    time.sleep(1)
    
    return torch.tensor(get_label_id), torch.tensor(get_token_id_list)
    

def collate_batch(batch):
    
    label_list, text_list, offsets = [], [],[0]

    for _label, _text in batch:
        label_list.append(_label)
        text_list.append(_text)
        offsets.append(_text.size(0))
    
    label_list=torch.tensor(label_list)
    text_list=torch.cat(text_list,0)
    offsets=torch.tensor(offsets).cumsum(dim=0)
    
    return label_list, text_list, offsets
    
text_dataset=TextDataset(root="./data/",split='train',transform=transform) 

""" 데이터가 잘 생성됨을 확인 """
# dataloader=DataLoader(text_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch, )
# next(iter(dataloader)) 

""" Multi processing 사용시 데이터 전처리 후 Data Loading이 빨라짐을 확인 """


multi_dataloader=DataLoader(text_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch, num_workers=2)
single_dataloader=DataLoader(text_dataset, batch_size=4, shuffle=True, collate_fn=collate_batch, num_workers=0)

start = time.time()
# next(iter(single_dataloader)) 
# next(iter(single_dataloader)) 
print(next(iter(single_dataloader))[0].size(0))
# print(next(iter(single_dataloader))[0].size(0))
print("elapsed time (single wokrer) = %.2f"%(time.time()-start))

start = time.time()
print(next(iter(multi_dataloader))[0].size(0))
# print(next(iter(multi_dataloader))[0].size(0))

print("elapsed time (multiple wokrer) = %.2f"%(time.time()-start))
