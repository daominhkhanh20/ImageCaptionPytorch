import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import spacy
import pandas as pd
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self,freq_threshold=1):
        # set index token to string token
        self.freq_threshold=freq_threshold
        self.itos={1:"<PAD>",2:"<SOS>",3:"<EOS>",4:"<UNK>"}
        # string token to index token
        self.stoi={value:index for index,value in self.itos.items()}
        self.spacy_eng=spacy.load("en_core_web_sm")

    def tokenize_en(self,text):
        return [tok.text for tok in self.spacy_eng.tokenizer(text)]

    def __len__(self):
        return len(self.itos)

    def build_vocab(self,caption_list):
        frequencies=Counter()
        idx=4# at this time, len vocabulary = 4

        for sentence in caption_list:
            for word in self.tokenize_en(sentence):
                frequencies[word]+=1
                # add word into vocabulary if it reaches min frequency
                if frequencies[word]==self.freq_threshold:
                    self.itos[idx]=word
                    self.stoi[word]=idx
                    idx+=1 # increase size vocabulary

    # conver text to vector int
    def numericalize(self,text):
        tokenize=self.tokenize_en(text)
        return [self.stoi[word] if word in self.stoi else self.stoi["<UNK>"] for word in tokenize]


class Flickr8kDataset(Dataset):
    def __init__(self,root_dir,caption_file,transforms=None,freq_threshold=1):
        self.root_dir=root_dir
        self.caption_file=caption_file
        self.transforms=transforms
        self.freq_threshold=freq_threshold
        self.df=pd.read_csv(root_dir+"/"+caption_file)
        self.images=self.df['image']
        self.captions=self.df['caption']

        self.vocab=Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        image_name=self.images[idx]
        path=self.root_dir+"/Images/"+image_name
        img=Image.open(path)
        caption=self.captions[idx]
        if self.transforms is not None:
            img=self.transforms(img)

        caption_vector=[]
        caption_vector.append(self.vocab.stoi["<SOS>"])
        caption_vector+=self.vocab.numericalize(caption)
        caption_vector.append(self.vocab.stoi["<EOS>"])
        return img,torch.tensor(caption_vector)


class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx=pad_idx # value when map blank into integer

    def __call__(self,batch):
        # batch has a lot of pair(image, caption_vector) in the batch_size
        imgs=[item[0].unsqueeze(0) for item in batch]
        targets=[item[1] for item in batch]
        imgs=torch.cat(imgs,dim=0)# batch_size*3*height*width
        # batch_first: output will be B*T*X if True, otherwise T*B*X
        targets=pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)
        return imgs,targets

def get_data_loader(root_dir,caption_file,transforms,batch_size=64,num_workers=8,shuffle=True,pin_memory=True):
    dataset=Flickr8kDataset(
        root_dir=root_dir,
        caption_file=caption_file,
        transforms=transforms
        )
    print(len(dataset.vocab))
    pad_idx=dataset.vocab.stoi["<PAD>"]

    dataloader=DataLoader(
        dataset=dataset, # dataset from which to load the data
        batch_size=batch_size,# how many sample per batch at every epoch
        shuffle=shuffle,# If True, data reshuffle at evey epochs
        num_workers=num_workers,#how many subprocesses to uses for data loading
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx) #merge list of samples to form a mini batch tensor
    )
    return dataset,dataloader

if __name__=="__main__":
    transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    dataset,data_loader=get_data_loader(
            root_dir="Flickr8k",
            caption_file="captions.txt",
            transforms=transform
        )
    