import torch
from load_data import get_data_loader,MyCollate
from model import CNN2RNN
from matplotlib import pyplot as plt
from torchvision import transforms
from torch import nn 
from torch import optim
import numpy as np 
from torch.utils.data import random_split
from torch.utils.data import DataLoader

transforms=transforms.Compose([
    transforms.Resize((356,356)),
    transforms.RandomCrop((299,299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

])


dataset,data_loader=get_data_loader(
        root_dir="Flickr8k",
        caption_file="captions.txt",
        transforms=transforms
    )


def convert_data_loader(pad_idx,loader):
    return DataLoader(
        dataset=loader, # dataset from which to load the data
        batch_size=64,# how many sample per batch at every epoch
        shuffle=True,# If True, data reshuffle at evey epochs
        num_workers=8,#how many subprocesses to uses for data loading
        pin_memory=True,
        collate_fn=MyCollate(pad_idx) #merge list of samples to form a mini batch tensor
    )


n=len(data_loader)
train_size=int(n*0.8)
val_size=int(n*0.1)
test_size=n-train_size-val_size


pad_idx=dataset.vocab.stoi["<PAD>"]
train_set,val_set,test_set=random_split(data_loader,[train_size,val_size,test_size])
train_set=convert_data_loader(pad_idx,train_set)
val_set=convert_data_loader(pad_idx,val_set)
test_set=convert_data_loader(pad_idx,test_set)
torch.save(test_set,'test.pth')

train_CNN=False

#hype-param
embedding_size=256
hidden_size=256
vocab_size=len(dataset.vocab)
num_layers=2
learning_rate=1e-4
epochs=50

print(vocab_size)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN2RNN(embedding_size,hidden_size,vocab_size,num_layers).to(device)
criterion=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer=optim.Adam(model.parameters(),lr=learning_rate)


def plot_loss(loss_train_his,loss_val_his,epoch):
    plt.plot(np.arange(epochs),loss_train_his,label="Train")
    plt.plot(np.arange(epochs),loss_val_his,label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylable("Loss")
    plt.show()
    plt.savefig("loss{}.png".format(epoch))

def save_model(epoch):
    model_state={
        'num_epochs':epoch,
        'embedding_size':embedding_size,
        'hidden_size':hidden_size,
        'vocab_size':vocab_size,
        'num_layers':num_layers,
        'state_dict':model.state_dict()
    }
    torch.save(model_state,"model_{}.pth".format(epoch))

i=0
def show_image(img,title):
    img[0]=img[0]*0.5
    img[1]=img[1]*0.5
    img[2]=img[2]*0.5
    img[0]+=0.5
    img[1]+=0.5
    img[2]+=0.5

    img=img.numpy().transpose((1,2,0))
    plt.imshow(img)

    if title is not None:
        plt.title(title)
    plt.savefig(img,"img{}.png".format(i))
    i+=1

def evaluate():
    model.eval()
    loss=[]
    for i,(imgs,caption) in enumerate(val_set):
        imgs=imgs.to(device)
        caption=caption.to(device)
        outputs=model(imgs,caption[:-1])
        loss1=criterion(outputs.reshape(-1,outputs.shape[2]),caption.reshape(-1))
        loss.append(loss1.item())
    return sum(loss)*1.0/len(loss)

def train():
    # setting no train CNN model
    for name,param in model.encoder.model.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad=True
        else:
            param.requires_grad=train_CNN

    loss_train_his=[]
    loss_val_his=[]
    #model.train()
    for epoch in range(1,epochs+1):
        model.train()
        for idx,(images,captions) in enumerate(train_set):
            optimizer.zero_grad()
            imgs=images.to(device)
            captions=captions.to(device)
            outputs=model(imgs,captions[:-1])#shape seq_length*batch_size*vocab_size
            loss=criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))
            loss.backward(loss)
            optimizer.step()
        
        loss_train_his.append(loss.item())
        loss_val=evaluate()
        loss_val_his.append(loss_val)
        print("Epoch:{} -- Loss:{} -- Val_loss:{}".format(epoch,loss.item(),loss_val))
        if epoch%10==10:
            save_model(epoch)
            plot_loss(loss_train_his,loss_val_his,epoch)
            with torch.no_grad():
                dataiter=iter(test_set)
                img,caption=next(dataiter)
                encode = model.encoder(img[0:1]).to(device)
                caption=model.caption_image(encode,dataset.vocab)
                show_image(img[0],caption)




if __name__=="__main__":
    train()

