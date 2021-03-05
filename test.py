import torch 
import argparse
from PIL import Image 
from model import CNN2RNN
from torchvision import transforms 
import pickle
from matplotlib import pyplot as plt 
from random import randint

transforms = transforms.Compose([
    transforms.Resize((356,356)),
    transforms.RandomCrop((299,299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=False,help="add image for test")
args=ap.parse_args()


with open(f'vocab.pickle','rb') as file:
    dataset=pickle.load(file)


# embedding_size=256
# hidden_size=256
# vocab_size=len(dataset.vocab)
# num_layers=2
# learning_rate=1e-4
# model=CNN2RNN(embedding_size,hidden_size,vocab_size,num_layers)
state_dict=torch.load('model_40.pth',map_location=torch.device('cpu'))
embedding_size=state_dict['embedding_size']
hidden_size=state_dict['hidden_size']
vocab_size=state_dict['vocab_size']
num_layers=state_dict['num_layers']
model=CNN2RNN(embedding_size,hidden_size,vocab_size,num_layers)
model.load_state_dict(state_dict['state_dict'])
model.eval()

if args.image is None:
    data_test=torch.load('test_set.pth')

def preprocessing():
    img=Image.open(args.image).convert("RGB")
    img=transforms(img)
    return img

def show_image(img,title=None):
    img[0]=img[0]*0.5
    img[1]=img[1]*0.5
    img[2]=img[2]*0.5
    img[0]+=0.5
    img[1]+=0.5
    img[2]+=0.5
    img=img.permute(1,2,0).numpy()

    fig,ax=plt.subplots(1,1)
    im=ax.imshow(img,interpolation='nearest')

    if title is not None:
        ax.set_title(title)
    if args.image is None:
        temp=randint(0,10000)
        name_file="test"+str(temp)+".png"
    else:
        s=str(args.image)
        name_file="result"+s[s.find('/')+1:s.find('.')]+".png"

    plt.savefig('Image/Result/{}'.format(name_file))
    plt.show()

if __name__=="__main__":

    if args.image is None:
        dataiter=iter(data_test)
        img,_=next(dataiter)
        caption=model.caption_image(img[0:1],dataset.vocab)
        show_image(img[0:1].squeeze(0),caption)

    else:
        img=preprocessing()
        caption=model.caption_image(img.unsqueeze(dim=0),dataset.vocab)
        show_image(img,caption)


