## Currently hasn't been finished

import torch
from torch import nn
from torchvision import models
from torch.nn import functional 

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.resnet=models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad=False
        modules=list(self.resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
    
    def forward(self,images):
        features=self.resnet(images)# batch_size*2048*7*7
        features=features.permute(0,2,3,1)#batch_size*7*7*2048
        features=features.reshape(features.size(0),-1,features.size(-1))#batch_size*49*2048
        return features


class BahdanuAttention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,attention_dim):
        super(BahdanuAttention,self).__init__()
        self.attention_dim=attention_dim
        self.encoder_dim=encoder_dim
        self.decoder_dim=decoder_dim

        #initialize param for calculate engery function
        self.decoder_attention=nn.Linear(decoder_dim,attention_dim)
        self.encoder_attention=nn.Linear(encoder_dim,attention_dim)
        self.full_attention=nn.Linear(attention_dim,1)
        self.relu=nn.ReLU()

    def forward(self,encoder_output,hidden_decoder):
        """
        encoder_output: shape (batch_size*num_pixel*encoder_dim)
        hidden_decoder: shape (batch_size*decoder_dim)
        """
        energy=self.full_attention(
            torch.tanh(
                self.decoder_attention(hidden_decoder).unsqueeze(1)+self.encoder_attention(encoder_output)
            )
        )#batch_size*num_pixel*attention_dim
        attention_weights=functional.softmax(energy,dim=-1)#batch_size*num_pixel

        context_vector=(encoder_output*attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector,attention_weights#batch_size*attention_dim




class Decoder(nn.Module):
    def __init__(self,embedding_size,attention_dim,encoder_dim,decoder_dim,vocab_size,drop_pro=0.3):
        super(Decoder,self).__init__()
        self.vocab_size=vocab_size

        self.embedded=nn.Embedding(vocab_size,embedding_size)
        self.attention=BahdanuAttention(encoder_dim,decoder_dim,attention_dim)
        self.init_h=nn.Linear(encoder_dim,decoder_dim)
        self.init_c=nn.Linear(encoder_dim,decoder_dim)
        self.lstm_cell=nn.LSTMCell(embedding_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta=nn.Linear(decoder_dim,encoder_dim)
        self.fcn=nn.Linear(decoder_dim,vocab_size)
        self.drop=nn.Dropout(drop_pro)
        self.init_weights()


    def init_weights(self):
        self.embedded.weight.data.uniform_(-0.1,0.1)
        self.fcn.bias.data.fill_(0)
        self.fcn.weight.data.uniform_(-0.1,0.1)  

    def init_hidden_state(self,encoder_outputs):
        mean_encoder_out=encoder_outputs.mean(dim=1)
        h=self.init_h(mean_encoder_out)#batch_size*decoder_dim
        c=self.init_c(mean_encoder_out)#batch_size*decoder_dim
        return h,c

    def forward(self,features,captions):
        embeddes=self.embedded(captions)#batch_size*vocab_size*embedding_size
        h,c=self.init_hidden_state(features)#batch_size*decoder_dim

        seq_length=len(captions[0])-1
        batch_size=captions.size()[0]
        num_features=features.size()[1]#encoder_dim

        preds=torch.zeros(batch_size,seq_length,self.vocab_size).to(device)
        attention_weights=torch.zeros(batch_size,seq_length,num_features).to(device)

        for s in range(seq_length):
            context,attention_weight=self.attention(features,h)
            lstm_in=torch.cat((embeddes[:,s],context),dim=1)
            h,c=self.lstm_cell(lstm_in,(h,c))
            output=self.fcn(self.drop(h))#batch_size*vocab_size
            preds[:,s]=output
            attention_weights[:,s]=attention_weight
        
        return preds,attention_weights
    
    def generate_caption(self,features,max_len=20,vocabulary=None):
        batch_size=features.size(0)
        h,c=self.init_hidden_state(features)
        attention_weights=[]
        captions=[]
        word=torch.tensor(vocabulary.stoi['<SOS>']).view(1,-1).to(device)
        embedds=self.embedded(word)

        for _ in range(max_len):
            contex,attention_weight=self.attention(features,h)
            lstm_in=torch.cat((embedds[:,0],contex),dim=1)
            h,c=self.lstm_cell(lstm_in,(h,c))
            output=self.fcn(self.drop(h))
            output=output.view(batch_size,-1)

            predict_word_index=output.argmax(dim=1)
            captions.append(predict_word_index.item())
            
            if vocabulary.itos[predict_word_index.item()]=="<EOS>":
                break
            embedds=self.embedded(predict_word_index.unsqueeze(0))

        return [vocabulary.itos[idx] for idx in captions]
    

class CNN2RNN(nn.Module):
    def __init__(self,embedding_size,attention_dim,encoder_dim,decoder_dim,vocab_size):
        super(CNN2RNN,self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder(embedding_size,attention_dim,encoder_dim,decoder_dim,vocab_size)

    def forward(self,images,captions):
        """
        image: batch_size*3*height*width
        caption: batch_size*vocab_size
        """
        features=self.encoder(images)#batch_size*7*7*2048
        outputs=self.decoder(features,captions)
        return outputs


# embedding_size=10
# attention_dim=20
# encoder_dim=12
# decoder_dim=15
# vocab_size=100

# image=torch.randint(1,255,(10,3,224,224))
# caption=torch.randint(1,100,(10,100))
# model=CNN2RNN(embedding_size,attention_dim,encoder_dim,decoder_dim,vocab_size)
# outputs=model(image,caption)
