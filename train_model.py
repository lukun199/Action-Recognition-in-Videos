import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torch.utils.data as data

from encoder import CNN_fc_EmbedEncoder
from decoder import DecoderRNN
from dataloader import Dataset_CRNN, Dataset_CRNN_VAL

# set path
data_dir = '../dataset' # '/processed_train'
save_model_path = "./ckpt/"

if not os.path.exists(save_model_path): os.makedirs(save_model_path)
if not os.path.exists('./tensor_board'): os.makedirs('./tensor_board')
summary = SummaryWriter('./tensor_board')

# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
dropout_p = 0.5      # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 2
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
category = 50             # number of target category
batch_size = 32
epochs = 202        # training epochs
learning_rate = 1e-5
log_interval = 10


# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:1" if use_cuda else "cpu")   # use CPU or GPU

# reset data loader
train_loader_params = {'batch_size': batch_size, 'shuffle': True}
val_loader_params = {'batch_size': batch_size, 'shuffle': False}
# dataloaders
loader_train = Dataset_CRNN(data_path = data_dir)
loader_val = Dataset_CRNN_VAL(data_path = data_dir)
train_data_loader = data.DataLoader(loader_train, **train_loader_params)
val_data_loader = data.DataLoader(loader_val, **val_loader_params)
dict_train = loader_train.idx2word
dict_val = loader_val.idx2word

# models
embed_encoder = CNN_fc_EmbedEncoder().to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=category).to(device)

crnn_params = list(list(rnn_decoder.parameters()) +list(embed_encoder.parameters()))
optimizer = torch.optim.Adam(crnn_params, lr=learning_rate, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

scores = []
g_minibatch_train = 0
g_minibatch_val = 0

def train(log_interval, cnn_encoder, rnn_decoder, device, train_loader, optimizer, epoch):
    global g_minibatch_train
    # set model as training mode
    cnn_encoder.train()
    rnn_decoder.train()

    this_batch_right, total_batch = 0, 0

    for batch_idx, (train_data, train_label, train_temp) in enumerate(train_loader):

        # distribute data to device
        train_data, train_label, train_temp = train_data.to(device), train_label.to(device), train_temp.cpu()

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(train_data, train_temp))   # output has dim = (batch, number of classes)

        loss = criterion(output, train_label.long())

        loss.backward()
        optimizer.step()

        print('[%4d / %4d]    '%(batch_idx, epoch) , '    loss = ', loss.item())
        summary.add_scalar('train_loss', loss.item(), g_minibatch_train)
        g_minibatch_train += 1

        this_batch_right += sum(torch.max(output, 1)[1] == train_label).item()
        total_batch += train_label.size()[0]

    print('train accuracy = %d/%d'%(this_batch_right, total_batch))
    summary.add_scalar('train_accuracy', this_batch_right/total_batch, epoch)

    # return loss, step_score

def validation(cnn_encoder, rnn_decoder, device, optimizer, val_loader, epoch):
    global g_minibatch_val

    # set model as testing mode
    cnn_encoder.eval()
    rnn_decoder.eval()

    this_batch_right, total_batch = 0, 0

    with torch.no_grad():
        for X, y, b in val_loader:
            # distribute data to device
            X, y, b = X.to(device), y.to(device), b.to(device)

            output = rnn_decoder(cnn_encoder(X, b))

            g_minibatch_val += 1

            list_train = list(map(lambda x: dict_train[x], torch.max(output, 1)[1].cpu().numpy()))
            list_val = list(map(lambda x: dict_val[x], y.cpu().numpy()))
            this_batch_right += sum(list(map(lambda x,y: x==y, list_train,list_val)))
            total_batch += y.size()[0]

    print('test accuracy =  %d/%d '%(this_batch_right, total_batch))
    summary.add_scalar('test_accuracy', this_batch_right/total_batch, epoch)


    if epoch%25 == 0:
        # save Pytorch models of best record
        torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))



# start training
for epoch in range(epochs):
    train(log_interval, embed_encoder, rnn_decoder, device, train_data_loader, optimizer, epoch)
    validation(embed_encoder, rnn_decoder, device, optimizer, val_data_loader, epoch)


