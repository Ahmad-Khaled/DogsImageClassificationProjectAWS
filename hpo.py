#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
# import io, webdataset
from awsio.python.lib.io.s3.s3dataset import S3Dataset

import argparse

class S3ImageSet(S3Dataset):
    def __init__(self, urls, transform=None):
        super().__init__(urls)
        self.transform = transform

    def __getitem__(self, idx):
        img_name, img = super(S3ImageSet, self).__getitem__(idx)
        # Convert bytes object to image
        img = Image.open(io.BytesIO(img)).convert('RGB')
        
        # Apply preprocessing functions on data
        if self.transform is not None:
            img = self.transform(img)
        return img

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("START VALIDATING")
    if hook:
        hook.set_mode(modes.EVAL)
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_optim(outputs, targets)
            test_loss += loss.item()

    epoch_time = time.time() - start
    epoch_times.append(epoch_time)
    print(
        "Epoch %d: train loss %.3f, val loss %.3f, in %.1f sec"
        % (i, train_loss, val_loss, epoch_time)
    )
    
def train(model, train_loader, loss_optim, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook = get_hook(create_if_not_exists=True)

    if hook:
        hook.register_loss(loss_optim)
    # train the model

    for i in range(epoch):
        print("START TRAINING")
        if hook:
            hook.set_mode(modes.TRAIN)
        start = time.time()
        model.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad() # Sets the gradients of all optimized torch.Tensor s to zero
            outputs = model(inputs) # input data for training
            loss = loss_optim(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

#         print("START VALIDATING")
#         if hook:
#             hook.set_mode(modes.EVAL)
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for _, (inputs, targets) in enumerate(validloader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = loss_optim(outputs, targets)
#                 val_loss += loss.item()

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print(
            "Epoch %d: train loss %.3f, in %.1f sec"
            % (i, train_loss, epoch_time)
        )
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    gpu = True
    model: str2bool = "resnet50"

    model = models.__dict__[model](pretrained=True)
    if gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    dataset1 = get_dataset(trian=True)
    dataset2 = get_dataset(test=True)
    # urls can be S3 prefix containing images or list of all individual S3 images
    # include all files under 's3_prefix' folder starting with '0' and '1'
    urls_train = ['s3://demo-udacity-training123/dogImages/train/0', 's3://demo-udacity-training123/dogImages/train/1']
    urls_train = ['s3://demo-udacity-training123/dogImages/test/0', 's3://demo-udacity-training123/dogImages/test/1']
    
    dataset1 = S3ImageSet(urls_train)
    dataset2 = S3ImageSet(urls_test)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--model", type=str, default="resnet50")
        
    args=parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}:{value}")
    
    main(args)

    
    image_datasets = {x: datasets.ImageFolder(root = (args.data_dir + '/' + x), transform = data_transforms[x]) for x in ['train', 'valid']}

