#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

import argparse
import sys
import os
import smdebug.pytorch as smd
from PIL import ImageFile
import torch.nn.functional as F
import torch.utils.data.distributed
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    logger.info("starting testing ...")
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # sum up batch loss
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, validation_loader, criterion, optimizer, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=10
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch: {epoch}, Phase {phase}")
            
            if phase =='train':
                hook.set_mode(smd.modes.TRAIN)
                model.train()
            else:
                hook.set_mode(smd.modes.EVAL)
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
#                 inputs = inputs.permute(0, 1, 2, 3)
#                 inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info(
                '{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, best_loss)
            )        
        if loss_counter == 1:
            break
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
               nn.Linear(num_features, 256),
               nn.ReLU(inplace=True),
               nn.Linear(256, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()])

    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()])
  
    train_set = torchvision.datasets.ImageFolder(root=train_data_path, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_set = torchvision.datasets.ImageFolder(root=test_data_path, transform=testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    validation_set = torchvision.datasets.ImageFolder(
        root=validation_data_path, transform=testing_transform)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size) 
    
    return train_loader, test_loader, validation_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    train_loader, test_loader, validation_loader = create_data_loaders(args.train_dir, args.batch_size)
    model = net()
    model = model.to(device)
    hook = smd.Hook.create_from_json_file() # Instantiate the hook from the estimator class
    hook.register_module(model) # Register the model to the hook
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    hook.register_loss(loss_criterion) # Register the loss function to the hook
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Start Model Training")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Start Model Testing")
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Model Saving")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    main(args)