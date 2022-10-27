import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JPEG_CONTENT_TYPE = 'image/jpeg'

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    model.to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Starting to load the model...")
        model.load_state_dict(torch.load(f, map_location = device))
        logger.info("Successfully loaded the model")
    
    logger.info('Done loading model')
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type in JPEG_CONTENT_TYPE:
        logger.info(f"Returning an image of type {content_type}" )
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Requested an unsupported Content-Type: {content_type}, Accepted Content-Type are: {JPEG_CONTENT_TYPE}")

def predict_fn(input_object, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Starting the prediction process...")
    test_transform =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() ]
    )
    logger.info("Starting to apply Transforms to the input image")
    input_object=test_transform(input_object)
    if torch.cuda.is_available():
        input_object = input_object.cuda()
    logger.info("Completed applying Transforms to the input image")
    model.eval()
    with torch.no_grad():
        logger.info("Starting the model invokation")
        prediction = model(input_object.unsqueeze(0))
    return prediction