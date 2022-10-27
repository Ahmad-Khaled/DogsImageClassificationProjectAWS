**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Image classification using AWS Sagemaker

<!-- **TODO:** Write a short introduction to your project. -->

###### Short Introduction: This project uses Convolutional Neural Networks (CNN) with Pytorch framework to classify dog breeds. Given an image of a dog, this algorithm will identify an estimate of the canine’s breed. 


## Dataset

### Overview
<!-- **TODO**: Explain about the data you are using and where you got it from -->
###### Using the link: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip, it was possible to download and unzip the dogImages directory and later send it to S3.

### Access
<!-- **TODO**: Explain how you are accessing the data in AWS and how you uploaded it -->

## Hyperparameter Tuning
<!-- **TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search -->

###### I used Resnet50 as from my previous practice it was fit for that type of image classification changed hyper parameters for learning rate and batch size: "lr": (0.001, 0.1), "batch_size": [128, 256]



Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
