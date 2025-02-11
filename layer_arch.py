from layers import  BatchNorm, BatchNorm2D, Conv2D, Dense, Dropout, Flatten, LeakyReLU, MaxPool2D, Model, SoftmaxCrossEntropy, SpatialDropout2D

layers2 = [
    Conv2D(in_nchannel=3, out_nchannel=6, kernel_size=3,padding=1),
    BatchNorm2D(),
    LeakyReLU(),
    MaxPool2D(),
    
    Conv2D(in_nchannel=6, out_nchannel=16, kernel_size=3,padding=1),
    BatchNorm2D(),
    LeakyReLU(),
    MaxPool2D(),
    
    Flatten(),
    
    Dense(out_dim=120),
    BatchNorm(),
    LeakyReLU(),
    
    Dense(out_dim=84),
    BatchNorm(),
    LeakyReLU(),
    
    Dense(out_dim=10)
]

'''
for now the best is 
with 57.5 % 
layers1 = [
    Conv2D(in_nchannel=3, out_nchannel=32, kernel_size=3, padding=1),
    BatchNorm2D(),
    LeakyReLU(),
    SpatialDropout2D(prob=0.2),
    MaxPool2D(dim=2, stride=2),
    
    Flatten(),

    Dense(out_dim=128),
    
    BatchNorm(),
    LeakyReLU(),
    Dropout(prob=0.4),  

    Dense(out_dim=10)
]
'''


'''
this is testing and find improving
'''
layers1 = [
    Conv2D(in_nchannel=3, out_nchannel=32, kernel_size=3, padding=1),
    BatchNorm2D(),
    LeakyReLU(),
    SpatialDropout2D(prob=0.2),
    MaxPool2D(dim=2, stride=2),
    
    Flatten(),

    # Dense(out_dim=128),

    Dense(out_dim=256),
    BatchNorm(),
    LeakyReLU(),
    Dropout(prob=0.4),  

    Dense(out_dim=10)
]



