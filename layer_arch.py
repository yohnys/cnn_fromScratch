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




# layers1 = [
#     Conv2D(in_nchannel=3, out_nchannel=32, kernel_size=3, padding=1),
#     BatchNorm2D(),
#     LeakyReLU(),
#     MaxPool2D(dim=2, stride=2),

#     Conv2D(in_nchannel=32, out_nchannel=64, kernel_size=3, padding=1),
#     BatchNorm2D(),
#     LeakyReLU(),
#     MaxPool2D(dim=2, stride=2),

#     Flatten(),

#     Dense(out_dim=128),
#     BatchNorm(),
#     LeakyReLU(),
#     Dropout(prob=0.4),  

#     Dense(out_dim=64),
#     BatchNorm(),
#     LeakyReLU(),
#     Dropout(prob=0.5),  

#     Dense(out_dim=10)
# ]

'''
SpatialDropout2D(prob=0.2) :
    Epoch 20/100 - Loss: 0.9103 | Val Loss: 1.3004, Val Acc: 54.10%
    Epoch 19/100 - Loss: 0.9384 | Val Loss: 1.3102, Val Acc: 55.30%

'''
layers1 = [
    Conv2D(in_nchannel=3, out_nchannel=32, kernel_size=3, padding=1),
    BatchNorm2D(),
    LeakyReLU(),
    SpatialDropout2D(prob=0.3),
    MaxPool2D(dim=2, stride=2),

    Flatten(),

    Dense(out_dim=64),
    BatchNorm(),
    LeakyReLU(),
    Dropout(prob=0.4),  

    Dense(out_dim=10)
]


