

import numpy as np
from layers import BatchNorm, BatchNorm2D, Conv2D, Dense, Dropout, Flatten, LeakyReLU, MaxPool2D, Model, SoftmaxCrossEntropy
from utils import SEED, generate_predictions_and_save, get_X_from_csv, load_data
from layer_arch import layers2,layers1


def main():
    
    np.random.seed(SEED)
    
    X, Y = load_data('train.csv')
    X_validate, Y_validate = load_data('validate.csv')
    
    model = Model(
        layers=layers1,
        learn_rate=0.001
    )
    
    best_w = model.fit(X, Y, X_validate, Y_validate,epochs=100,batch_size=64)
    if best_w is not None :
        model.set_weights(best_w)
        
    test_X = get_X_from_csv("test.csv")
    generate_predictions_and_save(model,test_X)
    
if __name__ == "__main__":
    main()