import numpy as np

SEED = 10

class AMSGrad:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.v_max = {}

        self.t = 0

    def update_lr(self,rate) :
        self.lr = self.lr * rate
    
    def update(self, params, grads):
        self.t += 1
        
        beta1_t = self.beta1**self.t  
        beta2_t = self.beta2**self.t 
        one_minus_beta1_t = (1.0 - beta1_t)
        one_minus_beta2_t = (1.0 - beta2_t)

        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)
                self.v_max[i] = np.zeros_like(g)

            self.m[i] *= self.beta1
            self.m[i] += (1.0 - self.beta1) * g

            self.v[i] *= self.beta2
            self.v[i] += (1.0 - self.beta2) * (g * g)

            np.maximum(self.v_max[i], self.v[i], out=self.v_max[i])

            m_hat = self.m[i] / one_minus_beta1_t
            v_hat = self.v_max[i] / one_minus_beta2_t

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def im2col(X, k):
 
    N, C, H, W = X.shape
    outH = H - k + 1
    outW = W - k + 1

    shape = (N, C, outH, outW, k, k)
    strides = (
        X.strides[0],  # stride over N
        X.strides[1],  # stride over C
        X.strides[2],  # stride down one row in H
        X.strides[3],  # stride right one col in W
        X.strides[2],  # stride down in the kernel
        X.strides[3]   # stride right in the kernel
    )

    patches = np.lib.stride_tricks.as_strided(
        X, shape=shape, strides=strides
    )
    
    col = patches.reshape(N * outH * outW, C * k * k)
    return col

def col2im(col, X_shape, k):
   
    N, C, H, W = X_shape
    outH = H - k + 1
    outW = W - k + 1

    grad_out = np.zeros(X_shape, dtype=col.dtype)
    idx = 0
    for n in range(N):
        for i in range(outH):
            for j in range(outW):
                
                patch_grad = col[idx].reshape(C, k, k)
                grad_out[n, :, i:i+k, j:j+k] += patch_grad
                idx += 1
    return grad_out


def load_data(csv_file):
    
    data = np.loadtxt(csv_file, delimiter=',')
    labels = data[:, 0].astype(int)   
    raw_features = data[:, 1:].astype(float)
    features = raw_features.reshape(-1, 3, 32, 32)
    num_classes = 10
    one_hot_labels = np.eye(num_classes)[labels - 1]
    return features, one_hot_labels

def get_X_from_csv(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', dtype=str)
    raw_features = data[:, 1:].astype(float)
    features = raw_features.reshape(-1, 3, 32, 32)

    return features


def generate_predictions_and_save(model, test_data, output_file="output.txt"):
   
    predictions = model.predict(test_data)
    
    predictions_adjusted = predictions + 1

    with open(output_file, "w") as f:
        for pred in predictions_adjusted:
            f.write(f"{pred}\n")
    
    print(f"Predictions saved to {output_file}")

