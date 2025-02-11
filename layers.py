import numpy as np
import copy
from utils import SEED, AMSGrad, col2im, im2col

LEARN_RATE = 0.001
np.random.seed(SEED)


class Model:
    def __init__(self, layers, learn_rate=LEARN_RATE):
        self.layers = layers
        self.loss_fn = SoftmaxCrossEntropy()
        self.learn_rate = learn_rate
        self.optimizer = AMSGrad(lr=self.learn_rate)
        self.output = None
        self.loss = None

    def forward(self, X, y=None, training=True):
        """
        Forward pass with a training flag.
        """
        out = X
        for layer in self.layers:
            # Set the training flag if the layer supports it.
            if hasattr(layer, 'training'):
                layer.training = training
            try:
                out = layer.forward(out, training=training)
            except TypeError:
                out = layer.forward(out)
        self.output = out
        if y is not None:
            self.loss = self.loss_fn.forward(out, y)
            return self.loss
        return out

    def backward(self):
        grad = self.loss_fn.backward(1.0)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, X_train, y_train, X_validate=None, y_validate=None,
            epochs=50, batch_size=64, patience=10):
        N = len(X_train)
        indices = np.arange(N)
        best_val_loss = float('inf')
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            self._shuffle_data(indices)
            self.update_lr(epoch)
            avg_loss = self._train_one_epoch(X_train, y_train, indices, batch_size)
            # Evaluate in inference mode (dropout off, BatchNorm using running stats)
            val_loss, val_acc = self.evaluate(X_validate, y_validate, batch_size=batch_size)
            self._log_epoch_results(epoch, epochs, avg_loss, val_loss, val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.get_weights())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                self.set_weights(best_weights)
                return best_weights

        print("Training completed.")
        self.set_weights(best_weights)
        return best_weights

    def _shuffle_data(self, indices):
        np.random.shuffle(indices)

    def update_lr(self, epoch):
        if epoch == 15:
            print("update 1")
            self.optimizer.update_lr(0.2)
        elif epoch == 20:
            print("update 2")
            self.optimizer.update_lr(0.2)
        # elif epoch >= 20:
        #     print("update")
        #     self.optimizer.update_lr(0.9)

    def _train_one_epoch(self, X_train, y_train, indices, batch_size):
        N = len(X_train)
        batch_losses = []
        for start_idx in range(0, N, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            loss = self.forward(X_batch, y_batch, training=True)
            batch_losses.append(loss)
            self.backward()
            params, grads = self.get_params_and_grads()
            self.optimizer.update(params, grads)
        return np.mean(batch_losses)

    def _log_epoch_results(self, epoch, epochs, avg_loss, val_loss, val_acc):
        log_message = (f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")
        print(log_message)

    def get_params_and_grads(self):
        all_params = []
        all_grads = []
        for layer in self.layers:
            if hasattr(layer, "parameters") and hasattr(layer, "grads"):
                all_params.extend(layer.parameters())
                all_grads.extend(layer.grads())
        return all_params, all_grads

    def predict(self, X, batch_size=32):
        N = len(X)
        outputs = []
        for start_idx in range(0, N, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            self.forward(X_batch, training=False)
            preds = np.argmax(self.output, axis=1)
            outputs.append(preds)
        return np.concatenate(outputs, axis=0)

    def evaluate(self, X, y, batch_size=32):
        N = len(X)
        total_loss = 0
        correct = 0
        for start_idx in range(0, N, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            loss_val = self.forward(X_batch, y_batch, training=False)
            total_loss += loss_val * len(X_batch)
            preds = np.argmax(self.output, axis=1)
            targets = np.argmax(y_batch, axis=1)
            correct += np.sum(preds == targets)
        avg_loss = total_loss / N
        accuracy = correct / N
        return avg_loss, accuracy

    def get_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                weights.append(layer.parameters())
        return weights

    def set_weights(self, weights):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "set_parameters"):
                layer.set_parameters(weights[idx])
                idx += 1


class Layer:
    def __init__(self):
        self.cache_X = None

    def forward(self, X):
        self.cache_X = X  

    def backward(self, grad_in):
        raise NotImplementedError


class BatchNorm2D(Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, X, training=True):
        super().forward(X)
        N, C, H, W = X.shape
        if self.gamma is None:
            self.gamma = np.ones((1, C, 1, 1))
        if self.beta is None:
            self.beta = np.zeros((1, C, 1, 1))
        if self.running_mean is None:
            self.running_mean = np.zeros((1, C, 1, 1))
        if self.running_var is None:
            self.running_var = np.ones((1, C, 1, 1))

        if training:
            batch_mean = X.mean(axis=(0, 2, 3), keepdims=True)
            batch_var = X.var(axis=(0, 2, 3), keepdims=True)
            self.X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            # In-place update of running statistics:
            self.running_mean *= self.momentum
            self.running_mean += (1 - self.momentum) * batch_mean
            self.running_var *= self.momentum
            self.running_var += (1 - self.momentum) * batch_var
            self.batch_mean = batch_mean
            self.batch_var = batch_var
            self.X = X
        else:
            self.X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * self.X_norm + self.beta

    def backward(self, grad_in):
        N, C, H, W = self.X.shape
        grad_gamma = (grad_in * self.X_norm).sum(axis=(0, 2, 3), keepdims=True)
        grad_beta = grad_in.sum(axis=(0, 2, 3), keepdims=True)
        self.grad_gamma = grad_gamma
        self.grad_beta = grad_beta
        grad_X_norm = grad_in * self.gamma
        ivar = 1.0 / np.sqrt(self.batch_var + self.epsilon)
        X_mu = self.X - self.batch_mean
        grad_var = np.sum(grad_X_norm * X_mu * (-0.5 * ivar**3), axis=(0, 2, 3), keepdims=True)
        grad_mean = np.sum(grad_X_norm * -ivar, axis=(0, 2, 3), keepdims=True)
        grad_mean += grad_var * np.sum(-2.0 * X_mu, axis=(0, 2, 3), keepdims=True) / (N * H * W)
        grad_X = (grad_X_norm * ivar)
        grad_X += (grad_var * 2.0 * X_mu / (N * H * W))
        grad_X += (grad_mean / (N * H * W))
        return grad_X

    def parameters(self):
        return [self.gamma, self.beta]

    def set_parameters(self, params):
        self.gamma, self.beta = params

    def grads(self):
        return [self.grad_gamma, self.grad_beta]


class Conv2D(Layer):
    def __init__(self, in_nchannel, out_nchannel, kernel_size, padding=0):
        self.in_nchannel = in_nchannel
        self.out_nchannel = out_nchannel
        self.kernel_size = kernel_size
        self.padding = padding
        fan_in = in_nchannel * kernel_size * kernel_size
        self.W = np.random.randn(out_nchannel, in_nchannel, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_nchannel, dtype=np.float32)
        self.dW = None
        self.db = None
        self.X_shape = None
        self.col = None

    def forward(self, X):
        super().forward(X)
        self.X_shape = X.shape
        N, C, H, W = self.X_shape
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X
        self.col = im2col(X_padded, self.kernel_size)
        W_2d = self.W.reshape(self.out_nchannel, -1).T
        out_2d = self.col @ W_2d
        out_2d += self.b  # Broadcasting self.b (shape (out_nchannel,)) to each row.
        outH = H + 2 * self.padding - self.kernel_size + 1
        outW = W + 2 * self.padding - self.kernel_size + 1
        out = out_2d.reshape(N, outH, outW, self.out_nchannel)
        out = out.transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_in):
        N, outC, outH, outW = grad_in.shape
        _, C, H, W = self.X_shape
        grad_2d = grad_in.transpose(0, 2, 3, 1).reshape(-1, outC)
        dW_2d = self.col.T @ grad_2d
        self.dW = dW_2d.reshape(self.W.shape)
        self.db = np.sum(grad_2d, axis=0)
        W_2d = self.W.reshape(self.out_nchannel, -1)
        dcol = grad_2d @ W_2d
        grad_out_padded = col2im(dcol, (N, C, H + 2 * self.padding, W + 2 * self.padding), self.kernel_size)
        if self.padding > 0:
            grad_out = grad_out_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_out = grad_out_padded
        return grad_out

    def parameters(self):
        return [self.W, self.b]

    def set_parameters(self, params):
        self.W, self.b = params

    def grads(self):
        return [self.dW, self.db]


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, X):
        super().forward(X)
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, grad_in):
        X = self.cache_X
        return np.where(X > 0, grad_in, self.alpha * grad_in)


class MaxPool2D(Layer):
    def __init__(self, dim=2, stride=2):
        self.dim = dim
        self.stride = stride
        self.max_indices = None
        self.cache_X = None

    def forward(self, X):
        super().forward(X)
        N, C, H, W = X.shape
        outH = (H - self.dim) // self.stride + 1
        outW = (W - self.dim) // self.stride + 1
        shape = (N, C, outH, outW, self.dim, self.dim)
        strides = (X.strides[0],
                   X.strides[1],
                   X.strides[2] * self.stride,
                   X.strides[3] * self.stride,
                   X.strides[2],
                   X.strides[3])
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        out = patches.max(axis=(4, 5))
        patches_reshaped = patches.reshape(N, C, outH, outW, -1)
        argmax_flat = patches_reshaped.argmax(axis=-1)
        local_row, local_col = np.unravel_index(argmax_flat, (self.dim, self.dim))
        # Broadcast indices to shape (N, C, outH, outW)
        oh_idx, ow_idx = np.indices((outH, outW))
        oh_idx = np.broadcast_to(oh_idx, (N, C, outH, outW))
        ow_idx = np.broadcast_to(ow_idx, (N, C, outH, outW))
        global_row = oh_idx * self.stride + local_row
        global_col = ow_idx * self.stride + local_col
        self.max_indices = np.stack([global_row, global_col], axis=-1)  # Shape: (N, C, outH, outW, 2)
        return out

    def backward(self, grad_in):
        X = self.cache_X
        N, C, H, W = X.shape
        _, _, outH, outW = grad_in.shape
        grad_out = np.zeros_like(X)
        rows = self.max_indices[..., 0].ravel()
        cols = self.max_indices[..., 1].ravel()
        grad_vals = grad_in.ravel()
        n_idx, c_idx, _, _ = np.indices((N, C, outH, outW))
        n_idx = n_idx.ravel()
        c_idx = c_idx.ravel()
        np.add.at(grad_out, (n_idx, c_idx, rows, cols), grad_vals)
        return grad_out


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        N = X.shape[0]
        return X.reshape(N, -1)

    def backward(self, grad_in):
        return grad_in.reshape(self.input_shape)


class Dense(Layer):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.W = None
        self.b = np.zeros(self.out_dim, dtype=np.float32)
        self.dW = None
        self.db = None

    def parameters(self):
        return [self.W, self.b]

    def set_parameters(self, params):
        self.W, self.b = params

    def grads(self):
        return [self.dW, self.db]

    def forward(self, X):
        super().forward(X)
        N, in_dim = X.shape
        if self.W is None:
            self.W = np.random.randn(in_dim, self.out_dim) * np.sqrt(2 / in_dim)
        return X.dot(self.W) + self.b.reshape(1, -1)

    def backward(self, grad_in):
        X = self.cache_X
        self.dW = X.T.dot(grad_in)
        self.db = np.sum(grad_in, axis=0)
        return grad_in.dot(self.W.T)


class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        self.cache_grad = None

    def forward(self, X, Y):
        super().forward(X)
        shift_x = X - np.max(X, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        sums = np.sum(exp_x, axis=1, keepdims=True)
        p = exp_x / sums
        N = X.shape[0]
        eps = 1e-15
        loss = -np.sum(Y * np.log(p + eps)) / N
        self.cache_grad = (p - Y) / N
        return loss

    def backward(self, grad_in=1.0):
        return self.cache_grad * grad_in


class Dropout(Layer):
    def __init__(self, prob=0.5):
        super().__init__()
        self.drop_prob = prob
        self.mask = None
        self.training = True

    def forward(self, X):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.drop_prob, size=X.shape).astype(np.float32)
            return (X * self.mask) / (1.0 - self.drop_prob)
        else:
            return X

    def backward(self, grad_in):
        if self.training and self.mask is not None:
            return grad_in * self.mask
        return grad_in


class BatchNorm(Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, X, training=True):
        self.X = X
        N, D = X.shape
        if self.gamma is None:
            self.gamma = np.ones((1, D))
        if self.beta is None:
            self.beta = np.zeros((1, D))
        if self.running_mean is None:
            self.running_mean = np.zeros((1, D))
        if self.running_var is None:
            self.running_var = np.ones((1, D))
        if training:
            batch_mean = np.mean(X, axis=0, keepdims=True)
            batch_var = np.var(X, axis=0, keepdims=True)
            self.X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            self.running_mean *= self.momentum
            self.running_mean += (1 - self.momentum) * batch_mean
            self.running_var *= self.momentum
            self.running_var += (1 - self.momentum) * batch_var
            self.batch_mean = batch_mean
            self.batch_var = batch_var
        else:
            self.X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * self.X_norm + self.beta

    def backward(self, grad_in):
        N, D = self.X.shape
        self.grad_gamma = np.sum(grad_in * self.X_norm, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_in, axis=0, keepdims=True)
        dx_hat = grad_in * self.gamma
        mean_dx_hat = np.mean(dx_hat, axis=0, keepdims=True)
        cross = np.mean(dx_hat * (self.X - self.batch_mean), axis=0, keepdims=True)
        inv_std = 1.0 / np.sqrt(self.batch_var + self.epsilon)
        dx = dx_hat - mean_dx_hat - (self.X - self.batch_mean) * (cross / (self.batch_var + self.epsilon))
        dx *= inv_std
        return dx

    def parameters(self):
        return [self.gamma, self.beta]

    def set_parameters(self, params):
        self.gamma, self.beta = params

    def grads(self):
        return [self.grad_gamma, self.grad_beta]


class SpatialDropout2D(Layer):
    def __init__(self, prob=0.5):
        super().__init__()
        self.drop_prob = prob
        self.mask = None
        self.training = True

    def forward(self, X):
        super().forward(X)
        if self.training:
            batch_size, channels, height, width = X.shape
         
            self.mask = np.random.binomial(1, 1 - self.drop_prob, 
                                           size=(batch_size, channels, 1, 1)).astype(np.float32)
            return (X * self.mask) / (1.0 - self.drop_prob)
        else:
            return X

    def backward(self, grad_in):
        if self.training and self.mask is not None:
            return grad_in * self.mask
        return grad_in