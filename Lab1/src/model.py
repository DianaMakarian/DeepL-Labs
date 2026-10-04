import numpy as np
import torch as tch
import torch.nn as nn

# функція активації
def ReLU(Z):
    return np.maximum(0,Z)

def mask_ReLU(Z):
    return (Z>0)

# для перекладу логітів
# перетворення логітів у ймовірності
def log_softmax(logits, axis = -1) -> np.array:
    m = np.max(logits, axis=axis, keepdims=True)
    log_p = logits - m  - np.log(np.sum(np.exp(logits - m), axis=axis, keepdims=True))
    return log_p

# функція втрат - наскільки помилилася нейромережа
def cros_entropy(logits, y,  normalize=True):
    N = logits.shape[0]
    log_p = log_softmax(logits)
    loss =  -np.mean(log_p[np.arange(N), y])

    # також dz2, щоб не витрачати зайві обчсилення 
    probs = np.exp(log_p)
    dZ2 = probs.copy()
    dZ2[np.arange(N), y] -= 1

    if normalize:
        dZ2 /= N
    return loss , dZ2

def set_weight():
    ...

class Model():
    '''
    
    '''
    def __init__(self, seed: int = 0):

        rng = np.random.default_rng(seed)

        std_1 = np.sqrt(2.0/4)
        std_2 = np.sqrt(2.0/(8+3))

        W1 = rng.normal(0.0, std_1, size = (4,8))
        b1 = np.zeros(8)
        W2 = rng.normal(0.0, std_2, size = (8, 3))
        b2 = np.zeros(3)

        self.params =  {'W1': W1,
                        'b1': b1,
                        'W2': W2,
                        'b2':b2}


    def forward(self, X)-> tuple[np.array, dict]:
        '''
        повертає обчислені логіти Z2 та проміжні результати {'X': X, 'Z1':Z1, 'A1': A1}
        '''
        Z1 = X @ self.params['W1'] + self.params['b1']
        A1 = ReLU(Z1)
        Z2 = A1 @ self.params['W2'] + self.params['b2']

        cashe = {'X': X, 'Z1':Z1, 'A1': A1}
        return Z2, cashe

    def backward(self, dZ2, cache) -> dict:
        '''
        повертає словник з обчислених градієнтів для ваг та зсувів кожного шару
        
        {'W1':dW1 ,'b1': db1,'W2': dW2,'b2':db2}
        '''
        X = cache['X']
        Z1 = cache['Z1']
        A1 = cache['A1']

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.params['W2'].T
        dZ1 = dA1 * mask_ReLU(Z1)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis = 0)

        return {'W1':dW1 ,'b1': db1,'W2': dW2,'b2':db2}
        
class TorchModel(nn.Module):
    def __init__(self):

        super().__init__()

        # Афіінні шари з узгодженням формату двонарної точності
        self.fc1= nn.Linear(4,8, dtype=tch.float64)
        self.fc2 = nn.Linear(8,3, dtype=tch.float64)

        self.relu = nn.ReLU()

    def forward(self, x):
        Z1 = self.fc1(x)
        A1 = self.relu(Z1)
        Z2 = self.fc2(A1)
        # PyTorch сам зберігає проміжні результати, backward вже імплементований в батьківському класі
        return Z2