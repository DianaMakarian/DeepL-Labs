import argparse
import numpy as np
import torch as tch
from torch import nn

from src.data import iris_data_preprocessing
from src.model import Model, TorchModel, cros_entropy


def sync_weights_to_torch(np_model: Model, torch_model: TorchModel):
    with tch.no_grad():
        torch_model.fc1.weight.copy_(tch.from_numpy(np_model.params["W1"].T))
        torch_model.fc1.bias.copy_(tch.from_numpy(np_model.params["b1"]))
        torch_model.fc2.weight.copy_(tch.from_numpy(np_model.params["W2"].T))
        torch_model.fc2.bias.copy_(tch.from_numpy(np_model.params["b2"]))


def run_pytorch_check(X_train, y_train, normalize=True):
    mode = "Правильна реалізація" if normalize else "Дослід з помилкою"
    print(f"\n === Звірка з PyTorch | {mode} === ")

    model = Model()
    logits_np, cache = model.forward(X_train)
    loss_np, dZ2 = cros_entropy(logits_np, y_train, normalize=normalize)
    grads_np = model.backward(dZ2, cache)

    torch_model = TorchModel().to(tch.float64)
    sync_weights_to_torch(model, torch_model)
    X_torch = tch.tensor(X_train, dtype=tch.float64)
    y_torch = tch.tensor(y_train, dtype=tch.long)
    logits_torch = torch_model(X_torch)
    criterion = nn.CrossEntropyLoss()
    loss_torch = criterion(logits_torch, y_torch)
    loss_torch.backward()

    loss_tch = loss_torch.item()
    grads_tch = {
        'W1': torch_model.fc1.weight.grad.numpy().T,
        'b1': torch_model.fc1.bias.grad.numpy(),
        'W2': torch_model.fc2.weight.grad.numpy().T,
        'b2': torch_model.fc2.bias.grad.numpy(),
    }

    diff_loss = np.abs(loss_np - loss_tch)
    print("| d       | numpy  | torch  |   diff    |  acc |")
    print(f"| loss    | {loss_np:.4f} | {loss_tch:.4f} | {diff_loss:.4e} | {diff_loss <= 1e-12} |")
    for key in grads_np.keys():
        max_np = grads_np[key].max()
        max_tch = grads_tch[key].max()
        diff = np.max(np.abs(grads_np[key] - grads_tch[key]))
        print(f"| d{key:<5} | {max_np:.4f} | {max_tch:.4f} | {diff:.4e} | {diff <= 1e-12} |")


def compute_numerical_derivative(model, key, idx, X, y, eps=1e-6):
    orig_val = model.params[key][idx]

    model.params[key][idx] = orig_val + eps
    logits_plus, _ = model.forward(X)
    loss_plus, _ = cros_entropy(logits_plus, y)

    model.params[key][idx] = orig_val - eps
    logits_minus, _ = model.forward(X)
    loss_minus, _ = cros_entropy(logits_minus, y)

    model.params[key][idx] = orig_val

    return (loss_plus - loss_minus) / (2 * eps)


def run_numerical_check(X_train, y_train, normalize=True):
    mode = "Правильна реалізація" if normalize else "Дослід з помилкою"
    print(f"\n === Чисельне диференціювання | {mode} === ")

    model = Model()
    logits_np, cache = model.forward(X_train)
    loss_np, dZ2 = cros_entropy(logits_np, y_train, normalize=normalize)
    grads_np = model.backward(dZ2, cache)

    targets = [("W1", (0, 0)),
                ("b1", (0,)),
                ("W2", (0, 0)),
                ("b2", (0,))]
    
    print(f"loss: {loss_np:.4f}")
    print("| param   | grad   | num_grad |   diff   |  acc |")
    for key, idx in targets:
        g_manual = grads_np[key][idx]
        g_num = compute_numerical_derivative(model, key, idx, X_train, y_train)
        diff = abs(g_manual - g_num)
        idx_str = f"{key}[{','.join(map(str, idx))}]"
        print(f"| {idx_str:<7} | {g_manual:.4f} | {g_num:.4f} | {diff:.4e} | {diff <= 1e-7} |")


def main():
    parser = argparse.ArgumentParser(description="ЛР1: Реалізація та перевірка зворотного поширення похибки")
    parser.add_argument("--error", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    X_train, y_train, _, _ = iris_data_preprocessing()

    if args.all:
        run_pytorch_check(X_train, y_train, normalize=True)
        run_numerical_check(X_train, y_train, normalize=True)
        run_pytorch_check(X_train, y_train, normalize=False)
        run_numerical_check(X_train, y_train, normalize=False)
    elif args.error:
        run_pytorch_check(X_train, y_train, normalize=False)
        run_numerical_check(X_train, y_train, normalize=False)
    else:
        run_pytorch_check(X_train, y_train, normalize=True)
        run_numerical_check(X_train, y_train, normalize=True)

if __name__ == "__main__":
    main()