import torch

def compute_loss(model, loader, loss_fn=torch.nn.CrossEntropyLoss(),
                 use_cuda=True, n_batch=100):
    total_loss = 0
    count = 0
    with torch.no_grad():
        for data in loader:
            count += 1
            if count < n_batch:
                inputs, labels = data
                if use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                preds = model(inputs)
                total_loss += loss_fn(preds.squeeze(), labels).item()
            
    return total_loss