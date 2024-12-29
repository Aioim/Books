import torch
def train(model, loss_fn, optimizer, X_train,y_train, device):
    size = len(X_train)
    model.train()
    model.to(device)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(zip(X_train,y_train)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss +=loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Train Error:, Avg loss: {train_loss:>8f} \n")
    
    
def test(net, lossFn,X_test,y_test, device):
    num_batches = len(y_test)
    net.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in zip(X_test,y_test):
            X, y = X.to(device), y.to(device)
            pred = net(X)  # pred此处可能的操作同上
            test_loss += lossFn(pred, y).item()
            # 获取pred每列的最大值的index与y比较
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )