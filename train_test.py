import torch.nn as nn
import torch.optim as optim

from architecture import PANNAcceptor
from data_reader import load_data


def train_model(model: PANNAcceptor, data):
    optimizer = optim.Adam(model.parameters(), lr=.01)
    loss_function = nn.CrossEntropyLoss()
    model.train()

    for i, (xs, ys, lengths) in enumerate(data):
        optimizer.zero_grad()
        output = model(xs, lengths)

        # Compute accuracy and loss
        accuracy = (output.argmax(dim=-1) == ys).sum() / len(ys)
        loss = loss_function(output, ys)

        # Optimizer
        loss.backward()
        optimizer.step()

        # Log
        if i % 100 == 0:
            print("Batch {}: acc={:.3f}, loss={:.3f}"
                  "".format(i, accuracy, loss))


def test_model(model: PANNAcceptor, data, dev: bool = False):
    loss_function = nn.CrossEntropyLoss()
    model.eval()

    num_correct = 0
    total_loss = 0
    num_samples = 0
    for i, (xs, ys, lengths) in enumerate(data):
        output = model(xs, lengths)
        num_correct += (output.argmax(dim=-1) == ys).sum()
        total_loss += loss_function(output, ys)
        num_samples += len(xs)

    accuracy = num_correct / num_samples
    avg_loss = total_loss / num_samples
    print("{}: acc={:.3f}, loss={:.3f}".format("Dev" if dev else "Test",
                                               accuracy, avg_loss))


if __name__ == "__main__":
    model = PANNAcceptor(4, 8)
    train_data = load_data("data/ea_train_short.txt", batch_size=10)
    train_model(model, train_data)
