from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from numerai.dataset import get_dataset, get_features
from numerai.model import NumeraiModel, create_loss_fn
from numerai.utils import compute_target_weight

device = "cuda" if torch.cuda.is_available() else "mps"
batch_size = 4096
version = "4.3"
collection = "medium"


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
    get_dataset(split="train", version=version, collection=collection, device=device),
    batch_size=batch_size,
    shuffle=True,
)
validation_loader = torch.utils.data.DataLoader(
    get_dataset(
        split="validation", version=version, collection=collection, device=device
    )
    .shuffle()
    .select(range(10240)),
    batch_size=batch_size,
    shuffle=False,
)

model = NumeraiModel(features=get_features(version=version, collection=collection)).to(
    device=device
)


# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 5, device=device)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 0, 3, 4], device=device)

print(dummy_outputs)
print(dummy_labels)

targets_weights = compute_target_weight({"target"})
print(targets_weights)

loss_fn = create_loss_fn(targets_weights, device=device)
loss = loss_fn(dummy_outputs, dummy_labels)
print("Total loss for this batch: {}".format(loss.item()))

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs = data
        labels = data["target"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 500 == 499:
            last_loss = running_loss / 500  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = "runs/fashion_trainer_{}".format(timestamp)
writer = SummaryWriter(logdir)
epoch_number = 0

EPOCHS = 15

best_vloss = 1_000_000.0

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs = vdata
            vlabels = vdata["target"]

            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch_number + 1,
    )
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_{}_{}".format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
