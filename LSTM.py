import time
import matplotlib.dates as md
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from datasets import plantData
from networks import LSTM
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: wrap the training in one function so I can try different parameters and targets

# Load data
dataset = plantData(out_seq_dim=1, residence_time=720, data_amount=1, target='Col-b')
train_set, val_set = torch.utils.data.random_split(dataset,
                                                   [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
# TODO: Find a better way to split data sequentially
train_set.indices = list(range(50000))
val_set.indices = list(range(50000, len(dataset)))

train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Create network
input_dim = 55
hidden_dim = 100

model = LSTM(input_dim, hidden_dim, layer_dim=1, output_dim=1, bidirectional=True)
model = model.to(device)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

#####################
# Train model
#####################
num_epochs = 100
train_loss_history = []
val_loss_history = []
early_stop_count = 0

for epoch in range(num_epochs):
    time_now = time.time()
    train_loss = 0
    val_loss = 0
    for i, (sequences, labels) in enumerate(train_loader):
        # set train mode
        model.train()

        # send data to device
        sequences = sequences.to(device)
        labels = labels.to(device).float()
        # change sequence batch to accumulate gradient
        sequences = sequences.float().requires_grad_()

        # clear gradient w.r.t parameters
        optimizer.zero_grad()

        outputs = model(sequences)

        # Calculate Loss: MSE Loss
        loss = criterion(outputs, labels)
        train_loss += loss
        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

    for (val_sequences, val_labels) in val_loader:
        model.eval()
        # send data to device
        val_sequences = val_sequences.to(device)
        val_labels = val_labels.to(device)

        prediction = model(val_sequences.float())
        # Calculate Loss: MSE Loss
        loss = criterion(prediction, val_labels)
        val_loss += loss

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)

    # saving loss history
    train_loss_history.append(train_loss.item())
    val_loss_history.append(val_loss.item())

    # print stats
    print("Epoch:{}, \n Train loss: {}, \n Validation loss: {}".format(epoch, train_loss, val_loss))
    if epoch >= 1:
        val_change = val_loss_history[epoch] - val_loss_history[epoch - 1]
        if val_change > 0:
            early_stop_count += 1
            if early_stop_count == 100:
                print('Early Stop')
                break
        else:
            early_stop_count = 0
        print("Time previous epoch: {} seconds".format(time.time() - time_now))
        print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        print("Validation loss change previous epoch: {} ".format(val_change))
        print("-------------------------------------------------")
    #scheduler.step(val_loss)

y_pred, y_real = np.array([]), np.array([])

for (val_sequences, val_labels) in val_loader:
    model.eval()
    # send data to device
    val_sequences = val_sequences.to(device)
    val_labels = val_labels.to(device)

    prediction = model(val_sequences.float())

    # for graph
    y_pred = np.append(y_pred, prediction.view(-1).cpu().detach())
    y_real = np.append(y_real, val_labels.cpu().detach())

# timestep in py datetime
y_predict_timestep = list(map(lambda x: dataset.get_timestamp(x), val_set.indices))

datenums = md.date2num(y_predict_timestep)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=25)
ax = plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(datenums, y_pred, label='Prediction')
plt.plot(datenums, y_real, label='Real data')
plt.legend()
plt.show()

# compare labels and predictions
plt.scatter(y_real, y_pred)
plt.show()

# loss history
plt.scatter(list(range(num_epochs)), val_loss_history)
plt.show()
plt.savefig('results.png')

# Checking train data
y_pred_train, y_real_train = np.array([]), np.array([])

for (train_sequences, train_labels) in train_loader:
    model.eval()
    # send data to device
    train_sequences = train_sequences.to(device)
    train_labels = train_labels.to(device)

    prediction = model(train_sequences.float())

    # for graph
    y_pred_train = np.append(y_pred_train, prediction.view(-1).cpu().detach())
    y_real_train = np.append(y_real_train, train_labels.cpu().detach())

y_train_timestep = list(map(lambda x: dataset.get_timestamp(x), train_set.indices))

datenums = md.date2num(y_train_timestep)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=25)
ax = plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.scatter(datenums, y_pred_train, label='Prediction')
plt.scatter(datenums, y_real_train, label='Real data')
plt.legend()
plt.show()

pickle.dump(dataset, open("dataset-colb-720.pickle", "wb"))
# pickle.load(open("dataset-colb.pickle", "rb"))

torch.save(model, 'colb-720s-35e-.pt')
# model = torch.load('last_model.pt')

# save to json
timestamp = dataset.data.loc[dataset.valid_sequences[50000:], 'Time'].dt.strftime('%B %d, %Y, %r').tolist()
timestamp_train = dataset.data.loc[dataset.valid_sequences[:50000], 'Time'].dt.strftime('%B %d, %Y, %r').tolist()

results = {'test_set': {'name': 'Sequence magical-720m test-set', 'timestamp': timestamp,
                        'y_pred': list(y_pred), 'y_real': list(y_real),
            'train_set': {'name': 'Sequence720m magical-train-set', 'timestamp': timestamp_train,
                          'y_pred_train': list(y_pred_train), 'y_real_train': list(y_real_train)}}}

# saving json files
import json

json.dump(results, open('data/results/magical-colb-720-35.json', "w"))
