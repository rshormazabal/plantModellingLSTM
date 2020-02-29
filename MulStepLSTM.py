import time
import matplotlib.dates as md
import datetime as dt
import pickle

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from datasets import plantDataSeqToSeq
from networks import LSTM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: wrap the training in one function so I can try different parameters and targets
training = False

if training:
    dataset = plantDataSeqToSeq(out_seq_dim=1, residence_time=720, data_amount=1, target='Col-b')
else:
    with open('dataset-colb.pickle', 'rb') as input_file:
        dataset = pickle.load(input_file)

train_set, val_set = torch.utils.data.random_split(dataset,
                                                   [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
# TODO: Find a better way to split data sequentially
train_set.indices = list(range(0, 5000))
val_set.indices = list(range(5000, 6000))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Create network
input_dim = 55
hidden_dim = 100

model = LSTM(input_dim, hidden_dim, layer_dim=1, output_dim=56, bidirectional=True, dropout=0)
model = model.to(device)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#####################
# Train model
#####################

num_epochs = 50
train_loss_history = []
val_loss_history = []
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

        # if i != 0:
        #    print([x.max().item() for x in model.lstm.all_weights[0]])
        #    print([x.max().item() for x in model.lstm.all_weights[1]])
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
    print("Epoch:{}, Train loss: {}, Validation loss: {}".format(epoch, train_loss, val_loss))
    if epoch >= 1:
        val_change = val_loss_history[epoch] - val_loss_history[epoch - 1]
        print("Time previous epoch: {} seconds".format(time.time() - time_now))
        print("Validation loss change previous epoch: {} ".format(val_change))

#####################
# Long term predictions
#####################
val_set.indices = list(range(5000, 5150))
val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)
val_sequences = list(val_loader)[0][0].to(device)
prediction = model(val_sequences.float())

prediction_sequence = []
for timestep in range(144):
    val_sequences = val_sequences[:, 1:, :].double()
    prediction = prediction[:, :-1].view(-1, 1, 55).double()

    val_sequences = torch.cat([val_sequences, prediction], 1)
    prediction = model(val_sequences.float())

    prediction_sequence.append(prediction.cpu().detach().numpy())

prediction_sequence = np.array(prediction_sequence)
real_sequence = list(val_loader)[0][1][:144, -1]

# compare labels and predictions
plt.scatter(prediction_sequence[:, 0, -1], real_sequence.view(-1))
plt.show()

# loss history
plt.scatter(list(range(num_epochs)), val_loss_history)
plt.show()

# timestep in py datetime
y_predict_timestep = list(map(lambda x: dataset.get_timestamp(x), val_set.indices))
y_train_timestep = list(map(lambda x: dataset.get_timestamp(x), train_set.indices))

datenums = md.date2num(y_predict_timestep)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=25)
ax = plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.plot(datenums[3000:5000], y_pred[3000:5000], label='Prediction')
plt.plot(datenums[3000:5000], y_real[3000:5000], label='Real data')
plt.legend()
plt.show()



pickle.dump(dataset, open("dataset-colb.pickle", "wb"))
# pickle.load(open("dataset-colb.pickle", "rb"))

#torch.save(model, 'last_model.pt')
#model = torch.load('last_model.pt')