from denoise2 import denoise_unknown_noise
from feature_extraction import *
import torch
import random

# set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

def segment_list(lst, segment_size=8, overlap=4):
    """
    Splits a list into segments of a specified size, with each segment starting
    overlap elements after the start of the previous segment.

    Parameters:
    - lst: The list to be segmented.
    - segment_size: The size of each segment (default is 100).
    - overlap: The number of elements to skip between the start of each segment (default is 30).

    Returns:
    - A list of lists, where each sublist is a segment of the original list.
    """
    segments = []
    start = 0
    while start < len(lst):
        end = start + segment_size
        segments.append(lst[start:end])
        start += segment_size - overlap
    return segments

# function to determine arc length
def arc(y, z):
    dy = np.diff(y)
    dz = np.diff(z)
    distances = np.sqrt(dy ** 2 + dz ** 2)
    totalArcLength = np.sum(distances)
    return totalArcLength

#import preprocessed data as dataframes
df1 = importData('0805segment/f1d_segmented.csv')
df2 = importData('0805segment/f2d_segmented.csv')
df3 = importData('0805segment/f3d_segmented.csv')
df4 = importData('0723/col34D.csv')
df5 = importData('0723/col31D.csv')

#seperate reps (one set per file)
grouped1 = df1.groupby('Source')
grouped2 = df2.groupby('Source')
grouped3 = df3.groupby('Source')
grouped4 = df4.groupby('Set #') #"Set #" should be "Rep #" - was mislabeled
grouped5 = df5.groupby('Set #')

allReps = []
allTestIndices = []
allValidationIndices = []

# Put data into more useful form
# Ensure all sets are represented when randomly choosing reps to put into the test set
for i in range(1,11): #10 reps in set 1 -> 1 test rep and 1 validation rep
    allReps.append(grouped1.get_group(f'set #{i}'))
set1_test_indices = random.sample(range(0, 8 + 1), 2)
allTestIndices.append(set1_test_indices[0])
allValidationIndices.append(set1_test_indices[1])

for i in range(1,10): #9 reps in set 2 -> 1 test rep and 1 validation rep
    allReps.append(grouped2.get_group(f'set #{i}'))
set2_test_indices = random.sample(range(10, 18 + 1), 2)
allTestIndices.append(set2_test_indices[0])
allValidationIndices.append(set2_test_indices[1])

for i in range(1,9): #8 reps in set 3 -> 1 test rep and 1 validation rep
    allReps.append(grouped3.get_group(f'set #{i}'))
set3_test_indices = random.sample(range(19, 26 + 1), 2)
allTestIndices.append(set3_test_indices[0])
allValidationIndices.append(set3_test_indices[1])

for i in range(1,10): #9 reps in set 4 -> 1 test rep and 1 validation rep
    allReps.append(grouped4.get_group(f'set #{i}'))
set4_test_indices = random.sample(range(27, 35 + 1), 2)
allTestIndices.append(set4_test_indices[0])
allValidationIndices.append(set4_test_indices[1])

for i in range(1,11): #10 reps in set 5 -> 1 test rep and 1 validation rep
    allReps.append(grouped5.get_group(f'set #{i}'))
set5_test_indices = random.sample(range(38, 45 + 1), 2)
allTestIndices.append(set5_test_indices[0])
allValidationIndices.append(set5_test_indices[1])

#transform EMG + hand coordinate data into features + approx joint angle
repFeatures = []
repArcPercent = []
for rep_df in allReps:

    #extract emg data (emg1 and emg2 record the same reps but at slightly different points on the muscle)
    rawEmg1 = rep_df['emg_1'].values
    rawEmg2 = rep_df['emg_2'].values
    #denoise emg data
    emg1 = denoise_unknown_noise(rawEmg1, 200, lowpassCutoff=None)
    emg2 = denoise_unknown_noise(rawEmg2, 200, lowpassCutoff=None)
    #prepare emg "chunks" for feature extraction
    emg1 = segment_list(emg1)
    emg2 = segment_list(emg2)
    #extract features
    features1 = collect_features_2(emg1)
    features2 = collect_features_2(emg2)

    #combine feature arrays from two emg recordings
    features = features1 + features2
    print(f"{len(features)} x {len(features[0])}")
    features = np.array(features).T

    repFeatures.append(features)


    # get total path length hand takes when performing rep (front raise)
    # rep (front raise) is assumed to only vary meaningfully in the YZ plane
    # +z direction = up
    # +y direction = dorsoventral (if subject was to walk strictly forwards only their y coordinate would increase)
    y = rep_df['y'].values
    z = rep_df['z'].values
    totalArcLength = arc(y, z)

    # get percentage total path length completed with respect to time
    # this will be interpreted as "percentage of rep completed"
    arcs = []
    for i in range(len(y)):
        partialArc = arc(y[:i], z[:i])
        arcs.append(100 * partialArc / totalArcLength)

    # "chunk" percentage data (required due to the EMG features chosen)
    percentage_chunks = segment_list(arcs)
    percentage_chunks = [np.mean(e) for e in percentage_chunks]

    repArcPercent.append(percentage_chunks)


trainingDataIndices = list(range(len(repFeatures)))

badDataRepsIndices = [9, 36, 37]
indicesNotInTrainingData = allTestIndices + allValidationIndices + badDataRepsIndices
indicesNotInTrainingData.sort(reverse=True)
for idx in indicesNotInTrainingData:
    trainingDataIndices.pop(idx)

random.shuffle(trainingDataIndices)
random.shuffle(allValidationIndices)
random.shuffle(allTestIndices)

#shuffle data while retaining features-percentage pairs
trainX = [repFeatures[idx] for idx in trainingDataIndices]
trainY = [repArcPercent[idx] for idx in trainingDataIndices]

validationX = [repFeatures[idx] for idx in allValidationIndices]
validationY = [repArcPercent[idx] for idx in allValidationIndices]

testX = [repFeatures[idx] for idx in allTestIndices]
testY = [repArcPercent[idx] for idx in allTestIndices]

#========================
#   Structure of sets
#========================
# train/validation/testX = [rep_1, ..., rep_n]
# rep_i = [featuresFromChunk_1, ..., featuresFromChunk_m]
# featuresFromChunk_i = [feature_1, ..., feature_96]
# feature_i = double
#
# train/validation/testY = [rep_1, ..., rep_n]
# rep_i = [percentageRepCompleted_1, ... percentageRepCompleted_m]
# percentageRepCompleted_i = double

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

# object that can be used as an input to the pytorch model
class FeatureSetDataset(Dataset):
    # Input is:
    # (train/validation/testX = [rep_1, ..., rep_n], train/validation/testY = [rep_1, ..., rep_n])
    def __init__(self, reps, expected_outputs):

        # Normalize features using sklearn (z-score normalized)
        scaler = StandardScaler()
        normalized_reps = []
        for rep in reps:
            #normalize the features using means/standard deviations of all features in all of the rep
            #normalize each rep with features only from that rep
            normalized_rep = scaler.fit_transform(rep)
            #normalized_rep is still a list of feature arrays
            normalized_reps.append(normalized_rep)

        #convert to list of tensors
        self.reps = [torch.tensor(rep, dtype=torch.float32) for rep in normalized_reps]
        #(self.reps)
        self.expected_outputs = [torch.tensor(rep_outputs, dtype=torch.float32) for rep_outputs in expected_outputs]

    def __len__(self):
        return len(self.reps)

    def __getitem__(self, idx):
        return self.reps[idx], self.expected_outputs[idx], idx  # idx serves as a rep identifier

# slightly altered base pytorch GRU cell to include zoneout
class ZoneoutGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, zoneout_prob):
        super(ZoneoutGRUCell, self).__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.zoneout_prob = zoneout_prob

    def forward(self, input, previous_hidden_state=None):

        new_hidden_state = self.gru_cell(input, previous_hidden_state)

        if self.training:
            # creates a tensor the same shape as the output but with only values of zoneout_prob
            # bernoulli replaces zoneout_prob with 1 or 0 based on zoneout_prob
            # weird looking code but just creates a tensor that is (zoneout_prob)*100% 1's and (1 - zoneout_prob)*100% 0's
            mask = torch.bernoulli(torch.full_like(new_hidden_state, self.zoneout_prob))

            # The mask determines which parts of the previous hidden state to keep and which parts to update:
            # - mask * hidden_state: Keeps the values from the previous hidden state wherever mask == 1
            # - (1 - mask) * new_hidden_state: Updates the remaining values with the new hidden state wherever mask == 0
            # applies zoneout by selectively preserving hidden state features from the previous timestep
            zoned_out_hidden_state = mask * previous_hidden_state + (1 - mask) * new_hidden_state
        else:
            # during regression, apply a deterministic combination of hidden_state and new_hidden_state
            # - self.zoneout_prob * hidden_state: Takes zoneout_prob proportion of the previous hidden state
            # - (1 - self.zoneout_prob) * new_hidden_state: Takes the remaining proportion from the new hidden state
            zoned_out_hidden_state = self.zoneout_prob * previous_hidden_state + (1 - self.zoneout_prob) * new_hidden_state

        return zoned_out_hidden_state

class GRURNN(nn.Module):
    def __init__(self, input_size=92, hidden_size=100, num_layers=2, output_size=1, zoneout_prob=0.25, dropout_prob=0.35):
        super(GRURNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create separate ZoneoutGRUCell layers
        self.gru_cells = nn.ModuleList([
            ZoneoutGRUCell((input_size if i == 0 else hidden_size), hidden_size, zoneout_prob) for i in range(num_layers)
        ])

        # Layer normalization for each layer except the last
        self.layer_normalization_layers = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers - 1)
        ])

        # Dropout layers for each layer except the last
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout_prob) for _ in range(num_layers - 1)
        ])

        #ReLU layer
        self.relu = nn.ReLU()

        #final fully connected layer
        self.fully_connected_layer = nn.Linear(hidden_size, output_size)

    def forward(self, feature_array, hidden=None):
        # initialize a hidden state of zeros at very first forward pass
        # - tensor of all zeros
        # - shape of tensor is (num_layers, hidden_size)
        if hidden is None:
            hidden = self.init_hidden()

        new_hidden = []

        # hidden_seq is now hidden_state_sequence
        hidden_state_sequence = []

        previous_output = feature_array
        #pass current feature array through all layers
        for i, gru_cell in enumerate(self.gru_cells):

            # sequence of hidden states
            # - hidden_state_sequence[0] corresponds to hidden state output from first GRU cell, etc
            h = gru_cell(previous_output, hidden[i])
            hidden_state_sequence.append(h)

            #initialize output
            output = h

            # Apply LayerNorm, ReLU, and Dropout between layers, except after the last layer
            if i < self.num_layers - 1:
                output = self.layer_normalization_layers[i](output)
                output = self.relu(output)
                output = self.dropout_layers[i](output)

            previous_output = output

        # Stack new hidden states
        hidden_state_sequence = torch.stack(hidden_state_sequence, dim=0)
        #print(hidden_state_sequence)

        final_output = self.fully_connected_layer(previous_output)
        #print(final_output)
        return final_output, hidden_state_sequence

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.hidden_size)

# function to get the gradients for tracking
def gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train_model(model, dataloader, optimizer, criterion, scheduler, num_epochs, device):
    model.to(device)

    # initialize lists to collect training metrics across epochs
    all_predictions = []
    all_targets = []
    losses = []
    gradients = []
    learning_rates = []
    validation_loss = []

    # load validation set
    validationDataLoader = DataLoader(FeatureSetDataset(validationX, validationY), batch_size=1, shuffle=True)

    # training loop
    for epoch in range(num_epochs):
        model.train()

        # initialize loss and gradients
        total_loss = 0
        grad_norms = np.zeros(len(dataloader))

        # loads one full rep at a time
        for rep, expected_outputs, rep_id in dataloader:

            #clean up dataloader output
            rep = rep.squeeze()
            expected_outputs = expected_outputs.squeeze()
            rep, expected_outputs = rep.to(device), expected_outputs.to(device)

            #initialize optimizer
            optimizer.zero_grad()

            # reset the hidden state for each rep
            # - makes sure reps are independent of each other
            hidden_state = None

            outputs = []
            for i in range(rep.size(0)):  # Iterate over each feature array in the rep

                # feature_array is the list of features at timestep i
                feature_array = rep[i]

                # forward step
                output, hidden = model(feature_array, hidden_state)
                outputs.append(output)

                # get hidden state to inform next timestep's forward pass
                hidden_state = hidden.detach()

            # Calculate loss
            outputs = torch.stack(outputs).squeeze()
            loss = criterion(outputs, expected_outputs)

            # Add L2 regularization
            l2_lambda = 0.05
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm

            # Add L1 regularization
            l1_lambda = 0.0001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

            # backpropagate
            loss.backward()

            # clip gradients
            # - determined through observation that a gradient greater than 450 has exploded
            if epoch > 35:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=450)  # max_norm is the threshold

            # log gradients to be displayed after training
            grad_norm = gradient_norm(model)
            grad_norms[rep_id.item()] = grad_norm

            # advance Adam optimizer and learning rate scheduler
            optimizer.step()
            scheduler.step()

            # log this timestep's loss to be plotted/printed
            total_loss += loss.item()

            # log this timestep's predictions + targets to be plotted
            all_predictions.append(outputs.detach().numpy())
            all_targets.append(expected_outputs.detach().numpy())

        # log all gradients
        gradients.append(grad_norms)

        # log the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

        # calculate and print average loss
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average training Loss: {avg_loss:.4f}')

        # perform validation and collect validation loss for plotting
        # - validation data printed in the evaluate_model function
        validation_set_loss, _, _, _, _ = evaluate_model(model, validationDataLoader, criterion, device=device, validation=True)
        validation_loss.append(validation_set_loss)

    # clean up all_predictions to match the shape of all_targets
    all_predictions = [arr.squeeze() for arr in all_predictions]

    return all_predictions, all_targets, losses, learning_rates, np.array(gradients).T, validation_loss

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, dataloader, criterion, device, validation):
    model.eval()  # Set the model to evaluation mode

    # initialize metric tracking
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # No need to track gradients
        for rep, expected_outputs, rep_id in dataloader:

            #clean up dataloader output
            rep = rep.squeeze()
            expected_outputs = expected_outputs.squeeze()
            rep, expected_outputs = rep.to(device), expected_outputs.to(device)

            #initialize optimizer
            optimizer.zero_grad()

            # reset the hidden state for each rep
            # - makes sure reps are independent of each other
            hidden_state = None

            outputs = []
            for i in range(rep.size(0)):  # Iterate over each feature array in the rep

                # feature_array is the list of features at timestep i
                feature_array = rep[i]

                # forward step
                output, hidden = model(feature_array, hidden_state)
                outputs.append(output)

                # get hidden state to inform next timestep's forward pass
                hidden_state = hidden.detach()

            # Calculate loss
            outputs = torch.stack(outputs).squeeze()
            loss = criterion(outputs, expected_outputs)
            total_loss += loss.item()

            # log this timestep's predictions + targets to be plotted
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(expected_outputs.cpu().numpy())

    # flatten all_targets and all_predictions to allow further calculations
    flat_predictions = np.concatenate(all_predictions).ravel()
    flat_targets = np.concatenate(all_targets).ravel()

    # calculate evaluation metrics
    mae = mean_absolute_error(flat_targets, flat_predictions)
    r2 = r2_score(flat_targets, flat_predictions)

    # print metrics
    if validation:
        print(f'Validation Loss: {total_loss/len(dataloader):.4f}')
        print(f'Validation Mean Absolute Error: {mae:.4f}')
        print(f'Validation R2 Score: {r2:.4f}')

        # put model back in training mode after validation
        model.train()
    else:
        print(f'Test Loss: {total_loss:.4f}')
        print(f'Test Mean Absolute Error: {mae:.4f}')
        print(f'Test R2 Score: {r2:.4f}')

    return total_loss, mae, r2, all_predictions, all_targets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# put training data into form model can use
dataset = FeatureSetDataset(trainX, trainY)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # batch_size=1 because each item is already a set

# initialize hyperparameters
input_size = 46 * 2  # Number of features
hidden_size = 450
num_layers = 3
output_size = 1 # regression
num_epochs = 400

# initialize model with hyperparameters
model = GRURNN(input_size, hidden_size, num_layers, output_size)

# set loss function to mean squared error
criterion = nn.MSELoss()
# set optimizer to Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def xavier_init(m): # xavier function to initialize weights
    if type(m) == nn.GRU:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
# initialize weights
model.apply(xavier_init)

from torch.optim.lr_scheduler import OneCycleLR

# initialize learning rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.003,  # The peak learning rate
    steps_per_epoch=len(dataloader),
    epochs=num_epochs,
    pct_start=0.3,
    anneal_strategy='cos',
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25.0,
    final_div_factor=10000.0,
    three_phase=False,
    last_epoch=-1,
    verbose=False
)

# train model
training_predictions, training_targets, training_losses, learning_rates, all_grads, validation_loss = train_model(model, dataloader, optimizer, criterion, scheduler, num_epochs=num_epochs, device=device)

# put test data into form model can use
test_set = FeatureSetDataset(testX, testY)
testDataloader = DataLoader(test_set, batch_size=1, shuffle=True)  # batch_size=1 because each item is already a set

# test the model!!
test_loss, test_mae, test_r2, test_predictions, test_targets = evaluate_model(model, testDataloader, criterion, device=device, validation=False)

# plot output and training metrics
# flatten training predictions/targets for plotting
flat_training_predictions = np.concatenate(training_predictions).ravel()
flat_training_targets = np.concatenate(training_targets).ravel()

# plot all training targets + predictions
# - can manually zoom into areas of interest
plt.subplot(2, 3, 1)
plt.plot(flat_training_predictions)
plt.plot(flat_training_targets)
plt.title(f'Training Predictions vs Targets')

# flatten training predictions/targets for plotting
flat_test_predictions = np.concatenate(test_predictions).ravel()
flat_test_targets = np.concatenate(test_targets).ravel()

# plot test targets + predictions
plt.subplot(2, 3, 2)
plt.plot(flat_test_predictions)
plt.plot(flat_test_targets)
plt.title(f'Test Predictions vs Targets')

# plot training loss
plt.subplot(2, 3, 3)
plt.plot(training_losses)
plt.title(f'Average Training Loss')

# plot learning rate
plt.subplot(2, 3, 4)
plt.plot(learning_rates)
plt.title(f'Learning Rate')

# plot all gradients
for grad in all_grads:
    plt.subplot(2, 3, 5)
    plt.plot(grad)

# plot validation loss
plt.subplot(2, 3, 6)
plt.plot(validation_loss)
plt.title('Average Validation Loss')

plt.show()