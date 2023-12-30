import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


#Importing the data and giving each column a name
hass_features = pd.read_csv("features.csv", names=["state", "Time", "entity_id"], delimiter=';')



#Time normalisation
hass_features['Time'] = pd.to_datetime(hass_features['Time'], unit='s')
hours = hass_features['Time'].dt.hour.astype(str)
minutes = hass_features['Time'].dt.minute.astype(str)
#Cleanup
hass_features['timestamp'] = hours+'.'+minutes #pd.to_datetime(hass_features['Time'], unit='s') # Sitas gali sukelt problemu
hass_features.drop('Time', axis=1, inplace=True)


hass_targets = pd.read_csv("targets.csv", names=["state", "Time", "entity_id"], delimiter=';')

#Time normalisation
hass_targets['Time'] = pd.to_datetime(hass_targets['Time'], unit='s')
hours = hass_targets['Time'].dt.hour.astype(str)
minutes = hass_targets['Time'].dt.minute.astype(str)
#Cleanup
hass_targets['timestamp'] = hours+'.'+minutes
hass_targets.drop('Time', axis=1, inplace=True)

#konvertuojam just to be safe
hass_features['timestamp'] = pd.to_numeric(hass_features['timestamp'])
hass_targets['timestamp'] = pd.to_numeric(hass_targets['timestamp'])

#Scaler, probs useless, but eh fuck it
###
#scaler = MinMaxScaler(feature_range=(0,24))
#hass_features['timestamp'] = scaler.fit_transform(hass_features[['timestamp']])
#hass_targets['timestamp'] = scaler.transform(hass_targets[['timestamp']])
###

#Printing out for debugging

print(hass_features.head())
print(hass_targets.head())

######################################################################################
#                             Slyksti Dalis (Reformatting)                           #
######################################################################################

############################################|- Features -|##################################
import pandas as pd
import numpy as np

# Assuming hass_features and hass_targets are already read from CSV files

# Convert 'timestamp' to numeric for both DataFrames
hass_features['timestamp'] = pd.to_numeric(hass_features['timestamp'], errors='coerce')
hass_targets['timestamp'] = pd.to_numeric(hass_targets['timestamp'], errors='coerce')

# Remove duplicates
hass_features = hass_features.drop_duplicates(subset=['timestamp', 'entity_id'], keep='last')
hass_targets = hass_targets.drop_duplicates(subset=['timestamp', 'entity_id'], keep='last')

# Create a unified timestamp array
all_timestamps = np.union1d(hass_features['timestamp'].unique(), hass_targets['timestamp'].unique())

# Pivot both DataFrames
pivoted_features = hass_features.pivot(index='timestamp', columns='entity_id', values='state')
pivoted_targets = hass_targets.pivot(index='timestamp', columns='entity_id', values='state')

pivoted_features = pivoted_features.apply(pd.to_numeric, errors='coerce', axis=1)
pivoted_targets = pivoted_targets.apply(pd.to_numeric, errors='coerce', axis=1)

# Reindex both pivoted DataFrames to include all timestamps
pivoted_features = pivoted_features.reindex(all_timestamps)
pivoted_targets = pivoted_targets.reindex(all_timestamps)

# Fill missing values

pivoted_features.fillna(method='ffill', inplace=True)
pivoted_features.fillna(0, inplace=True)
pivoted_targets.fillna(method='ffill', inplace=True)
pivoted_targets.fillna(0, inplace=True)

# Reset index to add timestamp back as a column
pivoted_features.reset_index(inplace=True)
pivoted_targets.reset_index(inplace=True)

pivoted_features.to_csv("features_array.csv", sep=";", header=True, index=True)
pivoted_targets.to_csv("targets_array.csv", sep=";", header=True, index=True)

# Print shapes for verification
print("Features DataFrame Shape:", pivoted_features.shape)
print("Targets DataFrame Shape:", pivoted_targets.shape)



######################################################################################
#                            Training the Neural Network                             #
######################################################################################

#May be useful, but it would defeat the purpose of the dataset
def find_closest_timestamp(feature_row, target_df):
    entity_id = feature_row['entity_id']
    timestamp = feature_row['timestamp']
    filtered_targets = target_df[target_df['entity_id'] == entity_id]
    closest_row = filtered_targets.iloc[(filtered_targets['timestamp'] - timestamp).abs().argsort()[:1]]
    return closest_row

X_train, X_test, y_train, y_test = train_test_split(pivoted_features, pivoted_targets, test_size=0.1, random_state=42)
print(X_train,"\n\n",X_test)
# Building the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape the data for LSTM layer
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train_reshaped, y_train, batch_size=1, epochs=20)

# Evaluate the model
test_loss = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {test_loss}")
