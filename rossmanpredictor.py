import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Normalization, CategoryEncoding, StringLookup
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Data preparation
##- Loads three CSV files: train.csv, store.csv, and test.csv
##- Merges store information with both training and test data using Store ID
##- Filters out closed stores from training data
##- Returns the merged training and test dataframes
def load_data():
    ross_df = pd.read_csv('train.csv', low_memory=False)
    store_df = pd.read_csv('store.csv')
    test_df = pd.read_csv('test.csv')
    
    merged_df = ross_df.merge(store_df, how='left', on='Store')
    merged_test_df = test_df.merge(store_df, how='left', on='Store')
    
    # Remove closed stores from training data
    merged_df = merged_df[merged_df.Open == 1].copy()
    
    return merged_df, merged_test_df

def prepare_features(df, is_training=True):
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract the date features
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['IsWeekend'] = df.Date.dt.dayofweek.isin([5, 6]).astype(int)
##    - Converts date strings to datetime objects
##    - Extracts temporal features: year, month, day, week of year
##    - Creates weekend indicator (1 for Sat/Sun, 0 otherwise)
    
    # Competition features
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + \
                           (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df.CompetitionOpen.map(lambda x: 0 if x < 0 else x).fillna(0)
    
    # Fill missing CompetitionDistance with large value
    max_distance = df.CompetitionDistance.max()
    df['CompetitionDistance'] = df.CompetitionDistance.fillna(max_distance)
##    - Calculates months since competition opened
##    - Handles negative values and missing data
##    - Fills missing competition distances with maximum observed distance
    
    # Promotion features
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) + \
                       (df.WeekOfYear - df.Promo2SinceWeek) * 7/30.5
    df['Promo2Open'] = df.Promo2Open.map(lambda x: 0 if x < 0 else x).fillna(0) * df.Promo2
##    - Calculates duration of Promo2 in months
##- Adjusts for week-based timing
##- Combines with Promo2 status

    return df

'''
Defines a deep feedforward neural network with:
    Hidden layers with 256, 512, 256, and 128 neurons.
    SELU activation for stability.
    Batch normalization and dropout for regularization.
    L2 regularization to prevent overfitting.
'''
def create_model(train_data, test_data):
    # Define numeric and categorical columns
    numeric_features = [
        'Store', 'DayOfWeek', 'Day', 'Month', 'Year', 'WeekOfYear',
        'CompetitionDistance', 'CompetitionOpen', 'Promo', 'Promo2',
        'Promo2Open', 'IsWeekend'
    ]
    
    categorical_features = ['StateHoliday', 'StoreType', 'Assortment']
    
    # Create normalization layers for numeric features
    normalizer = {}
    numeric_inputs = {}
    all_numeric_inputs = []
    
    for feature in numeric_features:
        numeric_inputs[feature] = tf.keras.Input(shape=(1,), name=feature)
        normalizer[feature] = Normalization(axis=None)
        # Reshape data for adaptation
        normalizer[feature].adapt(tf.cast(train_data[feature].values.reshape(-1, 1), tf.float32))
        all_numeric_inputs.append(normalizer[feature](numeric_inputs[feature]))
    
    # Create encoding layers for categorical features
    categorical_inputs = {}
    all_categorical_inputs = []

##    - Creates separate input for each numeric feature
##- Applies normalization layer to each feature
##- Adapts normalization parameters to training data
    
    for feature in categorical_features:
        # Convert inputs to string type for consistent handling
        vocabulary = np.unique(np.concatenate([train_data[feature].astype(str), 
                                             test_data[feature].astype(str)]))
        
        categorical_inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype='string')
        
        # Create lookup and encoding layers
        lookup = StringLookup(vocabulary=vocabulary, output_mode='int')
        encoding = CategoryEncoding(num_tokens=len(vocabulary) + 1, output_mode='one_hot')
        
        # Apply lookup and encoding
        encoded = encoding(lookup(categorical_inputs[feature]))
        all_categorical_inputs.append(encoded)
##        - Creates vocabulary from both train and test data
##        - Implements string lookup for categorical variables
##        - Applies one-hot encoding
    
    # Combine all inputs
    all_features = tf.keras.layers.concatenate(all_numeric_inputs + all_categorical_inputs)
    
    # Define the neural network architecture
    x = Dense(256, activation='selu', kernel_regularizer=l2(0.01))(all_features)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(512, activation='selu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(256, activation='selu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='selu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1, activation='relu')(x)
    
    # Create model with all inputs
    model = tf.keras.Model(
        inputs=list(numeric_inputs.values()) + list(categorical_inputs.values()),
        outputs=outputs
    )
    
    # Compile model
    #updating the model’s weights during training by minimizing the loss function
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='mse',
                 metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def prepare_input_data(df, numeric_features, categorical_features):
    # Prepare numeric data
    numeric_data = [df[feature].values.reshape(-1, 1) for feature in numeric_features]
    # Prepare categorical data
    categorical_data = [df[feature].astype(str).values.reshape(-1, 1) for feature in categorical_features]
    return numeric_data + categorical_data

def train_model():
    # Load data
    train_df, test_df = load_data()
    
    # Prepare features
    train_df = prepare_features(train_df)
    test_df = prepare_features(test_df, is_training=False)
    
    # Define feature columns
    numeric_features = [
        'Store', 'DayOfWeek', 'Day', 'Month', 'Year', 'WeekOfYear',
        'CompetitionDistance', 'CompetitionOpen', 'Promo', 'Promo2',
        'Promo2Open', 'IsWeekend'
    ]
    categorical_features = ['StateHoliday', 'StoreType', 'Assortment']
    
    # Create model
    model = create_model(train_df, test_df)
    
    # Prepare input data
    train_inputs = prepare_input_data(train_df, numeric_features, categorical_features)
    test_inputs = prepare_input_data(test_df, numeric_features, categorical_features)
    
    # Prepare target data (log transform)
    y_train = np.log1p(train_df['Sales'].values)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    ]
    
    # Train model
    history = model.fit(
        train_inputs, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(test_inputs)
    final_predictions = np.expm1(predictions)  # Transform back from log scale
    
    # Create submission
    submission_df = pd.DataFrame({
        'Id': test_df['Id'],
        'Sales': final_predictions.flatten()
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print('\nSubmission file created.')

if __name__ == "__main__":
    train_model()
