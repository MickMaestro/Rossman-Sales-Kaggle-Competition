# Rossmann Store Sales Forecasting

A deep learning solution for predicting sales across 1,115 Rossmann stores using TensorFlow and Keras. This project tackles the challenge of forecasting daily sales 6 weeks in advance, considering factors such as promotions, competition, holidays, and seasonality.

## Competition Overview

The [Rossmann Store Sales Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales) challenges participants to predict sales for Rossmann drug stores across Germany. The dataset includes historical sales data for 1,115 stores, along with information about promotions, competition, school holidays, and store characteristics.

**Final Score: 0.4** (Root Mean Square Percentage Error)

## Dataset

The solution uses three main data files:
- `train.csv` - Historical sales data with target values
- `test.csv` - Store/day combinations to predict
- `store.csv` - Store metadata including type, assortment, and competition information

## Architecture

### Neural Network Design
- **Input Layer**: Handles both numerical and categorical features
- **Hidden Layers**: 4-layer feedforward architecture (256 → 512 → 256 → 128 neurons)
- **Activation**: SELU (Scaled Exponential Linear Unit) for self-normalising properties
- **Regularisation**: 
  - Batch normalisation after each hidden layer
  - Dropout (0.3 → 0.3 → 0.2 → 0.2)
  - L2 regularisation (0.01) on all hidden layers
- **Output**: Single neuron with ReLU activation for sales prediction

### Key Features

#### Temporal Feature Engineering
- Date decomposition (year, month, day, week of year)
- Day of week encoding
- Weekend indicator
- Seasonal patterns capture

#### Competition Analysis
- Months since competition opened
- Competition distance handling (missing values filled with maximum distance)
- Competition impact modelling

#### Promotional Features
- Promo2 duration calculation
- Historical promotion tracking
- Combined promotional effects

#### Data Preprocessing
- Categorical encoding using StringLookup and one-hot encoding
- Numerical feature normalisation
- Log transformation of target variable
- Closed store filtering

## Implementation Details

### Model Training
- **Optimizer**: RMSprop with learning rate 0.001
- **Loss Function**: Mean Squared Error
- **Batch Size**: 64
- **Epochs**: 10 (with early stopping)
- **Validation Split**: 20%

### Callbacks
- **Early Stopping**: Patience of 3 epochs on validation loss
- **Learning Rate Reduction**: Factor 0.2 with patience of 2 epochs

### Feature Categories
**Numerical Features:**
- Store ID, Day of Week, Date components
- Competition distance and duration
- Promotion indicators
- Weekend flag

**Categorical Features:**
- State Holiday type
- Store Type (a, b, c, d)
- Assortment level (basic, extra, extended)

## Usage

```python
# Run the complete pipeline
python rossmanpredictor.py
```

The script will:
1. Load and merge the datasets
2. Engineer temporal and business features
3. Create and train the neural network
4. Generate predictions for the test set
5. Export results to `submission.csv`

## Requirements

```
pandas
numpy
tensorflow>=2.0
scikit-learn
```

## Model Performance

The model achieves a **0.4 RMSPE** on the Kaggle leaderboard, demonstrating effective sales forecasting across diverse store types and market conditions.

### Key Strengths
- Robust handling of categorical variables
- Comprehensive temporal feature engineering
- Effective regularisation preventing overfitting
- Business-aware feature creation (competition, promotions)

## Files Structure

```
├── rossmanpredictor.py    # Main implementation
├── train.csv              # Training data
├── test.csv               # Test data
├── store.csv              # Store metadata
├── submission.csv         # Generated predictions
└── README.md              # This file
```

## Future Improvements

- Ensemble methods combining multiple architectures
- Advanced time series features (rolling averages, lag variables)
- Hyperparameter optimisation using Bayesian methods
- Cross-validation for more robust model evaluation
- Feature selection and importance analysis

## Licence

This project is open source and available under the [MIT Licence](LICENSE).
