# %%
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import pickle

# %% [markdown]
# ### Supervised: Predicting Streams 

# %% [markdown]
# ### Step 1: Data Exploration & Step 2: Data Cleaning

# %%
# Read the data
music = pd.read_csv("spotify_dataset.csv")

# Convert to dataframe
df = pd.DataFrame(music)

# Data Exploration: Check the first 5 rows
df.head()

# %%
# Data Exploration: Check null data
df.isna().sum()

# %%
# Data Exploration: Check data type
df.info()

# %%
# Data Exploration: Which genre is the most common ones? 
# Create a list to contain all genre
total_genre = []

# Data Exploration: Split each genre lists as independent element  
df["Genre"] = df["Genre"].str.split(",")

# Data Exploration: Loop through each genre
for i, genre_list in enumerate(df["Genre"]):
    for single_genre in genre_list:
        cleaned_genre = single_genre.strip("['] ")
        # Store cleaned genre in the list
        total_genre.append(cleaned_genre)

# Data Exploration: Observe the most common genre
count_genre = Counter(total_genre)
print(count_genre.most_common())

# %%
# Data Cleaning: Convert streams to integer
df['Streams'] = (df['Streams'].replace(',','', regex = True)).astype(int)
print(df['Streams'])

# Data Cleaning: Convert release date into datetime
df["Release Date"] = df["Release Date"].str.strip()
df["Release Date"] = pd.to_datetime(df["Release Date"], format = "%Y-%m-%d")
df["Release Date"] = df["Release Date"].apply(lambda x: x.year)

# Data Cleaning: Convert other columns into numeric data
numeric_list = ["Artist Followers", "Popularity", "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Liveness", "Tempo", "Duration (ms)", "Valence"]
for i in numeric_list: 
    df[i] = pd.to_numeric(df[i], errors = "coerce")

# Data Cleaning: Create dummies for chord column
chord_dummy = pd.get_dummies(df["Chord"], prefix = "Chord")
df = df.join(chord_dummy)


# %%
# Data Cleaning: Initiate the dummy column as 0
df[["Rock", "Pop", "Hip Hop", "Rap", "Trap"]] = 0

# Data Cleaning: Loop through each genre
for i, genre_list in enumerate(df["Genre"]):
    for single_genre in genre_list:
        cleaned_genre = single_genre.strip("[']")
        # Update the value of dummy columns when matched
        if "rock" in cleaned_genre:
            df.loc[i, "Rock"] = 1 
        if "pop" in cleaned_genre:
            df.loc[i, "Pop"] = 1  
        if "hip hop" in cleaned_genre:
            df.loc[i, "Hip Hop"] = 1  
        if "trap" in cleaned_genre:
            df.loc[i, "Trap"] = 1 
        elif "rap" in cleaned_genre:
            df.loc[i, "Rap"] = 1 

# %%
# Data Cleaning: Drop columns that are not important or being replaced by dummies
cleaned_df = df.drop(["Week of Highest Charting", "Song ID", "Genre", "Weeks Charted", "Index", "Chord", "Song Name", "Artist"], axis = 1)

# Print the cleaned dataframe
cleaned_df 

# %%
# Data Exploration: Data Visualization
plt.hist(df["Streams"])
plt.xlabel("Streams")
plt.ylabel("Frequency")
plt.title("Streams frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Artist Followers"])
plt.xlabel("Number of Followers")
plt.ylabel("Frequency")
plt.title("Number of Followers frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Release Date"])
plt.xlabel("Release Date")
plt.ylabel("Frequency")
plt.title("Release Date frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Popularity"])
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.title("Year of Release frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Danceability"])
plt.xlabel("Danceability")
plt.ylabel("Frequency")
plt.title("Danceability frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Energy"])
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.title("Energy frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Loudness"])
plt.xlabel("Loudness")
plt.ylabel("Frequency")
plt.title("Loudness frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Speechiness"])
plt.xlabel("Speechiness")
plt.ylabel("Frequency")
plt.title("Speechiness frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Acousticness"])
plt.xlabel("Acousticness")
plt.ylabel("Frequency")
plt.title("Acousticness frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Liveness"])
plt.xlabel("Liveness")
plt.ylabel("Frequency")
plt.title("Liveness frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Tempo"])
plt.xlabel("Tempo")
plt.ylabel("Frequency")
plt.title("Tempo frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Duration (ms)"])
plt.xlabel("Duration (ms)")
plt.ylabel("Frequency")
plt.title("Duration (ms) frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Valence"])
plt.xlabel("Valence")
plt.ylabel("Frequency")
plt.title("Valence frequencies")

# %%
# Data Exploration: Data Visualization
plt.hist(df["Chord"])
plt.xlabel("Chord")
plt.ylabel("Frequency")
plt.title("Chord frequencies")

# %% [markdown]
# ### Step 3: Data Preprocessing

# %%
# Data Preprocessing: Scale the data except dummies, song name and artist
final_df = cleaned_df.copy()
scaled_feature = ["Number of Times Charted", "Streams", "Artist Followers", "Release Date", "Popularity", "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Liveness", "Tempo", "Duration (ms)", "Valence"]

# Use Standard Scaler to specific column
features = final_df[scaled_feature]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
final_df[scaled_feature] = features

# Drop NAs
final_df = final_df.dropna()
final_df

# %% [markdown]
# ### Step 4: Feature Selection

# %%
# Split the data into training group and testing group
random.seed(123)
x = final_df[["Highest Charting Position", "Number of Times Charted", "Artist Followers", "Release Date", "Popularity", "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Liveness", "Tempo", "Duration (ms)", "Valence", "Chord_ ", "Chord_A", "Chord_A#/Bb", "Chord_B", "Chord_C", "Chord_C#/Db", "Chord_D", "Chord_D#/Eb", "Chord_E", "Chord_F", "Chord_F#/Gb", "Chord_G", "Chord_G#/Ab", "Rock", "Pop", "Hip Hop", "Rap", "Trap"]]
y = final_df["Streams"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40)

# %%
# Feature Selection: Use CV to find the optimal alpha for LASSO
alphas = np.logspace(-4, 0, 100)

# Fit a Lasso regression model for each alpha value
model = LassoCV(alphas = alphas, cv = 5)
model.fit(x_train, y_train)

# Evaluate each model on the validation set using MSE and select the best MSE
mse_values = np.mean((model.predict(x_test) - y_test) ** 2, axis = 0)
best_alpha = model.alpha_
best_alpha

# %%
# LASSO Regression
model = Lasso(alpha = best_alpha)
model.fit(x_train, y_train)

# Validate with test set and get MSE
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

# Choose those coefficents != 0
lasso_x_train = x_train.iloc[:,model.coef_!=0]
print(lasso_x_train)
lasso_col = lasso_x_train.columns.tolist()
#print(lasso_col)

# %% [markdown]
# ### Step 5: Supervised Model Building: Prediction
# 
#     1. Linear Regression 
#     2. Tree 
#     3. Random Forest 
#     4. XGBoost 

# %%
# Extract Lasso selcted column from x_test
lasso_x_test = x_test[lasso_col]

# Linear Regression
linear_model = LinearRegression().fit(lasso_x_train , y_train)

# Cross Validation
cv_scores = cross_val_score(linear_model, lasso_x_train, y_train, cv=10)

# Get mean CV score
cv_scores_mean = np.mean(cv_scores)
print(cv_scores , "\n\n""mean =" ,"{:.2f}".format(cv_scores_mean))

# Save model
linear_filename = 'finalized_linear_model.sav'
pickle.dump(linear_model, open(linear_filename, 'wb'))

# Load the pickled model
linear_from_pickle = pickle.load(open(linear_filename, 'rb'))

# Use the loaded pickled model to make predictions
y_pred_linear = linear_from_pickle.predict(lasso_x_test)


# Use MSE to evaluate
linear_mse = mean_squared_error(y_pred_linear, y_test)
print("Linear Regression MSE:", linear_mse)

# Evaluate accuracy score
linear_score = linear_from_pickle.score(lasso_x_test , y_test)
print("Linear Regression Accuracy Score:", linear_score)

# %%
# Tree model
tree_model = DecisionTreeRegressor(random_state=44)
tree_model.fit(lasso_x_train, y_train)

# Cross Validation
cv_scores = cross_val_score(tree_model, lasso_x_train, y_train, cv=10)

# Get mean CV score
cv_scores_mean = np.mean(cv_scores)
print(cv_scores , "\n\n""mean =" ,"{:.2f}".format(cv_scores_mean))

# Save model
tree_filename = 'finalized_tree_model.sav'
pickle.dump(tree_model, open(tree_filename, 'wb'))

# Load the pickled model
tree_from_pickle = pickle.load(open(tree_filename, 'rb'))

# Use the loaded pickled model to make predictions
y_pred_tree = tree_from_pickle.predict(lasso_x_test)

# Use MSE to evaluate
tree_mse = mean_squared_error(y_pred_tree, y_test)
print("Tree MSE:", tree_mse)

# Evaluate accuracy score
tree_score = tree_from_pickle.score(lasso_x_test , y_test)
print("Tree Accuracy Score:", tree_score)

# %%
# Random Forest
rf = RandomForestRegressor(n_estimators = 1000, random_state = 40)

# Train the model on training data
rf.fit(lasso_x_train, y_train)

# Cross Validation
cv_scores = cross_val_score(rf, lasso_x_train, y_train, cv = 10)

# Get mean CV score
cv_scores_mean = np.mean(cv_scores)
print(cv_scores , "\n\n""mean =" ,"{:.2f}".format(cv_scores_mean))

# Save model
rf_filename = 'finalized_rf_model.sav'
pickle.dump(rf, open(rf_filename, 'wb'))

# Load the pickled model
rf_from_pickle = pickle.load(open(rf_filename, 'rb'))
  
# Use the loaded pickled model to make predictions
y_pred_rf = rf_from_pickle.predict(lasso_x_test)

# Use MSE to evaluate
rf_mse = mean_squared_error(y_pred_rf, y_test)
print("Random Forest MSE", rf_mse)

# Evaluate accuracy score
rf_accu = rf_from_pickle.score(lasso_x_test, y_test)
print("Random Forest Accuracy Score:", rf_accu)

# %%
# XGBoost
xgbr = xgb.XGBRegressor(booster = "gbtree", 
                        subsample = 0.8, 
                        eval_metric = 'rmse', 
                        max_depth = 5, 
                        objective = 'reg:squarederror',
                        verbosity = 0) 
xgbr.fit(lasso_x_train, y_train, eval_set = [(lasso_x_train, y_train), (lasso_x_test, y_test)], verbose = 100)

# Cross Validation
cv_scores = cross_val_score(xgbr, lasso_x_train, y_train, cv = 10)

# Get mean CV score
cv_scores_mean = np.mean(cv_scores)
print(cv_scores , "\n\n""mean =" ,"{:.2f}".format(cv_scores_mean))

# Save model
xgb_filename = 'finalized_xgb_model.sav'
pickle.dump(xgbr, open(xgb_filename, 'wb'))

# Load the pickled model
xgb_from_pickle = pickle.load(open(xgb_filename, 'rb'))

# Use the loaded pickled model to make predictions
y_pred_xgb = xgb_from_pickle.predict(lasso_x_test)

# Use MSE to evaluate
xgb_mse = mean_squared_error(y_pred_xgb, y_test)
print("XGB MSE:", xgb_mse)

# Evaluate accuracy score
xgb_accu = xgb_from_pickle.score(lasso_x_test, y_test)
print("XGB Accuracy Score:", xgb_accu)


