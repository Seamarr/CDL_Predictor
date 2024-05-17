import pandas as pd
import cudf
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.metrics import accuracy_score as cu_accuracy_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import classification_report


# Load and prepare the data
df = pd.read_csv("preprocessed_player_stats_Trial3.csv")
df = cudf.DataFrame.from_pandas(df)

# Set the threshold directly
X_value = 24.0  # Replace with the value of X we are interested in
df["Over_X_Kills"] = df["Kills"].apply(lambda x: 1 if x > X_value else 0)

# Select relevant features (excluding 'Kills')
features = [
    "Mode_Control",
    "Mode_HardPoint",
    "Mode_Overall",
    "Mode_Search_and_Destroy",
    "PlayerTeam_Atlanta_FaZe",
    "PlayerTeam_Boston_Breach",
    "PlayerTeam_Carolina_Royal_Ravens",
    "PlayerTeam_Las_Vegas_Legion",
    "PlayerTeam_Los_Angeles_Guerrillas",
    "PlayerTeam_Los_Angeles_Thieves",
    "PlayerTeam_Miami_Heretics",
    "PlayerTeam_New_York_Subliners",
    "PlayerTeam_OpTic_Texas",
    "PlayerTeam_Seattle_Surge",
    "PlayerTeam_Toronto_Ultra",
    "EnemyTeam_Atlanta_FaZe",
    "EnemyTeam_Boston_Breach",
    "EnemyTeam_Carolina_Royal_Ravens",
    "EnemyTeam_Las_Vegas_Legion",
    "EnemyTeam_Los_Angeles_Guerrillas",
    "EnemyTeam_Los_Angeles_Thieves",
    "EnemyTeam_Miami_Heretics",
    "EnemyTeam_New_York_Subliners",
    "EnemyTeam_OpTic_Texas",
    "EnemyTeam_Seattle_Surge",
    "EnemyTeam_Toronto_Ultra",
    "04_Player",
    "04_Teammate",
    "04_Enemy",
    "aBeZy_Player",
    "aBeZy_Teammate",
    "aBeZy_Enemy",
    "Abuzah_Player",
    "Abuzah_Teammate",
    "Abuzah_Enemy",
    "Afro_Player",
    "Afro_Teammate",
    "Afro_Enemy",
    "Arcitys_Player",
    "Arcitys_Teammate",
    "Arcitys_Enemy",
    "Asim_Player",
    "Asim_Teammate",
    "Asim_Enemy",
    "Assault_Player",
    "Assault_Teammate",
    "Assault_Enemy",
    "Attach_Player",
    "Attach_Teammate",
    "Attach_Enemy",
    "Beans_Player",
    "Beans_Teammate",
    "Beans_Enemy",
    "Breszy_Player",
    "Breszy_Teammate",
    "Breszy_Enemy",
    "Cellium_Player",
    "Cellium_Teammate",
    "Cellium_Enemy",
    "Clayster_Player",
    "Clayster_Teammate",
    "Clayster_Enemy",
    "CleanX_Player",
    "CleanX_Teammate",
    "CleanX_Enemy",
    "Dashy_Player",
    "Dashy_Teammate",
    "Dashy_Enemy",
    "Diamondcon_Player",
    "Diamondcon_Teammate",
    "Diamondcon_Enemy",
    "Drazah_Player",
    "Drazah_Teammate",
    "Drazah_Enemy",
    "Envoy_Player",
    "Envoy_Teammate",
    "Envoy_Enemy",
    "Estreal_Player",
    "Estreal_Teammate",
    "Estreal_Enemy",
    "Fame_Player",
    "Fame_Teammate",
    "Fame_Enemy",
    "FelonY_Player",
    "FelonY_Teammate",
    "FelonY_Enemy",
    "Flames_Player",
    "Flames_Teammate",
    "Flames_Enemy",
    "Ghosty_Player",
    "Ghosty_Teammate",
    "Ghosty_Enemy",
    "Gio_Player",
    "Gio_Teammate",
    "Gio_Enemy",
    "Gwinn_Player",
    "Gwinn_Teammate",
    "Gwinn_Enemy",
    "Huke_Player",
    "Huke_Teammate",
    "Huke_Enemy",
    "HyDra_Player",
    "HyDra_Teammate",
    "HyDra_Enemy",
    "Insight_Player",
    "Insight_Teammate",
    "Insight_Enemy",
    "JoeDeceives_Player",
    "JoeDeceives_Teammate",
    "JoeDeceives_Enemy",
    "Kenny_Player",
    "Kenny_Teammate",
    "Kenny_Enemy",
    "KiSMET_Player",
    "KiSMET_Teammate",
    "KiSMET_Enemy",
    "Kremp_Player",
    "Kremp_Teammate",
    "Kremp_Enemy",
    "Lucky_Player",
    "Lucky_Teammate",
    "Lucky_Enemy",
    "MettalZ_Player",
    "MettalZ_Teammate",
    "MettalZ_Enemy",
    "Nastie_Player",
    "Nastie_Teammate",
    "Nastie_Enemy",
    "Nero_Player",
    "Nero_Teammate",
    "Nero_Enemy",
    "Pentagrxm_Player",
    "Pentagrxm_Teammate",
    "Pentagrxm_Enemy",
    "Pred_Player",
    "Pred_Teammate",
    "Pred_Enemy",
    "Priestahh_Player",
    "Priestahh_Teammate",
    "Priestahh_Enemy",
    "Purj_Player",
    "Purj_Teammate",
    "Purj_Enemy",
    "ReeaL_Player",
    "ReeaL_Teammate",
    "ReeaL_Enemy",
    "Scrap_Player",
    "Scrap_Teammate",
    "Scrap_Enemy",
    "Shotzzy_Player",
    "Shotzzy_Teammate",
    "Shotzzy_Enemy",
    "Sib_Player",
    "Sib_Teammate",
    "Sib_Enemy",
    "Simp_Player",
    "Simp_Teammate",
    "Simp_Enemy",
    "Skyz_Player",
    "Skyz_Teammate",
    "Skyz_Enemy",
    "SlasheR_Player",
    "SlasheR_Teammate",
    "SlasheR_Enemy",
    "Snoopy_Player",
    "Snoopy_Teammate",
    "Snoopy_Enemy",
    "TJHaLy_Player",
    "TJHaLy_Teammate",
    "TJHaLy_Enemy",
    "Vikul_Player",
    "Vikul_Teammate",
    "Vikul_Enemy",
    "Seany_Player",
    "Seany_Teammate",
    "Seany_Enemy",
    "oJohnny_Player",
    "oJohnny_Teammate",
    "oJohnny_Enemy",
]

df = df[features + ["Over_X_Kills", "Match_ID"]]

# Ensure all boolean features are converted to float32
for col in features:
    df[col] = df[col].astype("float32")

# Handle missing values
df.fillna(0, inplace=True)

# Check correlation with target
correlations = df.to_pandas().corr()["Over_X_Kills"].sort_values(ascending=False)
print(correlations.head(10))
print(correlations.tail(10))

# Prepare data for training
X = df.drop(columns=["Over_X_Kills", "Match_ID"])
y = df["Over_X_Kills"]
groups = df["Match_ID"]

# Convert y and groups to int
y = y.astype("int32")
groups = groups.astype("int32")

# Group-based cross-validation
group_kfold = GroupKFold(n_splits=5)

# Define RandomForestClassifier
model = cuRF(n_estimators=100, random_state=42, n_streams=1)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}

# Convert param_grid values to int where necessary
param_grid = {
    key: [int(value) if value is not None else value for value in values]
    for key, values in param_grid.items()
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=group_kfold,
    scoring="accuracy",
    error_score="raise",
)
grid_search.fit(X.to_pandas(), y.to_pandas(), groups=groups.to_pandas())

# Print best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Use the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the final model
y_pred = best_model.predict(X)

accuracy = cu_accuracy_score(y, y_pred)
report = classification_report(y.to_pandas(), y_pred.to_pandas())

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")


# Making Predictions
def prepare_input(game_mode, player_team, enemy_team, player, teammates, enemies):
    input_data = {feature: 0 for feature in features}
    input_data[f"Mode_{game_mode}"] = 1
    input_data[f"PlayerTeam_{player_team}"] = 1
    input_data[f"EnemyTeam_{enemy_team}"] = 1
    input_data[f"{player}_Player"] = 1
    for teammate in teammates:
        input_data[f"{teammate}_Teammate"] = 1
    for enemy in enemies:
        input_data[f"{enemy}_Enemy"] = 1
    input_df = pd.DataFrame([input_data])
    input_df = input_df.astype("float32")
    return cudf.DataFrame.from_pandas(input_df)


# Example usage
player = "aBeZy"
game_mode = "HardPoint"
player_team = "Atlanta_FaZe"
enemy_team = "Miami_Heretics"
teammates = ["Cellium", "Drazah", "Simp"]
enemies = ["ReeaL", "Vikul", "MettalZ", "Lucky"]

input_data = prepare_input(
    game_mode, player_team, enemy_team, player, teammates, enemies
)
prediction = best_model.predict(input_data)

print(f'Prediction: {"Over" if prediction[0] == 1 else "Under"} {X_value} kills')
