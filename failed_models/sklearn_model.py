import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Step 1: Load and prepare the data
df = pd.read_csv("preprocessed_player_stats_Trial3.csv")

# Set the threshold directly
X_value = 20.0  # Replace with the value of X we are interested in
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

# Handle missing values
df.fillna(0, inplace=True)

# Step 2: Group-based Cross-Validation for Model Selection and Hyperparameter Tuning
X = df.drop(columns=["Over_X_Kills", "Match_ID"])
y = df["Over_X_Kills"]
groups = df["Match_ID"]

# Define the parameter grids
param_grid_xgb = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}

param_grid_lgbm = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127],  # Adding num_leaves parameter
}

param_grid_lr = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}

param_grid_svc = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "max_iter": [10000]}

param_grid_knn = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

# Create the models with GridSearchCV
xgb_model = GridSearchCV(
    XGBClassifier(tree_method="hist", gpu_id=0, random_state=42),
    param_grid_xgb,
    cv=3,
    n_jobs=-1,
)
lgbm_model = GridSearchCV(
    LGBMClassifier(device="gpu", random_state=42), param_grid_lgbm, cv=3, n_jobs=-1
)
lr_model = GridSearchCV(
    LogisticRegression(solver="liblinear"), param_grid_lr, cv=3, n_jobs=-1
)
svc_model = GridSearchCV(SVC(probability=True), param_grid_svc, cv=3, n_jobs=-1)
knn_model = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=3, n_jobs=-1)

# Ensemble model for soft voting
soft_voting_model = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("lgbm", lgbm_model),
        ("lr", lr_model),
        ("svc", svc_model),
        ("knn", knn_model),
    ],
    voting="soft",
)

# Group-based cross-validation for soft voting
group_kfold = GroupKFold(n_splits=5)
soft_cv_scores = cross_val_score(soft_voting_model, X, y, cv=group_kfold, groups=groups)

print(f"Soft Voting - Cross-validation scores: {soft_cv_scores}")
print(f"Soft Voting - Mean cross-validation score: {soft_cv_scores.mean()}")

# Train-Test Split for Final Model Evaluation (Group-based)
train_idx, test_idx = next(group_kfold.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train the final soft voting model
soft_voting_model.fit(X_train, y_train)

# Evaluate the final soft voting model
y_pred_soft = soft_voting_model.predict(X_test)

accuracy_soft = accuracy_score(y_test, y_pred_soft)
report_soft = classification_report(y_test, y_pred_soft)

print(f"Soft Voting - Accuracy: {accuracy_soft}")
print(f"Soft Voting - Classification Report:\n{report_soft}")

# Step 3: Stacking Method
stacking_estimators = [
    ("xgb", xgb_model),
    ("lgbm", lgbm_model),
    (
        "lr",
        make_pipeline(
            StandardScaler(),
            GridSearchCV(
                LogisticRegression(solver="liblinear"), param_grid_lr, cv=3, n_jobs=-1
            ),
        ),
    ),
    (
        "svc",
        make_pipeline(
            StandardScaler(),
            GridSearchCV(SVC(probability=True), param_grid_svc, cv=3, n_jobs=-1),
        ),
    ),
    (
        "knn",
        make_pipeline(
            StandardScaler(),
            GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=3, n_jobs=-1),
        ),
    ),
]

stacking_model = StackingClassifier(
    estimators=stacking_estimators, final_estimator=LogisticRegression()
)

# Group-based cross-validation for stacking
stacking_cv_scores = cross_val_score(
    stacking_model, X, y, cv=group_kfold, groups=groups
)

print(f"Stacking - Cross-validation scores: {stacking_cv_scores}")
print(f"Stacking - Mean cross-validation score: {stacking_cv_scores.mean()}")

# Train the final stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the final stacking model
y_pred_stacking = stacking_model.predict(X_test)

accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
report_stacking = classification_report(y_test, y_pred_stacking)

print(f"Stacking - Accuracy: {accuracy_stacking}")
print(f"Stacking - Classification Report:\n{report_stacking}")


# Step 4: Making Predictions
def prepare_input(game_mode, player_team, enemy_team, player, teammates, enemies):
    input_data = {feature: 0 for feature in features}

    # Set game mode
    input_data[f"Mode_{game_mode}"] = 1

    # Set player team and enemy team
    input_data[f"PlayerTeam_{player_team}"] = 1
    input_data[f"EnemyTeam_{enemy_team}"] = 1

    # Set player
    input_data[f"{player}_Player"] = 1

    # Set teammates
    for teammate in teammates:
        input_data[f"{teammate}_Teammate"] = 1

    # Set enemies
    for enemy in enemies:
        input_data[f"{enemy}_Enemy"] = 1

    # Convert to DataFrame for prediction with correct feature names
    input_df = pd.DataFrame([input_data])
    return input_df


# Example usage
player = "aBeZy"
game_mode = "HardPoint"  # 'HardPoint', 'Search_and_Destroy', 'Control'
player_team = "Atlanta_FaZe"
enemy_team = "Miami_Heretics"
teammates = ["Cellium", "Drazah", "Simp"]
enemies = ["ReeaL", "Vikul", "MettalZ", "Lucky"]

input_data = prepare_input(
    game_mode, player_team, enemy_team, player, teammates, enemies
)

# Prediction using soft voting model
prediction_soft = soft_voting_model.predict(input_data)
print(
    f'Soft Voting Prediction: {"Over" if prediction_soft[0] == 1 else "Under"} {X_value} kills'
)

# Prediction using stacking model
prediction_stacking = stacking_model.predict(input_data)
print(
    f'Stacking Prediction: {"Over" if prediction_stacking[0] == 1 else "Under"} {X_value} kills'
)
