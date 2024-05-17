from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin


class PyTorchModelWrapperV1(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # PyTorch models are already trained
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X_tensor)
            return (predictions >= 0.5).float().numpy().ravel()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X_tensor)
            # Return probabilities for both classes (1-p and p)
            probabilities = torch.cat((1 - predictions, predictions), dim=1)
            return probabilities.numpy()


# Data loading and preparation


class ModelV1:
    def __init__(
        self,
        df,
        features,
        X_value=25,
        player="aBeZy",
        game_mode="HardPoint",
        player_team="Atlanta_FaZe",
        enemy_team="OpTic_Texas",
        teammates=["Gwinn", "TJHaLy", "Clayster"],
        enemies=["Shotzzy", "Kenny", "Pred", "Dashy"],
    ):
        self.model = None
        self.X_value = X_value
        self.player = player
        self.game_mode = game_mode
        self.player_team = player_team
        self.enemy_team = enemy_team
        self.teammates = teammates
        self.enemies = enemies
        self.df = df
        self.features = features

    def train_and_predict(self):
        df = self.df.copy()
        df["Over_X_Kills"] = df["Kills"].apply(lambda x: 1 if x > self.X_value else 0)
        X = df[self.features]
        y = df["Over_X_Kills"]

        # Remove constant features
        X = X.loc[:, (X != X.iloc[0]).any()]

        # Feature Selection
        selector = SelectKBest(score_func=f_classif, k=20)
        X_new = selector.fit_transform(X, y)

        # Save the selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y, test_size=0.2, random_state=42
        )

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        class NeuralNetwork1(nn.Module):
            def __init__(self):
                super(NeuralNetwork1, self).__init__()
                self.fc1 = nn.Linear(X_train.shape[1], 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                return x

        class NeuralNetwork2(nn.Module):
            def __init__(self):
                super(NeuralNetwork2, self).__init__()
                self.fc1 = nn.Linear(X_train.shape[1], 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                return x

        class NeuralNetwork3(nn.Module):
            def __init__(self):
                super(NeuralNetwork3, self).__init__()
                self.fc1 = nn.Linear(X_train.shape[1], 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 1)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                return x

        def train_model(model, X_train, y_train, num_epochs=20):
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 2 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Train the models
        model1 = NeuralNetwork1()
        model2 = NeuralNetwork2()
        model3 = NeuralNetwork3()

        train_model(model1, X_train, y_train)
        train_model(model2, X_train, y_train)
        train_model(model3, X_train, y_train)

        # Updated PyTorchModelWrapper class with predict_proba
        class PyTorchModelWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                # PyTorch models are already trained
                return self

            def predict(self, X):
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                    predictions = self.model(X_tensor)
                    return (predictions >= 0.5).float().numpy().ravel()

            def predict_proba(self, X):
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                    predictions = self.model(X_tensor)
                    # Return probabilities for both classes (1-p and p)
                    probabilities = torch.cat((1 - predictions, predictions), dim=1)
                    return probabilities.numpy()

        dt_model = DecisionTreeClassifier(random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        lr_model = LogisticRegression(solver="liblinear")

        # Create voting classifier with additional models
        ensemble = VotingClassifier(
            estimators=[
                ("nn1", PyTorchModelWrapper(model1)),
                ("nn2", PyTorchModelWrapper(model2)),
                ("nn3", PyTorchModelWrapper(model3)),
                ("dt", dt_model),
                ("rf", rf_model),
                ("knn", knn_model),
                ("lr", lr_model),
            ],
            voting="soft",
        )

        # Train the ensemble
        ensemble.fit(X_train.numpy(), y_train.numpy().ravel())

        # Evaluate the ensemble
        ensemble_predictions = ensemble.predict(X_test.numpy())
        accuracy = accuracy_score(y_test.numpy(), ensemble_predictions)
        report = classification_report(
            y_test.numpy(), ensemble_predictions, zero_division=0
        )

        print(f"Ensemble Test Accuracy: {accuracy}")
        print(f"Ensemble Classification Report:\n{report}")

        # Making predictions
        def prepare_input(
            game_mode, player_team, enemy_team, player, teammates, enemies
        ):
            input_data = {feature: 0 for feature in self.features}

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

            # Remove constant features
            input_df = input_df.loc[:, input_df.columns.isin(selected_features)]

            # Select the same features as during training
            input_df = input_df[selected_features]

            # Standardize the data
            input_array = scaler.transform(input_df.to_numpy())

            return input_array

        input_data = prepare_input(
            game_mode=self.game_mode,
            player_team=self.player_team,
            enemy_team=self.enemy_team,
            player=self.player,
            teammates=self.teammates,
            enemies=self.enemies,
        )

        # Making a prediction with the ensemble
        ensemble_prediction = ensemble.predict(input_data)
        result = "Over" if ensemble_prediction[0] == 1 else "Under"

        return f"{result} {self.X_value} kills"


class ModelV2:
    def __init__(
        self,
        df,
        features,
        X_value=25,
        player="aBeZy",
        game_mode="HardPoint",
        player_team="Atlanta_FaZe",
        enemy_team="OpTic_Texas",
        teammates=["Gwinn", "TJHaLy", "Clayster"],
        enemies=["Shotzzy", "Kenny", "Pred", "Dashy"],
    ):
        self.model = None
        self.X_value = X_value
        self.player = player
        self.game_mode = game_mode
        self.player_team = player_team
        self.enemy_team = enemy_team
        self.teammates = teammates
        self.enemies = enemies
        self.df = df
        self.features = features

    def train_and_predict(self):

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Print whether CPU or CUDA is being used
        if device.type == "cuda":
            print("Using CUDA (GPU)")
        else:
            print("Using CPU")

        df = self.df.copy()
        df["Over_X_Kills"] = df["Kills"].apply(lambda x: 1 if x > self.X_value else 0)

        df = df[self.features + ["Over_X_Kills", "Match_ID"]]

        # Handle missing values
        df.fillna(0, inplace=True)

        # Split the data
        X = df.drop(columns=["Over_X_Kills", "Match_ID"])
        y = df["Over_X_Kills"]

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to PyTorch tensors and move to GPU
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = (
            torch.tensor(y.values, dtype=torch.float32).to(device).unsqueeze(1)
        )  # Ensure y is a 2D tensor

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        class NeuralNetwork1(nn.Module):
            def __init__(self):
                super(NeuralNetwork1, self).__init__()
                self.layer1 = nn.Linear(X.shape[1], 128)
                self.layer2 = nn.Linear(128, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = self.layer2(x)
                x = self.sigmoid(x)
                return x

        class NeuralNetwork2(nn.Module):
            def __init__(self):
                super(NeuralNetwork2, self).__init__()
                self.layer1 = nn.Linear(X.shape[1], 64)
                self.layer2 = nn.Linear(64, 32)
                self.layer3 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                x = self.layer3(x)
                x = self.sigmoid(x)
                return x

        class NeuralNetwork3(nn.Module):
            def __init__(self):
                super(NeuralNetwork3, self).__init__()
                self.layer1 = nn.Linear(X.shape[1], 256)
                self.layer2 = nn.Linear(256, 128)
                self.layer3 = nn.Linear(128, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                x = self.layer3(x)
                x = self.sigmoid(x)
                return x

        # Define loss function
        criterion = nn.BCELoss()

        # Hyperparameter values
        lr_values = [0.001, 0.01, 0.1]
        batch_sizes = [32, 64, 128]

        # Function to train PyTorch models
        def train_model(model, optimizer, X_train, y_train, num_epochs=20):
            model.train()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

        # Function to evaluate PyTorch models
        def evaluate_model(model, X_test, y_test):
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                predictions = (outputs >= 0.5).float()
                accuracy = (predictions == y_test).float().mean()
                return accuracy.item()

        # Cross-validation and hyperparameter tuning for PyTorch models
        def cross_val_tune_model(model_class, X, y, num_epochs=20):
            best_model = None
            best_accuracy = 0
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for lr in lr_values:
                for batch_size in batch_sizes:
                    accuracies = []
                    for train_index, val_index in kf.split(X):
                        X_train_cv, X_val_cv = X[train_index], X[val_index]
                        y_train_cv, y_val_cv = y[train_index], y[val_index]

                        model = model_class().to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        train_model(
                            model,
                            optimizer,
                            X_train_cv,
                            y_train_cv,
                            num_epochs=num_epochs,
                        )
                        accuracy = evaluate_model(model, X_val_cv, y_val_cv)
                        accuracies.append(accuracy)

                    mean_accuracy = np.mean(accuracies)
                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_model = model_class().to(device)
                        optimizer = optim.Adam(best_model.parameters(), lr=lr)
                        train_model(best_model, optimizer, X, y, num_epochs=num_epochs)

            return best_model

        # Find the best models for each neural network using cross-validation
        best_model1 = cross_val_tune_model(NeuralNetwork1, X_train, y_train)
        best_model2 = cross_val_tune_model(NeuralNetwork2, X_train, y_train)
        best_model3 = cross_val_tune_model(NeuralNetwork3, X_train, y_train)

        # Define hyperparameter grids for other models
        param_grid_knn = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

        param_grid_lr = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear"]}

        param_grid_rf = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }

        # Create model instances with GridSearchCV
        knn_model = GridSearchCV(
            KNeighborsClassifier(), param_grid_knn, cv=3, n_jobs=-1
        )
        logistic_regression_model = GridSearchCV(
            LogisticRegression(), param_grid_lr, cv=3, n_jobs=-1
        )
        random_forest_model = GridSearchCV(
            RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1
        )

        # Train the GridSearchCV models
        knn_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy().ravel())
        logistic_regression_model.fit(
            X_train.cpu().numpy(), y_train.cpu().numpy().ravel()
        )
        random_forest_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy().ravel())

        # Create a new voting classifier that includes additional models
        class PyTorchModelWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                # PyTorch models are already trained
                return self

            def predict(self, X):
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(
                        torch.tensor(X, dtype=torch.float32).to(device)
                    )
                    return (predictions >= 0.5).cpu().numpy()

            def predict_proba(self, X):
                self.model.eval()
                with torch.no_grad():
                    predictions = (
                        self.model(torch.tensor(X, dtype=torch.float32).to(device))
                        .cpu()
                        .numpy()
                    )
                    return np.hstack([1 - predictions, predictions])

        # Create the ensemble
        ensemble = VotingClassifier(
            estimators=[
                ("nn1", PyTorchModelWrapper(best_model1)),
                ("nn2", PyTorchModelWrapper(best_model2)),
                ("nn3", PyTorchModelWrapper(best_model3)),
                ("knn", knn_model),
                ("log_reg", logistic_regression_model),
                ("rf", random_forest_model),
            ],
            voting="soft",
        )

        # Train the ensemble
        ensemble.fit(X_train.cpu().numpy(), y_train.cpu().numpy().ravel())

        # Evaluate the ensemble
        ensemble_predictions = ensemble.predict(X_test.cpu().numpy())
        accuracy = accuracy_score(y_test.cpu().numpy(), ensemble_predictions)
        report = classification_report(
            y_test.cpu().numpy(), ensemble_predictions, zero_division=0
        )

        print(f"Ensemble Test Accuracy: {accuracy}")
        print(f"Ensemble Classification Report:\n{report}")

        # Prepare the input data as before
        def prepare_input(
            game_mode, player_team, enemy_team, player, teammates, enemies
        ):
            input_data = {feature: 0 for feature in self.features}

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

            # Select the same features as during training
            input_df = input_df[self.features]

            # Standardize the data
            input_array = scaler.transform(input_df)

            return input_array

        input_data = prepare_input(
            self.game_mode,
            self.player_team,
            self.enemy_team,
            self.player,
            self.teammates,
            self.enemies,
        )

        # Make predictions with the updated ensemble model
        input_array = input_data  # Already a numpy array

        # Get prediction
        prediction = ensemble.predict(input_array)
        result = "Over" if prediction[0] == 1 else "Under"

        # If you want probability estimates
        probabilities = ensemble.predict_proba(input_array)

        return f'Prediction: {result} {self.X_value} kills\nProbability of "Over": {probabilities[0][1]:.4f}\nProbability of "Under": {probabilities[0][0]:.4f}'


def main():
    X_value = 22.5  # Number of kills to predict for over/under
    player = "aBeZy"
    game_mode = "HardPoint"  # 'HardPoint', 'Search_and_Destroy', 'Control'
    player_team = "Atlanta_FaZe"
    enemy_team = "Miami_Heretics"
    teammates = ["Drazah", "Cellium", "Simp"]
    enemies = ["Lucky", "MettalZ", "Vikul", "ReeaL"]

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

    # Load the data
    df = pd.read_csv("preprocessed_player_stats_Trial3.csv")

    modelv1 = ModelV1(
        df=df,
        features=features,
        X_value=X_value,
        player=player,
        game_mode=game_mode,
        player_team=player_team,
        enemy_team=enemy_team,
        teammates=teammates,
        enemies=enemies,
    )

    modelv1_message = modelv1.train_and_predict()

    modelv2 = ModelV2(
        df=df,
        features=features,
        X_value=X_value,
        player=player,
        game_mode=game_mode,
        player_team=player_team,
        enemy_team=enemy_team,
        teammates=teammates,
        enemies=enemies,
    )

    modelv2_message = modelv2.train_and_predict()

    print("ModelV1:\n", modelv1_message)
    print("ModelV2:\n", modelv2_message)


if __name__ == "__main__":
    main()
