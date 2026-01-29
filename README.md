# Pass Arrival Prediction

Predict the landing location (`end_x`, `end_y`) of a football pass based on the sequence of previous events.

This repository implements a **Graph-based approach** combined with **CatBoost** to model the spatial and contextual relationships between players, teams, pitch zones, and actions.

## 📌 Methodology

### 1. Graph Construction (NetworkX)
We model the football match data as a **heterogeneous graph** where nodes represent entities and edges represent interactions:

*   **Nodes**:
    *   **Player (`P:{id}`)**: Individual players.
    *   **Team (`T:{id}`)**: Teams involved.
    *   **Zone (`Z:{id}`)**: The pitch is discretized into a 12x8 grid (96 zones). Each coordinate is mapped to a zone ID.
    *   **Action (`A:{type}`)**: Event types (e.g., 'Pass', 'Carry').

*   **Edges**:
    *   **Team - Player**: Membership relationship.
    *   **Player - Zone (Start)**: Players initiating actions in specific zones.
    *   **Action - Zone (Start)**: Which actions typically occur in which zones.
    *   **Zone (Start) - Zone (End)**: Spatial transitions of the ball (e.g., a pass from Zone A to Zone B).

### 2. Node Embedding (Node2Vec / Word2Vec)
We use `gensim`'s `Word2Vec` to generate vector embeddings for all nodes in the graph. This effectively captures the "context" of each player, team, and zone based on their connectivity in the match graph.

*   **Logic**: Players who play in similar zones or make similar passes will have closer embeddings.

### 3. Prediction Model (CatBoost Regressor)
We construct tabular features for each pass event by concatenating the learned embeddings and raw features:
*   **Features**:
    *   `start_x`, `start_y`, `time_seconds`
    *   Embeddings for: `player_id`, `team_id`, `zone_id`, `type_name`
*   **Target**: `end_x`, `end_y`
*   **Model**: Two separate `CatBoostRegressor` models are trained for X and Y coordinates respectively.
*   **Validation**: Group K-Fold Cross Validation (grouped by `game_id`).

## 📂 Project Structure

*   `using_Graph.ipynb`: **Main Notebook**. Contains the complete pipeline:
    1.  Data Loading & Preprocessing
    2.  Graph Construction (`build_train_graph`)
    3.  Node Embedding Generation (`Word2Vec`)
    4.  Feature Engineering (Embedding Lookup)
    5.  Model Training (`CatBoostRegressor`) with CV
    6.  Inference & Submission file generation
*   `train.csv`: Training dataset containing match events.
*   `test.csv`: Test dataset for prediction.
*   `pass_arrival_pred.ipynb`: (Placeholder / Alternative approach).

## 🛠️ Requirements

To run the notebook, you need the following Python libraries:

```bash
pip install pandas numpy matplotlib networkx gensim catboost scikit-learn
```

## 🚀 How to Run

1.  Place `train.csv` and `test.csv` in the project root.
2.  Open `using_Graph.ipynb`.
3.  Run all cells to:
    *   Visualize the pitch zones.
    *   Build the graph and train embeddings.
    *   Train the CatBoost models.
    *   Generate `submission.csv`.

## 📊 Results

The model evaluates performance using Mean Absolute Error (MAE) via 5-fold cross-validation during training.
