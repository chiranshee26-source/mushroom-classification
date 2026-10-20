"""
Export the trained models to docs/model.js so the website can make predictions in the browser.

Run from the project root:
    python export_model.py

It repeats the same preparation and training as mushroom_classification.ipynb
(same split, same random_state), then saves:
  - the Decision Tree (every question and answer box)
  - the KNN training mushrooms (KNN predicts by comparing against these)
  - the test mushrooms (used for the "Try an example" buttons)
  - the evaluation results shown on the Results page
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Human-readable names for every column and letter code (from the UCI dataset description)
FEATURES = {
    "cap_shape": ("Cap shape", {"b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat", "k": "Knobbed", "s": "Sunken"}),
    "cap_surface": ("Cap surface", {"f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"}),
    "cap_color": ("Cap colour", {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Grey", "r": "Green", "p": "Pink",
                                 "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"}),
    "bruises": ("Bruises when pressed", {"t": "Yes", "f": "No"}),
    "odor": ("Smell (odour)", {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty",
                               "n": "No smell", "p": "Pungent", "s": "Spicy"}),
    "gill_attachment": ("Gill attachment", {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"}),
    "gill_spacing": ("Gill spacing", {"c": "Close", "w": "Crowded", "d": "Distant"}),
    "gill_size": ("Gill size", {"b": "Broad", "n": "Narrow"}),
    "gill_color": ("Gill colour", {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Grey", "r": "Green",
                                   "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"}),
    "stalk_shape": ("Stalk shape", {"e": "Enlarging", "t": "Tapering"}),
    "stalk_root": ("Stalk root", {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs", "r": "Rooted",
                                  "missing": "Unknown / not recorded"}),
    "stalk_surface_above_ring": ("Stalk surface above ring", {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"}),
    "stalk_surface_below_ring": ("Stalk surface below ring", {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"}),
    "stalk_color_above_ring": ("Stalk colour above ring", {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Grey", "o": "Orange",
                                                           "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"}),
    "stalk_color_below_ring": ("Stalk colour below ring", {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Grey", "o": "Orange",
                                                           "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"}),
    "veil_color": ("Veil colour", {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"}),
    "ring_number": ("Number of rings", {"n": "None", "o": "One", "t": "Two"}),
    "ring_type": ("Ring type", {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant",
                                "s": "Sheathing", "z": "Zone"}),
    "spore_print_color": ("Spore print colour", {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green",
                                                 "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"}),
    "population": ("Population (how many grow together)", {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered",
                                                           "v": "Several", "y": "Solitary"}),
    "habitat": ("Habitat (where it grows)", {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban",
                                             "w": "Waste", "d": "Woods"}),
}

# 1. Prepare the data exactly like the notebook
df = pd.read_csv("data/mushrooms.csv")
df["stalk_root"] = df["stalk_root"].replace("?", "missing")
df = df.drop(columns=[c for c in df.columns if df[c].nunique() == 1])  # drops veil_type
X_raw = df.drop(columns="type")
y = (df["type"] == "p").astype(int).to_numpy()  # 0 = edible, 1 = poisonous
columns = list(X_raw.columns)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X = encoder.fit_transform(X_raw)
feature_names = list(encoder.get_feature_names_out(columns))

idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train, X_test, y_train, y_test = X[idx_train], X[idx_test], y[idx_train], y[idx_test]

# 2. Train both models
dt = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
k_values = list(range(1, 16, 2))
k_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5).mean() for k in k_values]
best_k = max(k for k, s in zip(k_values, k_scores) if np.isclose(s, max(k_scores)))
knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
dt_pred, knn_pred = dt.predict(X_test), knn.predict(X_test)
dt_small = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(X_train, y_train)


# 3. Turn the tree into a list of plain nodes: each question is "is <column> equal to <value>?"
def split_name(name):
    for col in sorted(columns, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col, name[len(col) + 1:]
    raise ValueError(name)


tree = dt.tree_
nodes = []
for i in range(tree.node_count):
    counts = tree.value[i][0] / tree.value[i][0].sum() * tree.n_node_samples[i]
    node = {"samples": [int(round(counts[0])), int(round(counts[1]))]}
    if tree.children_left[i] != -1:
        col, val = split_name(feature_names[tree.feature[i]])
        # In scikit-learn, "feature <= 0.5" (the switch is OFF) goes left
        node.update({"col": col, "val": val, "no": int(tree.children_left[i]), "yes": int(tree.children_right[i])})
    nodes.append(node)


def rows_to_strings(index):
    # Each mushroom is stored as one 21-letter string, one letter per column in column order
    # ("missing" is stored as "?" to keep it one letter), e.g. "xsntpfcnkeesswwwoopksu"
    return ["".join("?" if v == "missing" else v for v in X_raw.iloc[i]) for i in index]


importances = sorted(zip(feature_names, dt.feature_importances_), key=lambda t: -t[1])[:10]
model = {
    "columns": columns,
    "features": {c: {"label": FEATURES[c][0],
                     "options": {v: FEATURES[c][1].get(v, v) for v in sorted(X_raw[c].unique())}} for c in columns},
    "tree": nodes,
    "knn": {"k": best_k, "rows": rows_to_strings(idx_train), "labels": y_train.tolist()},
    "examples": {"rows": rows_to_strings(idx_test), "labels": y_test.tolist()},
    "results": {
        "n_total": int(len(df)), "n_train": int(len(idx_train)), "n_test": int(len(idx_test)),
        "n_edible": int((y == 0).sum()), "n_poisonous": int((y == 1).sum()),
        "missing_stalk_root": int((df["stalk_root"] == "missing").sum()),
        "n_encoded_features": len(feature_names),
        "tree_depth": int(dt.get_depth()), "tree_leaves": int(dt.get_n_leaves()),
        "dt": {"test_acc": accuracy_score(y_test, dt_pred),
               "cv_acc": cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE), X_train, y_train, cv=5).mean(),
               "cm": confusion_matrix(y_test, dt_pred).tolist()},
        "knn": {"test_acc": accuracy_score(y_test, knn_pred), "cv_acc": max(k_scores),
                "cm": confusion_matrix(y_test, knn_pred).tolist()},
        "small_tree_acc": accuracy_score(y_test, dt_small.predict(X_test)),
        "k_scores": [{"k": k, "score": s} for k, s in zip(k_values, k_scores)],
        "importances": [{"feature": "{} = {}".format(FEATURES[c][0], FEATURES[c][1].get(v, v)), "value": imp}
                        for c, v, imp in ((*split_name(n), imp) for n, imp in importances)],
    },
}

with open("docs/model.js", "w") as f:
    f.write("// Generated by export_model.py: do not edit by hand.\n")
    f.write("window.MODEL = " + json.dumps(model, separators=(",", ":")) + ";\n")

print(f"Saved docs/model.js  (Decision Tree: {len(nodes)} nodes, KNN: k={best_k}, {len(idx_train)} training mushrooms)")
print(f"Test accuracy  Decision Tree: {model['results']['dt']['test_acc']:.4f}  KNN: {model['results']['knn']['test_acc']:.4f}")
