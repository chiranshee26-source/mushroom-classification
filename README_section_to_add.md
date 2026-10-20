## 🌐 Web app (frontend)

The project includes a small website where anyone can describe a mushroom and see what **both models** predict, along with **why**: the Decision Tree's questions step by step and the most similar mushrooms KNN found.

**Live demo:** https://YOUR-USERNAME.github.io/mushroom-classification/

### Features
- **Classify a mushroom:** 21 questions grouped by part of the mushroom (smell, cap, gills, stalk, ring, habitat). The verdict updates live as you answer.
- **Explainable results:** the Decision Tree shows every yes/no question it asked, and KNN shows its 5 closest training mushrooms and how many answers match.
- **Try real examples:** load a random edible or poisonous mushroom from the test set (never seen during training) and check whether the models get it right.
- **Safety-first design:** if the models disagree, the verdict is "Treat as poisonous", and unusual combinations get a warning.
- **Results page:** accuracy, cross-validation scores, confusion matrices and the most useful features.
- Works on phones and desktops, has light and dark themes, and runs entirely in the browser (no server, nothing is sent anywhere).

### How it works
The website is plain **HTML + CSS + JavaScript**, with no frameworks and nothing to install.

```
export_model.py   →   docs/model.js   →   docs/app.js   →   docs/index.html
(trains the models      (trained models     (runs the models     (the page you
 like the notebook)      saved as data)      in the browser)      see)
```

| File | What it does |
|---|---|
| `export_model.py` | Repeats the notebook's data preparation and training (same split, same `random_state=42`), then saves the Decision Tree, the KNN training mushrooms, the test examples and the results into `docs/model.js` |
| `docs/index.html` | Page structure: classifier, "How it works", results and about sections |
| `docs/style.css` | The design: colours, layout, light/dark themes, mobile layout |
| `docs/app.js` | The logic: builds the form, runs the Decision Tree and KNN on your answers, and shows the explanation |
| `docs/model.js` | The trained models as data (generated, so don't edit it by hand) |

The JavaScript version of both models was checked against all 1,625 test mushrooms and gives the same answers as scikit-learn (100% accuracy for both).

### Run it locally
Just open `docs/index.html` in your browser (double-click it).

To rebuild `docs/model.js` after changing the model:
```bash
python export_model.py
```
