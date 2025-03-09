from catboost import CatBoostRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
import joblib

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 6]
}

# Обучение модели
model = CatBoostRegressor()
scorer = make_scorer(f1_score, average='weighted')
grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scorer, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

gs_model = grid_search.best_estimator_

# Сохранение модели
joblib.dump(gs_model, 'iris_model.joblib')
