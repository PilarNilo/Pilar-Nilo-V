#importción de librerias
import mlflow
import mlflow.xgboost
import xgboost as xgb
import optuna
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

path = './Lab12/'
#lectura base de datos
df = pd.read_csv(os.path.join(path, 'water_potability.csv'))
df

#con los valores nlos los eliminamos
df = df.dropna()

#separamos el dataset en la clase y el resto de columnas
x = df.drop(columns=['Potability'])
y = df['Potability']

#separo en conjunto entrenamiento, valid y test
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=60)

def opt_hiper(trial):
  param={
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "eta": trial.suggest_float("eta", 0.01, 0.1),
        "gamma": trial.suggest_float("gamma", 0.01, 1.0)

  }

  #creamos y entrenamos el modelo
  model = xgb.XGBClassifier(**param)
  model.fit(x_train, y_train)

  #vemos las predis y se calcula f1
  y_pred = model.predict(x_test)
  valid_f1 = f1_score(y_test, y_pred, average='weighted')
  if mlflow.active_run():
    mlflow.end_run()  # Finaliza cualquier ejecución activa

  with mlflow.start_run(run_name=f"XGBoost Trial {trial.number}"):
    mlflow.log_params(param)
    mlflow.log_metric("valid_f1", valid_f1)

  return valid_f1

def optimize_model():
  mlflow.set_experiment('lab12')

  #se crea el estudio de optuna
  study=optuna.create_study(direction="maximize")

  #se optimiza
  study.optimize(opt_hiper,n_trials=100)

  #se obtiene el mejor modelo
  best_trial=study.best_trial
  best_model=xgb.XGBClassifier(**best_trial.params)
  best_model.fit(x_train,y_train)

  #guardamos el mejorcito
  with open(os.path.join(path, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

  mlflow.log_artifact(os.path.join(path, 'best_model.pkl'))

  #obtenemos gráficos y guardamos
  fig1=optuna.visualization.plot_optimization_history(study)
  fig1.write_image(os.path.join(path, 'optimization_history.png'))
  fig2=optuna.visualization.plot_param_importances(study)
  fig2.write_image(os.path.join(path, 'param_importances.png'))

  mlflow.log_artifact(os.path.join(path, 'optimization_history.png'))
  mlflow.log_artifact(os.path.join(path, 'param_importances.png'))

  return best_model
if __name__ == "__main__":
  best_model=optimize_model()

