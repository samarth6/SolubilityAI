import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

@st.cache_resource
def train_all_models(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=150, learning_rate=0.08, random_state=42),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42, verbosity=0)
    except ImportError:
        pass

    trained, metrics = {}, {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        cv = cross_val_score(model, X_scaled, y, cv=min(5, len(y)), scoring="r2")
        trained[name] = model
        metrics[name] = {
            "R2":     round(float(r2_score(y, preds)), 3),
            "RMSE":   round(float(np.sqrt(mean_squared_error(y, preds))), 3),
            "CV_R2":  round(float(cv.mean()), 3),
            "CV_Std": round(float(cv.std()), 3),
            "MAE":    round(float(np.mean(np.abs(y - preds))), 3),
        }
    return trained, scaler, metrics

def predict_single(model, scaler, features):
    return float(model.predict(scaler.transform(features))[0])

def get_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    return None
