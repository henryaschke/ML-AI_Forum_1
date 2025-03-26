import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

print("Starting advanced gender prediction analysis...")
print("Loading dataset...")
data = pd.read_csv('faces_embeddings.csv')

# Extract embedding features and gender labels
feature_cols = [str(i) for i in range(128)]
X = data[feature_cols]
y = data['gender']

# Create a proper train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset shape: {data.shape}")
print(f"Number of features: {len(feature_cols)}")
print(f"Gender balance: {y.value_counts().to_dict()}")
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Create different modeling pipelines
print("\nBuilding model pipelines...")

# 1. PCA + LogReg pipeline
pca_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # n_components will be tuned
    ('classifier', LogisticRegression(random_state=42))
])

# 2. Feature selection + LogReg pipeline
select_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif)),
    ('classifier', LogisticRegression(random_state=42))
])

# 3. Ridge classifier pipeline
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RidgeClassifier(random_state=42))
])

# 4. SVM pipeline
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(probability=True, random_state=42))
])

# 5. Random Forest pipeline
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define parameter search spaces
print("\nSetting up hyperparameter grids...")

pca_params = {
    'pca__n_components': [10, 20, 30, 50],
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
}

select_params = {
    'selector__k': [10, 20, 30, 50],
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
}

ridge_params = {
    'classifier__alpha': [0.1, 1.0, 10.0, 100.0],
}

svm_params = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['linear', 'rbf'],
}

rf_params = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, 8],
    'classifier__min_samples_split': [2, 5],
}

# Train and tune each pipeline
print("\nPerforming grid search for each model...")
print("This may take several minutes...")

# 1. PCA + LogReg tuning
print("\nTuning PCA + LogReg pipeline...")
pca_grid = GridSearchCV(pca_logreg, pca_params, cv=cv, scoring='accuracy')
pca_grid.fit(X_train, y_train)
print(f"Best parameters: {pca_grid.best_params_}")
print(f"Best CV accuracy: {pca_grid.best_score_:.4f}")
print(f"Test accuracy: {pca_grid.score(X_test, y_test):.4f}")

# 2. Feature Selection + LogReg tuning
print("\nTuning Feature Selection + LogReg pipeline...")
select_grid = GridSearchCV(select_logreg, select_params, cv=cv, scoring='accuracy')
select_grid.fit(X_train, y_train)
print(f"Best parameters: {select_grid.best_params_}")
print(f"Best CV accuracy: {select_grid.best_score_:.4f}")
print(f"Test accuracy: {select_grid.score(X_test, y_test):.4f}")

# 3. Ridge tuning
print("\nTuning Ridge Classifier pipeline...")
ridge_grid = GridSearchCV(ridge_pipe, ridge_params, cv=cv, scoring='accuracy')
ridge_grid.fit(X_train, y_train)
print(f"Best parameters: {ridge_grid.best_params_}")
print(f"Best CV accuracy: {ridge_grid.best_score_:.4f}")
print(f"Test accuracy: {ridge_grid.score(X_test, y_test):.4f}")

# 4. SVM tuning
print("\nTuning SVM pipeline...")
svm_grid = GridSearchCV(svm_pipe, svm_params, cv=cv, scoring='accuracy')
svm_grid.fit(X_train, y_train)
print(f"Best parameters: {svm_grid.best_params_}")
print(f"Best CV accuracy: {svm_grid.best_score_:.4f}")
print(f"Test accuracy: {svm_grid.score(X_test, y_test):.4f}")

# 5. Random Forest tuning
print("\nTuning Random Forest pipeline...")
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=cv, scoring='accuracy')
rf_grid.fit(X_train, y_train)
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV accuracy: {rf_grid.best_score_:.4f}")
print(f"Test accuracy: {rf_grid.score(X_test, y_test):.4f}")

# Collect all the models
models = {
    'PCA + LogReg': pca_grid,
    'Feature Selection + LogReg': select_grid,
    'Ridge': ridge_grid,
    'SVM': svm_grid,
    'Random Forest': rf_grid
}

# Find the best model
best_model_name = max(models, key=lambda m: models[m].best_score_)
best_model = models[best_model_name]

print(f"\nBest overall model: {best_model_name}")
print(f"CV accuracy: {best_model.best_score_:.4f}")
print(f"Test accuracy: {best_model.score(X_test, y_test):.4f}")

# Detailed evaluation of best model
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Try to generate ROC curve for the best model
try:
    plt.figure(figsize=(8, 6))
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        
        # Convert target values for ROC curve if needed
        y_test_binary = (y_test + 1) / 2 if -1 in y_test.unique() else y_test
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model_name}')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()
        print("ROC curve saved to roc_curve.png")
    else:
        print("Best model doesn't support probability predictions, skipping ROC curve")
except Exception as e:
    print(f"Couldn't generate ROC curve: {e}")

# Create an ensemble of our best models
print("\nBuilding ensemble model...")
estimators = [(name, model.best_estimator_) for name, model in models.items()]
voting_clf = VotingClassifier(estimators=estimators, voting='hard')
voting_clf.fit(X_train, y_train)

# Evaluate ensemble
ensemble_acc = voting_clf.score(X_test, y_test)
ensemble_pred = voting_clf.predict(X_test)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(y_test, ensemble_pred))

# Evaluate all models with cross-validation on full dataset
print("\nFinal 5-fold cross-validation on full dataset:")
for name, model in models.items():
    cv_scores = cross_val_score(model.best_estimator_, X, y, cv=5, scoring='accuracy')
    print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Ensemble CV
cv_scores = cross_val_score(voting_clf, X, y, cv=5, scoring='accuracy')
print(f"Ensemble: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\nAnalysis complete!") 