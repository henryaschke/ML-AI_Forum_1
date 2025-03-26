import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# Load the face embeddings dataset
print("Loading dataset...")
data = pd.read_csv('faces_embeddings.csv')

# Basic info about the dataset
print(f"Dataset shape: {data.shape}")
print(f"First few rows:")
print(data.head())

# Check for missing data
print(f"Missing values: {data.isnull().sum().sum()}")

# Extract features and target
# All numeric columns from 0-127 are embedding features
feature_cols = [str(i) for i in range(128)]
X = data[feature_cols]
y = data['gender']

print(f"Feature matrix shape: {X.shape}")
print(f"Gender distribution:")
print(y.value_counts())

# Scale the features (important for face embeddings)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize the data with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Gender (-1: Male, 1: Female)')
plt.title('PCA Visualization of Face Embeddings')
plt.xlabel(f'PC1 ({var_explained[0]*100:.2f}% variance)')
plt.ylabel(f'PC2 ({var_explained[1]*100:.2f}% variance)')
plt.savefig('pca_visualization.png')
plt.close()

print(f"PCA explained variance: {var_explained}")
print(f"Total variance explained by 2 PCs: {sum(var_explained)*100:.2f}%")

# Feature selection - pick the top features 
print("\nSelecting important features...")
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)
selected_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_indices]

print(f"Selected features: {selected_features}")

# Build & evaluate models using cross-validation
models = {
    'Logistic Regression': LogisticRegression(C=0.1, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\nEvaluating models with 5-fold cross-validation:")
for name, model in models.items():
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Final model evaluation on a train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Use the best model based on CV results
best_model_name = max(results, key=lambda k: results[k].mean())
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"\nBest model: {best_model_name}")
print("\nTest set results:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

print("\nAnalysis complete. Check the generated images for visualizations.") 