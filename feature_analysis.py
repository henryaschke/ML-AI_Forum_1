import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# Load dataset
print("Loading face embeddings dataset...")
data = pd.read_csv('faces_embeddings.csv')

# Extract features and target
feature_cols = [str(i) for i in range(128)]
X = data[feature_cols]
y = data['gender']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_df = pd.DataFrame(X_scaled, columns=feature_cols)

print(f"Dataset size: {data.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Gender distribution: {y.value_counts().to_dict()}")

# APPROACH 1: Random Forest feature importance
print("\n1. Random Forest Feature Importance")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Get feature importances
importance = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

# Plot top 20 important features
plt.figure(figsize=(12, 8))
plt.title("Feature Importance from Random Forest", fontsize=14)
plt.bar(range(20), importance[indices[:20]], yerr=std[indices[:20]], align="center")
plt.xticks(range(20), [feature_cols[i] for i in indices[:20]], rotation=90)
plt.ylabel("Relative Importance")
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

print("Top 10 features from Random Forest:")
for i in range(10):
    print(f"Feature {feature_cols[indices[i]]}: {importance[indices[i]]:.4f}")

# APPROACH 2: Logistic Regression coefficients
print("\n2. Logistic Regression Coefficients")
lasso = LogisticRegression(C=0.01, penalty='l1', solver='liblinear', random_state=42)
lasso.fit(X_scaled, y)

# Get coefficients
coef = pd.Series(lasso.coef_[0], index=feature_cols)
importance_coef = coef.sort_values(key=abs, ascending=False)

# Plot
plt.figure(figsize=(12, 8))
plt.title("Feature Importance from Logistic Regression", fontsize=14)
colors = ['blue' if c < 0 else 'red' for c in importance_coef[:20]]
importance_coef[:20].plot(kind='bar', color=colors)
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.savefig('logreg_coefficients.png')
plt.close()

print("Top 10 features from Logistic Regression:")
print(importance_coef[:10])

# APPROACH 3: Recursive Feature Elimination
print("\n3. Recursive Feature Elimination")
rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=20)
rfe.fit(X_scaled, y)

# Get selected features
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
feature_ranking = [(feature_cols[i], rfe.ranking_[i]) for i in range(len(feature_cols))]
feature_ranking.sort(key=lambda x: x[1])

print("Top 10 features from RFE:")
for feature, rank in feature_ranking[:10]:
    print(f"Feature {feature}: rank {rank}")

# APPROACH 4: Permutation Importance
print("\n4. Permutation Importance")
# Use a simple model for permutation importance
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Calculate permutation importance
result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42)
perm_importance = pd.Series(result.importances_mean, index=feature_cols)
perm_importance = perm_importance.sort_values(ascending=False)

# Plot
plt.figure(figsize=(12, 8))
plt.title("Feature Importance from Permutation Analysis", fontsize=14)
perm_importance[:20].plot(kind='bar')
plt.ylabel("Mean Importance")
plt.tight_layout()
plt.savefig('permutation_importance.png')
plt.close()

print("Top 10 features from Permutation Importance:")
print(perm_importance[:10])

# Find consistent important features across methods
print("\nIdentifying consistently important features...")

# Get top 20 features from each method
rf_top20 = [feature_cols[i] for i in indices[:20]]
logreg_top20 = importance_coef[:20].index.tolist()
rfe_top20 = [f for f, r in feature_ranking[:20]]
perm_top20 = perm_importance[:20].index.tolist()

# Find intersection
common_features = set(rf_top20) & set(logreg_top20) & set(rfe_top20) & set(perm_top20)
very_important = set(rf_top20[:10]) & set(logreg_top20[:10]) & set(rfe_top20[:10]) & set(perm_top20[:10])

print(f"\nFeatures important in ALL methods (top 20): {len(common_features)}")
if common_features:
    for feature in common_features:
        print(feature)

print(f"\nFeatures important in ALL methods (top 10): {len(very_important)}")
if very_important:
    for feature in very_important:
        print(feature)

# PCA Analysis
print("\nRunning PCA to understand feature relationships...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.grid(True)
plt.savefig('pca_variance.png')
plt.close()

# Find how many components for 95% variance
n_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"\nNumber of components needed for 95% variance: {n_95}")

# Visualize first two components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Gender (-1: Male, 1: Female)')
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('pca_components.png')
plt.close()

# Correlation heatmap for important features
if very_important:
    plt.figure(figsize=(12, 10))
    important_list = list(very_important)
    correlation = X_df[important_list].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.close()
    print("\nCorrelation heatmap saved as feature_correlation.png")
else:
    print("\nNo common important features found across all methods for correlation analysis.")

print("\nFeature analysis complete. Check the generated images for visualizations.") 