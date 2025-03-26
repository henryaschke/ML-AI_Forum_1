import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import os

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Set aesthetics for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

print("Loading and preparing data...")
# Load the face embeddings dataset
data = pd.read_csv('faces_embeddings.csv')

# Extract features and target
feature_cols = [str(i) for i in range(128)]
X = data[feature_cols]
y = data['gender']

# Get gender counts for plotting
gender_counts = y.value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
gender_counts['Gender'] = gender_counts['Gender'].map({-1: 'Male', 1: 'Female'})

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Generating visualizations...")

# 1. Gender Distribution Visualization
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Gender', y='Count', data=gender_counts, palette=['#3498db', '#e74c3c'])
ax.set_title('Gender Distribution in Dataset', fontweight='bold')
ax.bar_label(ax.containers[0], fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/gender_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. PCA Visualization with Enhanced Design
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))
plt.scatter(X_pca[y == -1, 0], X_pca[y == -1, 1], s=100, c='#3498db', label='Male', alpha=0.7, edgecolor='w', linewidth=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s=100, c='#e74c3c', label='Female', alpha=0.7, edgecolor='w', linewidth=0.5)

plt.title('Principal Component Analysis of Face Embeddings', fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.legend(title='Gender', title_fontsize=14, fontsize=12, markerscale=1.5)

# Add a color background
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Add 95% confidence ellipses
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # Computing the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # Computing the standard deviation of y from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

ax = plt.gca()
confidence_ellipse(X_pca[y == -1, 0], X_pca[y == -1, 1], ax, n_std=2.0, 
                  edgecolor='#3498db', linewidth=2, linestyle='--', alpha=0.5)
confidence_ellipse(X_pca[y == 1, 0], X_pca[y == 1, 1], ax, n_std=2.0, 
                  edgecolor='#e74c3c', linewidth=2, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('visualizations/pca_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PCA 3D Visualization
pca3d = PCA(n_components=3)
X_pca3d = pca3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter
males = ax.scatter(X_pca3d[y == -1, 0], X_pca3d[y == -1, 1], X_pca3d[y == -1, 2], 
                    s=70, c='#3498db', label='Male', alpha=0.7, edgecolor='w', linewidth=0.5)
females = ax.scatter(X_pca3d[y == 1, 0], X_pca3d[y == 1, 1], X_pca3d[y == 1, 2], 
                     s=70, c='#e74c3c', label='Female', alpha=0.7, edgecolor='w', linewidth=0.5)

ax.set_title('3D PCA Visualization of Face Embeddings', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca3d.explained_variance_ratio_[0]*100:.2f}%)')
ax.set_ylabel(f'PC2 ({pca3d.explained_variance_ratio_[1]*100:.2f}%)')
ax.set_zlabel(f'PC3 ({pca3d.explained_variance_ratio_[2]*100:.2f}%)')
ax.legend(title='Gender')

# Improve 3D visualization
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/pca_3d.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. t-SNE Visualization
print("Running t-SNE dimensionality reduction (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))
plt.scatter(X_tsne[y == -1, 0], X_tsne[y == -1, 1], s=100, c='#3498db', label='Male', alpha=0.7, edgecolor='w', linewidth=0.5)
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], s=100, c='#e74c3c', label='Female', alpha=0.7, edgecolor='w', linewidth=0.5)

plt.title('t-SNE Visualization of Face Embeddings', fontweight='bold')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Gender', title_fontsize=14, fontsize=12, markerscale=1.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. PCA Cumulative Explained Variance
pca_full = PCA()
pca_full.fit(X_scaled)

# Create high-quality cumulative variance plot
plt.figure(figsize=(12, 8))
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components = len(pca_full.explained_variance_ratio_)

# Plot styled line
plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', markersize=6, 
         linewidth=2, color='#2980b9')

# Add threshold lines
plt.axhline(y=0.95, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label='95% Threshold')
plt.axhline(y=0.99, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7, label='99% Threshold')

# Find and mark number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
plt.axvline(x=n_components_95, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.7)
plt.scatter(n_components_95, cumulative_variance[n_components_95-1], s=100, 
            color='#e74c3c', zorder=5, edgecolor='white', linewidth=1.5)
plt.annotate(f'{n_components_95} components\n(95% variance)', 
             xy=(n_components_95, cumulative_variance[n_components_95-1]),
             xytext=(n_components_95+5, cumulative_variance[n_components_95-1]-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, ha='center')

plt.title('Cumulative Explained Variance by Number of Components', fontweight='bold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(np.arange(0, n_components+1, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlim(0, min(n_components, 80))  # Show a reasonable number of components
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('visualizations/pca_cumulative_variance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Top Feature Importance Visualization (Random Forest)
print("Calculating feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Get feature importances and sort
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
top_features = feature_importance.head(15)

plt.figure(figsize=(12, 8))
bars = plt.barh(np.arange(len(top_features)), top_features['Importance'], align='center', 
        color=[plt.cm.viridis(x/max(top_features['Importance'])) for x in top_features['Importance']])
plt.yticks(np.arange(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 15 Most Important Features (Random Forest)', fontweight='bold')
plt.gca().invert_yaxis()  # Display highest importance at the top

# Add value labels
for i, v in enumerate(top_features['Importance']):
    plt.text(v + 0.002, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Feature Correlation Heatmap for top features
top_10_features = feature_importance.head(10)['Feature'].tolist()
correlation = X[top_10_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8})

plt.title('Correlation Between Top 10 Features', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Learning Curves
print("Generating learning curves...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=20)),
    ('classifier', LogisticRegression(random_state=42))
])

# Calculate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
)

# Calculate mean and std for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(12, 8))
plt.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training score', linewidth=3, markersize=8)
plt.plot(train_sizes, test_mean, 'o-', color='#e74c3c', label='Validation score', linewidth=3, markersize=8)

# Add bands for standard deviation
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#3498db')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='#e74c3c')

# Add titles and labels
plt.title('Learning Curve (Logistic Regression with Feature Selection)', fontweight='bold')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.xticks(train_sizes)
plt.ylim(0.4, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations/learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Validation Curve for regularization strength
print("Generating validation curves...")
param_range = np.logspace(-4, 4, 10)
train_scores, test_scores = validation_curve(
    LogisticRegression(random_state=42), X_scaled, y,
    param_name="C", param_range=param_range, cv=5, scoring="accuracy"
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(12, 8))
plt.semilogx(param_range, train_mean, 'o-', color='#3498db', label='Training score', linewidth=3, markersize=8)
plt.semilogx(param_range, test_mean, 'o-', color='#e74c3c', label='Validation score', linewidth=3, markersize=8)

# Add bands for standard deviation
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#3498db')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='#e74c3c')

# Mark the optimal parameter
best_index = np.argmax(test_mean)
best_param = param_range[best_index]
best_score = test_mean[best_index]
plt.scatter(best_param, best_score, s=150, c='gold', edgecolor='k', marker='*', zorder=5,
            label=f'Best C: {best_param:.4f}')

plt.title('Validation Curve (Logistic Regression C Parameter)', fontweight='bold')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Accuracy Score')
plt.ylim(0.5, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations/validation_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Model Comparison
print("Generating model comparison visualization...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'L1 Regularization': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
    'L2 Regularization': LogisticRegression(penalty='l2', random_state=42)
}

# Define feature selection pipeline
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)

# Compare models with cross-validation
from sklearn.model_selection import cross_val_score
model_scores = []
model_names = []

for name, model in models.items():
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    model_scores.append(scores)
    model_names.append(name)

plt.figure(figsize=(14, 8))
box = plt.boxplot(model_scores, patch_artist=True, labels=model_names)

# Set colors for each boxplot
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add individual points
for i, (name, scores) in enumerate(zip(model_names, model_scores)):
    x = np.random.normal(i+1, 0.07, size=len(scores))
    plt.scatter(x, scores, s=60, alpha=0.7, c='k')

plt.title('Model Performance Comparison', fontweight='bold')
plt.ylabel('Accuracy Score')
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random Guess')
plt.ylim(0.4, 1.05)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. Combined Visualization Dashboard
print("Creating visualization dashboard...")
plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# PCA scatter plot
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(X_pca[y == -1, 0], X_pca[y == -1, 1], s=70, c='#3498db', label='Male', alpha=0.7, edgecolor='w', linewidth=0.5)
ax1.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s=70, c='#e74c3c', label='Female', alpha=0.7, edgecolor='w', linewidth=0.5)
ax1.set_title('PCA Visualization')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
ax1.legend(title='Gender')
ax1.grid(True, alpha=0.3)

# t-SNE scatter plot
ax2 = plt.subplot(gs[0, 1])
ax2.scatter(X_tsne[y == -1, 0], X_tsne[y == -1, 1], s=70, c='#3498db', label='Male', alpha=0.7, edgecolor='w', linewidth=0.5)
ax2.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], s=70, c='#e74c3c', label='Female', alpha=0.7, edgecolor='w', linewidth=0.5)
ax2.set_title('t-SNE Visualization')
ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.legend(title='Gender')
ax2.grid(True, alpha=0.3)

# Feature importance plot
ax3 = plt.subplot(gs[1, 0])
top_10 = feature_importance.head(10)
ax3.barh(np.arange(len(top_10)), top_10['Importance'], align='center', 
        color=[plt.cm.viridis(x/max(top_10['Importance'])) for x in top_10['Importance']])
ax3.set_yticks(np.arange(len(top_10)))
ax3.set_yticklabels(top_10['Feature'])
ax3.set_xlabel('Importance')
ax3.set_title('Top 10 Important Features')
ax3.invert_yaxis()  # Display highest importance at the top

# Model performance plot
ax4 = plt.subplot(gs[1, 1])
bars = ax4.bar(model_names, [np.mean(scores) for scores in model_scores], 
               yerr=[np.std(scores) for scores in model_scores],
               color=colors, alpha=0.7, capsize=10)
ax4.set_title('Model Performance Comparison')
ax4.set_ylabel('Mean Accuracy Score')
ax4.set_ylim(0.5, 1.05)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom')

plt.suptitle('Face Embedding Gender Prediction: Key Insights', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('visualizations/dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations generated successfully!")
print("Output saved to 'visualizations' directory") 