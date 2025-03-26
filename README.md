# Face Embedding Gender Prediction

This project explores whether we can predict gender from facial embeddings. It's based on a small dataset of 128-dimensional face embedding vectors extracted from facial images.

## Background

For a computer vision class experiment, we collected face images from students and generated 128-dimensional embeddings using a face recognition library. These embeddings are designed to capture the unique characteristics of faces for identification purposes, but they may also encode demographic attributes like gender.

The dataset includes 92 face embedding samples with gender labels (-1 for male, +1 for female). While small, this dataset presents an interesting challenge: can we predict gender using these embeddings, despite having more features (128) than samples (92)?

## The Challenge

This project tackles a classic "small n, large p" problem in machine learning:
- 92 samples
- 128 features
- Potential for overfitting

To address this challenge, I've explored several approaches:
1. Feature selection to identify the most gender-predictive dimensions
2. Regularization to control model complexity
3. Dimensionality reduction (PCA) to compress the feature space
4. Ensemble methods to improve robustness

## What's Included

- `basic_model.py`: Core analysis with feature selection and simple models
- `advanced_model.py`: Advanced techniques with hyperparameter tuning and ensemble models
- `feature_analysis.py`: In-depth analysis of which embedding dimensions best predict gender
- `requirements.txt`: Dependencies needed to run the code

## Key Findings

After extensive experimentation, I found that:

1. **Gender is strongly encoded in face embeddings**
   - Even with simple models, we can achieve >90% accuracy
   - PCA visualization shows clear separation between genders

2. **Only about 20-30 of the 128 features are needed**
   - Feature selection significantly improves model performance
   - Consistent important features were identified across multiple methods
   
3. **Regularization is crucial**
   - L1/L2 regularization helps control overfitting
   - Ridge regression performs well on this dataset

4. **Cross-validation is essential for reliable evaluation**
   - Given the small dataset, k-fold cross-validation provides more realistic performance estimates

5. **Best performing models**
   - Logistic regression with selected features (accuracy ~98%)
   - PCA + logistic regression (accuracy ~97%)
   - Ensemble of multiple model types (accuracy ~98%)

## Usage

To run the analysis:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the basic analysis
python basic_model.py

# Run advanced model tuning
python advanced_model.py

# Run feature importance analysis
python feature_analysis.py
```

## Future Work

Some interesting directions for extending this work:
- Test the gender predictive features on larger, more diverse datasets
- Explore which facial features correspond to the most predictive embedding dimensions
- Investigate whether these embeddings encode other demographic attributes (age, ethnicity)
- Develop techniques to create more privacy-preserving embeddings that minimize demographic information

## Conclusion

This project demonstrates that face embedding vectors contain significant gender information, despite being primarily designed for identity recognition. With appropriate feature selection and regularization techniques, we can build highly accurate gender classifiers even from a very small dataset. 