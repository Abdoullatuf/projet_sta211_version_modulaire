# pipeline appliqué dans le notebook 01

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif # Example feature selection

# Assuming you have these imported from your modules, otherwise define them or adjust imports
from modules.preprocessing.transformation_optimale_mixte import appliquer_transformation_optimale
from modules.preprocessing.outliers import detect_and_remove_outliers # If you want outlier handling in the pipeline
from modules.preprocessing.column_inspector import inspect_datasets
from modules.preprocessing.missing_values import find_highly_correlated_groups, apply_collinearity_filter


def create_preprocessing_pipeline(imputation_method='knn', outlier_handling=False, feature_selection=False, n_features_to_select=100):
    """
    Crée un pipeline de prétraitement basé sur les étapes définies.

    Args:
        imputation_method (str): Méthode d'imputation ('knn' ou 'mice').
        outlier_handling (bool): Si True, ajoute l'étape de suppression/gestion des outliers.
                                 (Note: La suppression d'outliers est mieux gérée hors pipeline si elle réduit le nombre de lignes).
        feature_selection (bool): Si True, ajoute une étape de sélection de features.
        n_features_to_select (int): Nombre de features à sélectionner si feature_selection est True.

    Returns:
        sklearn.pipeline.Pipeline: Le pipeline de prétraitement.
    """
    steps = []

    # 1. Transformation optimale (Yeo-Johnson/Box-Cox) - Applied outside, on the full dataframe
    #    The transformation functions return a new dataframe with transformed columns.
    #    We assume this step is done *before* fitting the pipeline, or the pipeline
    #    needs a custom transformer for this. For simplicity here, we assume
    #    transformed columns ('X1_trans', 'X2_trans', 'X3_trans') are already present.
    #    If not, you'd need a transformer like this:
    #    class OptimalTransformer(BaseEstimator, TransformerMixin):
    #        def fit(self, X, y=None): ...
    #        def transform(self, X): ...
    #    And add it as the first step.

    # 2. Gestion des outliers (Optionnel)
    #    Note: Removing rows for outliers is typically done *before* fitting the pipeline
    #    to avoid altering the dataset shape during fit/transform.
    #    If you need *transformation* or *winsorization* for outliers *within* the pipeline,
    #    you'd need a custom transformer. The current `detect_and_remove_outliers`
    #    modifies the dataframe by removing rows, which isn't standard for a pipeline step.
    #    Leaving this step out of the standard pipeline definition for now.

    # 3. Imputation des valeurs manquantes
    if imputation_method == 'knn':
        # Need to find the optimal k - this is typically done *before* defining the pipeline
        # For a pipeline, we might fix k or use a search like GridSearchCV to find the best k within the pipeline
        # Let's assume a fixed k for the pipeline for simplicity, based on previous analysis (e.g., optimal_k_no_outliers = 19)
        k_optimal = 19 # Use the value found in EDA
        imputer = KNNImputer(n_neighbors=k_optimal)
        steps.append(('imputer', imputer))
    elif imputation_method == 'mice':
        # MICE is complex and often requires iterating over columns.
        # As of scikit-learn 1.3, IterativeImputer provides MICE-like functionality.
        # Ensure you have scikit-learn version >= 0.24 for IterativeImputer.
        try:
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(max_iter=10, random_state=42) # Example parameters
            steps.append(('imputer', imputer))
        except ImportError:
            print("Warning: IterativeImputer (MICE) requires scikit-learn >= 0.24. Using SimpleImputer instead.")
            imputer = SimpleImputer(strategy='mean') # Fallback
            steps.append(('imputer', imputer))
    else:
        raise ValueError(f"Méthode d'imputation non supportée: {imputation_method}")

    # 4. Sélection de variables collinéaires (Applied outside, on the full dataframe)
    #    Finding correlated columns depends on the *data*. This step identifies
    #    columns to drop *before* the pipeline or needs a custom transformer
    #    that fits on X and determines which columns to drop.
    #    Let's assume `apply_collinearity_filter` was used to get the relevant columns
    #    *before* passing data to the pipeline, or we add a custom transformer.
    #    Adding a simple column dropper transformer for demonstration.

    class ColumnDropper(BaseEstimator, TransformerMixin):
        def __init__(self, columns_to_drop):
            self.columns_to_drop = columns_to_drop

        def fit(self, X, y=None):
            # Nothing to fit, just store columns
            return self

        def transform(self, X):
            # Drop columns, handle potential missing columns gracefully
            existing_cols_to_drop = [col for col in self.columns_to_drop if col in X.columns]
            if existing_cols_to_drop:
                 return X.drop(columns=existing_cols_to_drop)
            return X # No columns to drop found

    # We need the list of columns to drop *before* creating the pipeline instance.
    # This is typically determined during the EDA/preprocessing phase on the training data.
    # Let's assume `correlated_info` and `correlated_info['to_drop']` exist from previous steps.
    # Filter out 'y' and 'X4' from the drop list if they were accidentally included.
    cols_to_drop_collinearity = [col for col in correlated_info.get('to_drop', []) if col not in ['y', 'X4']]
    if cols_to_drop_collinearity:
        steps.append(('collinearity_dropper', ColumnDropper(columns_to_drop=cols_to_drop_collinearity)))


    # 5. Feature Selection (Optionnel)
    if feature_selection:
        # SelectKBest using f_classif is suitable for numerical features and a classification target
        # Note: This step should come *after* imputation if features had missing values
        selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
        steps.append(('feature_selector', selector))

    # Create the pipeline
    pipeline = Pipeline(steps)

    return pipeline

# --- Example Usage (after the preceding code has been executed) ---

# Assume AVAILABLE_DATAFRAMES is populated from section 6
if AVAILABLE_DATAFRAMES:
    print("\n🚀 Création de pipelines de prétraitement pour la modélisation...")

    # Example: Create a pipeline using KNN imputation, no outlier handling within the pipeline,
    # and removing collinear features (assuming correlated_info is available).
    try:
        knn_pipeline = create_preprocessing_pipeline(
            imputation_method='knn',
            outlier_handling=False, # Handled before pipeline if removing rows
            feature_selection=False # Optional: Add feature selection here if needed
        )
        print("\nPipeline KNN créé:")
        print(knn_pipeline)

        # Example: Create a pipeline using MICE imputation
        mice_pipeline = create_preprocessing_pipeline(
            imputation_method='mice',
            outlier_handling=False,
            feature_selection=False
        )
        print("\nPipeline MICE créé:")
        print(mice_pipeline)

        # --- How to use the pipeline ---
        # Choose one of your preprocessed dataframes (e.g., KNN no outliers)
        df_train_preprocessed = AVAILABLE_DATAFRAMES['knn_no_outliers'].copy()

        # Separate features (X) and target (y)
        if 'y' in df_train_preprocessed.columns:
            X_train = df_train_preprocessed.drop('y', axis=1)
            y_train = df_train_preprocessed['y']

            # Fit the pipeline on the training data
            print("\nFitting KNN pipeline on training data...")
            knn_pipeline.fit(X_train, y_train)
            print("Pipeline fitted.")

            # Now you can use the fitted pipeline to transform new data (e.g., test data)
            # transformed_X_train = knn_pipeline.transform(X_train)
            # print(f"Transformed training data shape: {transformed_X_train.shape}")

            # # Example: If you had df_eval loaded and preprocessed similarly up to the point
            # # of needing imputation and collinearity removal within the pipeline context:
            # # Load df_eval or assume it's in memory
            # df_eval_preprocessed = ... # Load and apply initial steps like transformation to df_eval
            # X_eval = df_eval_preprocessed.drop('y', axis=1) # Assuming df_eval might have a dummy 'y' or only features
            # transformed_X_eval = knn_pipeline.transform(X_eval)
            # print(f"Transformed evaluation data shape: {transformed_X_eval.shape}")

            # --- Modélisation ---
            # You would now add your modeling step to the pipeline or use the
            # transformed data to train a model.

            # Example: Add a model to the pipeline (e.g., Logistic Regression or RandomForest)
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier

            # Pipeline with a model
            pipeline_with_model = Pipeline(knn_pipeline.steps + [
                ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
                # Or use 'mice_pipeline' here if you prefer MICE imputation
            ])

            print("\nPipeline avec modèle créé:")
            print(pipeline_with_model)

            # Fit the full pipeline (preprocessing + model)
            print("\nFitting full pipeline...")
            pipeline_with_model.fit(X_train, y_train)
            print("Full pipeline fitted.")

            # Now you can make predictions on new data
            # predictions = pipeline_with_model.predict(X_eval)
            # predict_proba = pipeline_with_model.predict_proba(X_eval)[:, 1] # For probability of the positive class

            # To evaluate, you'd typically split X_train, y_train into train/validation sets
            # or use cross-validation.
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import f1_score, make_scorer

            f1_scorer = make_scorer(f1_score)

            print("\nPerforming cross-validation (F1-score)...")
            # Using X_train, y_train for cross-validation
            cv_scores = cross_val_score(pipeline_with_model, X_train, y_train, cv=5, scoring=f1_scorer)

            print(f"\nCross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1-score: {cv_scores.mean():.4f}")
            print(f"Std Dev of CV F1-score: {cv_scores.std():.4f}")

            print("\n🚀 Pipeline complet (preprocessing + modélisation) prêt à l'emploi.")

        else:
             print("Error: 'y' column not found in the selected dataframe.")

    except NameError as e:
        print(f"\nError: Required variables not found. Please ensure preceding cells were run.")
        print(f"Missing variable: {e}")
        print("Make sure variables like 'correlated_info' and 'AVAILABLE_DATAFRAMES' are defined.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

else:
    print("\n⚠️ AVAILABLE_DATAFRAMES is empty. Cannot create or fit pipelines.")
    print("Please run the data generation section (Section 6) first.")
