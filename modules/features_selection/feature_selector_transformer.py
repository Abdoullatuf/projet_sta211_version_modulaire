# modules/feature_selection/feature_selector_transformer.py
"""
Transformer personnalisé pour intégrer la sélection de variables dans un pipeline.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureSelectorByMask(BaseEstimator, TransformerMixin):
    """
    Transformer qui sélectionne les features basées sur un masque booléen.
    Utile pour intégrer la sélection RFECV dans un pipeline Scikit-learn.
    """

    def __init__(self, support_mask=None):
        """
        Args:
            support_mask (array-like of bool, optional): Masque de sélection (True pour conserver).
                                                         Si None, toutes les features sont conservées.
        """
        self.support_mask = np.asarray(support_mask) if support_mask is not None else None

    def fit(self, X, y=None):
        """
        Stocke la forme des données pour validation.
        """
        self.n_features_in_ = X.shape[1]
        if self.support_mask is not None:
            if len(self.support_mask) != self.n_features_in_:
                raise ValueError("La longueur du masque ne correspond pas au nombre de features d'entrée.")
        return self

    def transform(self, X):
        """
        Applique la sélection de features.
        """
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X a {X.shape[1]} features, mais le fit a été fait sur {self.n_features_in_}.")

        if self.support_mask is None:
            return X # Pas de sélection
        else:
            return X[:, self.support_mask]

    def get_feature_names_out(self, input_features=None):
        """
        Retourne les noms des features sélectionnées.
        """
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if self.support_mask is None:
            return np.array(input_features)
        else:
            return np.array(input_features)[self.support_mask]

# --- Exemple d'utilisation dans un pipeline ---
# from sklearn.pipeline import Pipeline
# preprocessor = ... # Votre pipeline de prétraitement existant
# rfecv_support_mask = results_gradboost['rfecv']['support'] # Obtenue de analyze_feature_importance
#
# pipeline_with_selection = Pipeline([
#     ('preprocessor', preprocessor),
#     ('selector', FeatureSelectorByMask(support_mask=rfecv_support_mask)),
#     ('classifier', votre_nouveau_classifieur) # Par exemple, un nouveau GradBoost
# ])
# pipeline_with_selection.fit(X_train_mice, y_train_mice)
# predictions = pipeline_with_selection.predict(X_test_mice)