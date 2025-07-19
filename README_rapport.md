# Rapport Projet STA211 - Guide de Compilation

## 📋 Description

Ce dossier contient le rapport complet du projet STA211 en format LaTeX. Le rapport présente une analyse approfondie du challenge de classification de publicités internet, incluant l'exploration des données, la modélisation et les résultats obtenus.

## 📁 Fichiers Inclus

- `rapport_projet_sta211.tex` - Source LaTeX du rapport
- `Makefile` - Script de compilation automatisée
- `README_rapport.md` - Ce fichier d'aide

## 🚀 Compilation du Rapport

### Prérequis

Assurez-vous d'avoir une distribution LaTeX installée sur votre système :
- **Windows** : MiKTeX ou TeX Live
- **macOS** : MacTeX
- **Linux** : TeX Live

### Compilation Automatique (Recommandée)

```bash
# Compiler le rapport
make rapport

# Ou utiliser la cible par défaut
make
```

### Compilation Manuelle

```bash
# Première compilation
pdflatex rapport_projet_sta211.tex

# Deuxième compilation (pour les références)
pdflatex rapport_projet_sta211.tex
```

### Nettoyage

```bash
# Nettoyer les fichiers temporaires
make clean

# Nettoyer tout (inclut le PDF)
make clean-all
```

## 📖 Structure du Rapport

### 1. Résumé Exécutif
- Objectifs du projet
- Principales contributions
- Résultats obtenus

### 2. Introduction
- Contexte du challenge STA211
- Description du dataset Internet Advertisements
- Objectifs et contraintes

### 3. Exploration et Nettoyage des Données
- Description du dataset (3279 observations, 1558 variables)
- Analyse des valeurs manquantes (MAR/MCAR)
- Transformations appliquées (Yeo-Johnson, Box-Cox)
- Traitement des outliers

### 4. Analyse Exploratoire
- Analyse univariée et bivariée
- Analyse multivariée avec réduction de dimensionnalité
- Visualisations et insights

### 5. Modélisation Supervisée
- Stratégie de modélisation
- Validation croisée (StratifiedKFold)
- Modèles implémentés :
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - Logistic Regression
  - Multi-Layer Perceptron
- Stacking sans refit

### 6. Optimisation et Évaluation
- Optimisation du seuil (0.2-0.8)
- Métriques de performance
- Gestion des classes déséquilibrées

### 7. Interprétation et Conclusion
- Importance des variables
- Compromis biais-variance
- Comparaison des modèles
- Recommandations

### 8. Prédictions Finales
- Pipeline automatisé
- Résultats de prédiction
- Validation du format

## 📊 Résultats Principaux

### Performance du Modèle Champion
- **Modèle** : Stacking sans refit KNN
- **F1-score** : 0.923
- **Précision** : 0.847
- **Rappel** : 0.912
- **Seuil optimal** : 0.200

### Distribution des Prédictions Finales
- **Total** : 820 prédictions
- **Publicités (ad.)** : 111 (13.5%)
- **Non-publicités (noad.)** : 709 (86.5%)

## 🔧 Personnalisation

### Modifier le Contenu

Pour modifier le rapport, éditez le fichier `rapport_projet_sta211.tex` :

```latex
% Modifier les informations de l'étudiant
{\large\textbf{Étudiant :} Votre Nom\par}

% Ajouter de nouvelles sections
\section{Nouvelle Section}
Contenu de la nouvelle section...
```

### Ajouter des Figures

```latex
% Insérer une figure
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{chemin/vers/figure.png}
    \caption{Description de la figure}
    \label{fig:label}
\end{figure}
```

### Modifier le Style

Les styles sont définis dans l'en-tête du document :

```latex
% Modifier les marges
\geometry{margin=2cm}

% Modifier les couleurs
\definecolor{codegreen}{rgb}{0,0.8,0}
```

## 📝 Notes Techniques

### Packages LaTeX Utilisés
- `amsmath`, `amsfonts`, `amssymb` : Mathématiques
- `booktabs`, `array` : Tableaux
- `graphicx` : Images
- `hyperref` : Liens
- `listings` : Code source
- `fancyhdr` : En-têtes et pieds de page

### Structure du Code
- **Code Python** : Coloration syntaxique avec `listings`
- **Tableaux** : Formatage professionnel avec `booktabs`
- **Équations** : Numérotation automatique
- **Références** : Système de citations

## 🎯 Utilisation Académique

Ce rapport est conçu pour répondre aux exigences du cours STA211 du CNAM :

### Critères Respectés
- ✅ Exploration et nettoyage approfondi
- ✅ Gestion des valeurs manquantes (MCAR/MAR/MNAR)
- ✅ Transformations pour normalité
- ✅ Analyse exploratoire complète
- ✅ Modélisation supervisée avec validation croisée
- ✅ Évaluation avec métriques appropriées
- ✅ Gestion des classes déséquilibrées
- ✅ Interprétation et conclusions

### Format Professionnel
- Page de titre complète
- Table des matières
- Sections structurées
- Code source documenté
- Tableaux et figures
- Bibliographie
- Annexes

## 🚨 Dépannage

### Erreurs de Compilation

**Erreur : "Package not found"**
```bash
# Installer les packages manquants
tlmgr install nom_du_package
```

**Erreur : "File not found"**
```bash
# Vérifier que tous les fichiers sont présents
ls -la *.tex
```

**Erreur : "Encoding"**
```bash
# Utiliser UTF-8
pdflatex -interaction=nonstopmode rapport_projet_sta211.tex
```

### Optimisation

Pour une compilation plus rapide :
```bash
# Compilation en mode batch
pdflatex -interaction=batchmode rapport_projet_sta211.tex
```

## 📞 Support

Pour toute question concernant :
- La compilation LaTeX
- La structure du rapport
- L'ajout de contenu
- Les erreurs techniques

Consultez la documentation LaTeX ou contactez l'auteur du projet.

---

**Note** : Ce rapport est le résultat d'un travail académique dans le cadre du cours STA211 du CNAM. Il présente une approche complète et méthodique de la classification de publicités internet. 