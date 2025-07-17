## Workflow complet de stacking sans refit

import numpy as np
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def run_complete_stacking_workflow():
    """
    Exécute le workflow complet de stacking sans refit
    """
    print("🚀 DÉMARRAGE DU WORKFLOW COMPLET DE STACKING")
    print("="*60)
    
    # 1. Test du workflow
    print("\n1️⃣ Test du workflow...")
    try:
        exec(open('modules/modeling/test_stacking_workflow.py').read())
        print("✅ Tests passés")
    except Exception as e:
        print(f"❌ Erreur lors des tests : {e}")
        return False
    
    # 2. Stacking KNN
    print("\n2️⃣ Stacking sans refit - KNN...")
    try:
        exec(open('modules/modeling/stacking_no_refit_knn.py').read())
        print("✅ Stacking KNN terminé")
    except Exception as e:
        print(f"❌ Erreur stacking KNN : {e}")
        return False
    
    # 3. Stacking MICE
    print("\n3️⃣ Stacking sans refit - MICE...")
    try:
        exec(open('modules/modeling/stacking_no_refit_mice.py').read())
        print("✅ Stacking MICE terminé")
    except Exception as e:
        print(f"❌ Erreur stacking MICE : {e}")
        return False
    
    # 4. Comparaison
    print("\n4️⃣ Comparaison des résultats...")
    try:
        exec(open('modules/modeling/compare_stacking_no_refit.py').read())
        print("✅ Comparaison terminée")
    except Exception as e:
        print(f"❌ Erreur comparaison : {e}")
        return False
    
    # 5. Résumé
    print("\n5️⃣ Création du résumé...")
    try:
        exec(open('modules/modeling/stacking_summary.py').read())
        print("✅ Résumé créé")
    except Exception as e:
        print(f"❌ Erreur résumé : {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 WORKFLOW COMPLET TERMINÉ AVEC SUCCÈS !")
    print("="*60)
    print("✅ Stacking KNN et MICE exécutés")
    print("✅ Comparaisons générées")
    print("✅ Résumés créés")
    print("✅ Tous les fichiers sauvegardés")
    print("="*60)
    
    return True

def display_final_results():
    """
    Affiche un résumé final des résultats
    """
    print("\n📊 RÉSUMÉ FINAL DES RÉSULTATS")
    print("="*60)
    
    try:
        # Charger les performances
        with open(stacking_dir / "stack_no_refit_knn_performance.json", "r") as f:
            perf_knn = json.load(f)
        
        with open(stacking_dir / "stack_no_refit_mice_performance.json", "r") as f:
            perf_mice = json.load(f)
        
        # Créer un tableau de comparaison
        comparison_data = {
            "Méthode": ["KNN", "MICE"],
            "F1-Score": [perf_knn["f1_score"], perf_mice["f1_score"]],
            "Précision": [perf_knn["precision"], perf_mice["precision"]],
            "Rappel": [perf_knn["recall"], perf_mice["recall"]],
            "Seuil optimal": [perf_knn["threshold"], perf_mice["threshold"]]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Déterminer le meilleur
        best_idx = df_comparison["F1-Score"].idxmax()
        best_method = df_comparison.loc[best_idx, "Méthode"]
        best_f1 = df_comparison.loc[best_idx, "F1-Score"]
        
        print(f"\n🏆 Meilleur modèle : {best_method} (F1 = {best_f1:.4f})")
        
        # Différences
        f1_diff = abs(perf_knn["f1_score"] - perf_mice["f1_score"])
        print(f"📊 Différence de F1-score : {f1_diff:.4f}")
        
        if f1_diff < 0.01:
            print("💡 Les deux méthodes sont très similaires")
        elif f1_diff < 0.05:
            print("💡 Différence modérée entre les méthodes")
        else:
            print("💡 Différence notable entre les méthodes")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'affichage des résultats : {e}")

def list_generated_files():
    """
    Liste tous les fichiers générés
    """
    print("\n📁 FICHIERS GÉNÉRÉS")
    print("="*60)
    
    expected_files = [
        "stack_no_refit_knn.joblib",
        "stack_no_refit_mice.joblib",
        "best_thr_stack_no_refit_knn.json",
        "best_thr_stack_no_refit_mice.json",
        "stack_no_refit_knn_performance.json",
        "stack_no_refit_mice_performance.json",
        "comparison_stacking_no_refit.csv",
        "comparison_stacking_no_refit.png",
        "stacking_summary.json",
        "stacking_summary.md"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_name in expected_files:
        file_path = stacking_dir / file_name
        if file_path.exists():
            existing_files.append(file_name)
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"✅ {file_name} ({file_size:.1f} KB)")
        else:
            missing_files.append(file_name)
            print(f"❌ {file_name}")
    
    print(f"\n📊 Total : {len(existing_files)}/{len(expected_files)} fichiers générés")
    
    if missing_files:
        print(f"⚠️  Fichiers manquants : {len(missing_files)}")
    else:
        print("🎉 Tous les fichiers sont présents !")

def provide_next_steps():
    """
    Fournit les prochaines étapes
    """
    print("\n🎯 PROCHAINES ÉTAPES")
    print("="*60)
    
    print("1️⃣ Pour faire des prédictions finales :")
    print("   from modules.modeling.final_predictions_stacking import make_final_predictions")
    print("   results_df, model_info = make_final_predictions(X_test, stacking_dir, output_dir)")
    
    print("\n2️⃣ Pour charger un modèle spécifique :")
    print("   from modules.modeling.load_and_use_stacking_no_refit import load_stacking_no_refit_knn")
    print("   stack_model, threshold = load_stacking_no_refit_knn(stacking_dir)")
    
    print("\n3️⃣ Pour analyser les résultats :")
    print("   - Consultez stacking_summary.md pour la documentation")
    print("   - Regardez comparison_stacking_no_refit.png pour les graphiques")
    print("   - Vérifiez les métriques dans les fichiers JSON")
    
    print("\n4️⃣ Pour le rapport final :")
    print("   - Utilisez les résultats du meilleur modèle")
    print("   - Documentez la méthodologie de stacking")
    print("   - Comparez avec les modèles individuels")

# Exécution du workflow complet
if __name__ == "__main__":
    print("🚀 WORKFLOW COMPLET DE STACKING SANS REFIT")
    print("="*60)
    
    # Exécuter le workflow
    success = run_complete_stacking_workflow()
    
    if success:
        # Afficher les résultats
        display_final_results()
        
        # Lister les fichiers
        list_generated_files()
        
        # Prochaines étapes
        provide_next_steps()
        
        print("\n" + "="*60)
        print("🎉 WORKFLOW TERMINÉ AVEC SUCCÈS !")
        print("="*60)
    else:
        print("\n❌ Le workflow a échoué")
        print("Vérifiez les erreurs ci-dessus") 