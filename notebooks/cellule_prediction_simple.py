# =============================================================================
# CELLULE DE PRÉDICTION SIMPLE - NOTEBOOK 03
# =============================================================================
# 
# Cette cellule importe et exécute le script prediction_submission.py
# pour générer les prédictions finales du challenge STA211
# 
# Résultat : Fichier de soumission conforme aux exigences R
# =============================================================================

print("🚀 DÉMARRAGE DE LA GÉNÉRATION DES PRÉDICTIONS FINALES")
print("=" * 60)

# -----------------------------------------------------------------------------
# IMPORT ET EXÉCUTION DU SCRIPT DE PRÉDICTION
# -----------------------------------------------------------------------------

# Option 1 : Exécution directe du script (avec chemin corrigé)
print("\n📂 Option 1 : Exécution directe du script...")
try:
    # Chemin vers le script depuis le dossier notebooks/
    script_path = "../prediction_submission.py"
    
    # Exécuter le script prediction_submission.py
    exec(open(script_path).read())
    print("✅ Script exécuté avec succès !")
except Exception as e:
    print(f"❌ Erreur lors de l'exécution directe : {e}")
    
    # Option 2 : Import et exécution via fonction
    print("\n📂 Option 2 : Import et exécution via fonction...")
    try:
        import sys
        from pathlib import Path
        
        # Ajouter le répertoire racine au path si nécessaire
        root_dir = Path.cwd().parent  # Remonter d'un niveau depuis notebooks/
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        
        # Importer la fonction predict du script
        from prediction_submission import predict
        
        # Exécuter la fonction predict
        predict(
            models_dir=root_dir / "models",
            test_data_path=root_dir / "data/raw/data_test.csv",
            output_dir=root_dir / "outputs/predictions"
        )
        print("✅ Fonction predict exécutée avec succès !")
        
    except Exception as e2:
        print(f"❌ Erreur lors de l'import : {e2}")
        
        # Option 3 : Exécution via subprocess
        print("\n📂 Option 3 : Exécution via subprocess...")
        try:
            import subprocess
            import sys
            
            # Exécuter le script Python depuis le répertoire racine
            root_dir = Path.cwd().parent
            result = subprocess.run([
                sys.executable, 
                "prediction_submission.py"
            ], capture_output=True, text=True, cwd=root_dir)
            
            if result.returncode == 0:
                print("✅ Script exécuté via subprocess avec succès !")
                print("📋 Sortie :")
                print(result.stdout)
            else:
                print(f"❌ Erreur subprocess : {result.stderr}")
                
        except Exception as e3:
            print(f"❌ Erreur subprocess : {e3}")

# -----------------------------------------------------------------------------
# VÉRIFICATION DES FICHIERS GÉNÉRÉS
# -----------------------------------------------------------------------------
print("\n📁 VÉRIFICATION DES FICHIERS GÉNÉRÉS")
print("=" * 40)

# Chemin vers les fichiers depuis le dossier notebooks/
root_dir = Path.cwd().parent
output_dir = root_dir / "outputs/predictions"
submission_path = output_dir / "predictions_finales_stacking_knn_submission.csv"
detailed_path = output_dir / "predictions_finales_stacking_knn_detailed.csv"

# Vérifier si les fichiers existent
if submission_path.exists():
    print(f"✅ Fichier de soumission trouvé : {submission_path}")
    
    # Vérifier le nombre de lignes
    with open(submission_path, 'r') as f:
        lines = f.readlines()
        n_lines = len(lines)
    
    print(f"📊 Nombre de lignes : {n_lines}")
    print(f"✅ Conforme aux exigences (820 lignes) : {'OUI' if n_lines == 820 else 'NON'}")
    
    # Vérifier le contenu
    if lines:
        first_line = lines[0].strip()
        print(f"📝 Première ligne : '{first_line}'")
        print(f"✅ Pas d'en-tête : {'OUI' if first_line in ['ad.', 'noad.'] else 'NON'}")
        
        # Compter les prédictions
        ad_count = sum(1 for line in lines if line.strip() == 'ad.')
        noad_count = sum(1 for line in lines if line.strip() == 'noad.')
        
        print(f"📊 Distribution :")
        print(f"   - Publicités (ad.) : {ad_count} ({ad_count/n_lines*100:.1f}%)")
        print(f"   - Non-publicités (noad.) : {noad_count} ({noad_count/n_lines*100:.1f}%)")
    
else:
    print(f"❌ Fichier de soumission non trouvé : {submission_path}")

if detailed_path.exists():
    print(f"✅ Fichier détaillé trouvé : {detailed_path}")
else:
    print(f"⚠️ Fichier détaillé non trouvé : {detailed_path}")

# -----------------------------------------------------------------------------
# RÉCAPITULATIF FINAL
# -----------------------------------------------------------------------------
print("\n🎉 RÉCAPITULATIF FINAL")
print("=" * 40)

if submission_path.exists() and n_lines == 820:
    print("✅ SUCCÈS : Prédictions générées avec succès !")
    print("✅ Fichier de soumission conforme aux exigences R")
    print("✅ Prêt pour la soumission au challenge STA211")
    print(f"\n📁 Fichier de soumission : {submission_path}")
else:
    print("❌ ÉCHEC : Problème lors de la génération des prédictions")
    print("🔧 Vérifiez les erreurs ci-dessus et corrigez le script")

print("\n" + "=" * 60)
print("🚀 FIN DE LA GÉNÉRATION DES PRÉDICTIONS")
print("=" * 60) 