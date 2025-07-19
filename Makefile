# Makefile pour la compilation du rapport LaTeX
# Usage: make rapport

# Variables
LATEX = pdflatex
BIBTEX = bibtex
MAIN = rapport_projet_sta211
CLEAN_EXTENSIONS = aux log out toc lot lof blg bbl fdb_latexmk fls synctex.gz

# Cible principale
rapport: $(MAIN).pdf

# Compilation LaTeX
$(MAIN).pdf: $(MAIN).tex
	@echo "Compilation du rapport LaTeX..."
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	@echo "Rapport compilé avec succès: $(MAIN).pdf"

# Nettoyage des fichiers temporaires
clean:
	@echo "Nettoyage des fichiers temporaires..."
	@for ext in $(CLEAN_EXTENSIONS); do \
		find . -name "*.$$ext" -delete; \
	done
	@echo "Nettoyage terminé."

# Nettoyage complet (inclut le PDF)
clean-all: clean
	@echo "Suppression du fichier PDF..."
	@rm -f $(MAIN).pdf
	@echo "Nettoyage complet terminé."

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  make rapport    - Compile le rapport LaTeX"
	@echo "  make clean      - Nettoie les fichiers temporaires"
	@echo "  make clean-all  - Nettoie tout (inclut le PDF)"
	@echo "  make help       - Affiche cette aide"

.PHONY: rapport clean clean-all help 