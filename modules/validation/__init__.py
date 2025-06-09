# ============================================================================
# 3. modules/validation/__init__.py
# ============================================================================

"""
Module de validation et diagnostic des données pour STA211.

Contient les outils pour :
- Diagnostic des dimensions et cohérence
- Validation de la qualité des données
- Comparaison entre datasets
- Génération de rapports de diagnostic
"""

from .data_diagnostics import (
    DataDiagnostics,
    quick_dimension_check,
    compare_manual_vs_pipeline,
    print_diagnostic_summary
)

__all__ = [
    'DataDiagnostics',
    'quick_dimension_check',
    'compare_manual_vs_pipeline',
    'print_diagnostic_summary'
]