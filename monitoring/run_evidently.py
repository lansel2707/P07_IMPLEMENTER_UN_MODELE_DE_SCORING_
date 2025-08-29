# monitoring/run_evidently.py
from pathlib import Path
import os
import numpy as np
import pandas as pd

from evidently import Report
from evidently.metric_preset import DataDriftPreset


# chemins (relatifs à la racine du projet)
ROOT = Path(__file__).resolve().parents[1]
REF_CSV = ROOT / "monitoring" / "data" / "reference.csv"
CUR_CSV = ROOT / "monitoring" / "data" / "current.csv"
OUT_HTML = ROOT / "reports" / "drift_report.html"

def _synth(n=500, seed=42):
    """Jeu de données synthétique si on n'a pas de CSV légers (fixtures)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "montant_en_retard": rng.normal(100, 30, n),
        "nb_previous": rng.poisson(2.5, n),
        "taux_refus": rng.beta(2, 8, n),
        "montant_moyen_pret": rng.normal(5000, 1500, n),
        "nb_paiements": rng.poisson(10, n),
        "retard_moyen": rng.normal(5, 1.5, n),
        "montant_paiement_moyen": rng.normal(250, 60, n),
    })
    return df

def _load_pairs():
    """Charge reference/current si présents, sinon crée des fixtures automatiquement."""
    # 1) si des CSV existent dans monitoring/data/, on les utilise
    if REF_CSV.exists() and CUR_CSV.exists():
        return pd.read_csv(REF_CSV), pd.read_csv(CUR_CSV)

    # 2) sinon, si un X_test existe (petit), on fait deux échantillons
    x_test = ROOT / "notebooks" / "X_test_all.csv"
    if x_test.exists():
        X = pd.read_csv(x_test)
        n = min(500, len(X))
        ref = X.sample(n=n, random_state=42)
        cur = X.sample(n=n, random_state=43)
        return ref, cur

    # 3) fallback : données synthétiques (aucun nouveau fichier n'est créé/commité)
    ref = _synth(500, seed=42)
    cur = _synth(500, seed=43)
    # on force un léger décalage pour que le drift soit visible
    cur["taux_refus"] = cur["taux_refus"] * 1.2
    return ref, cur

def main():
    ref, cur = _load_pairs()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(str(OUT_HTML))

    try:
        rel = OUT_HTML.relative_to(ROOT)
    except Exception:
        rel = OUT_HTML
    print(f"✅ Rapport Evidently généré -> {rel} (ref={len(ref)}, cur={len(cur)})")

if __name__ == "__main__":
    main()
