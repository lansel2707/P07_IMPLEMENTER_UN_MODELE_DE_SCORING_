# Scoring Crédit – Prêt à dépenser

Projet de data science visant à prédire la probabilité de défaut d’un client,  
et à industrialiser le pipeline de scoring de bout en bout.

---

## 📁 Structure du projet

- `notebooks/` : notebooks d'analyse, EDA, feature engineering, modélisation
- `api/` : code de l’API de scoring
- `mlops/` : scripts pour MLflow, Evidently, CI/CD…
- `data/` : données brutes (à ne pas versionner)
- `models/` : modèles exportés
- `utils/` : fonctions utilitaires

---

## 📝 **Checklist projet – étapes clés**

### 1. Versionning & initialisation

- [ ] Dépôt Git/GitHub créé
- [ ] Structure de dossiers conforme
- [ ] Fichier `.gitignore` en place
- [ ] README.md structuré

### 2. Analyse exploratoire (EDA)

- [ ] Chargement et inspection des datasets
- [ ] Statistiques descriptives, valeurs manquantes
- [ ] Premières observations documentées

### 3. Feature engineering

- [ ] Agrégations/fusions sur les tables secondaires
- [ ] Création de nouvelles features métier
- [ ] Encodage, gestion des valeurs manquantes
- [ ] Datasets prêts pour la modélisation

### 4. Modélisation ML

- [ ] Séparation X/y
- [ ] Cross-validation robuste
- [ ] Gestion du déséquilibre des classes
- [ ] Test et tuning de plusieurs modèles (GridSearchCV…)
- [ ] Évaluation par score métier (coût FN/FP), AUC, accuracy

### 5. Optimisation du score métier

- [ ] Fonction de coût métier définie (pondération FN/FP)
- [ ] Optimisation du seuil
- [ ] Comparaison des modèles sur score métier

### 6. Interprétabilité

- [ ] Feature importance globale (modèle)
- [ ] Importance locale (SHAP/LIME)
- [ ] Documentation des variables clés

### 7. Tracking & MLOps

- [ ] MLflow opérationnel (tracking runs, modèles, artefacts)
- [ ] Modèle exporté et versionné

### 8. Détection de Data Drift

- [ ] Evidently installé/configuré
- [ ] Rapports drift générés et interprétés

### 9. Déploiement API

- [ ] API scoring en place (Flask/FastAPI/Streamlit)
- [ ] Tests et déploiement cloud (Render, Huggingface, etc.)
- [ ] Interface de test (Streamlit ou notebook interactif)

### 10. Soutenance & Documentation

- [ ] Synthèse des résultats et arbitrages
- [ ] README et doc à jour
- [ ] Préparation à la soutenance et anticipation des questions

---

## 🚀 Démarrage rapide

1. Cloner le repo  
2. Installer les requirements  
3. Ouvrir les notebooks dans `notebooks/`  
4. Suivre la checklist étape par étape !

---




