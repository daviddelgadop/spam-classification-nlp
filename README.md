# Classification de Spams – Projet NLP

Ce projet a pour objectif de détecter automatiquement les emails spam à l’aide de techniques de **Traitement du Langage Naturel (NLP)** et de **machine learning supervisé**.

Il couvre l’ensemble du pipeline, depuis le nettoyage des données textuelles brutes jusqu’à l’évaluation des modèles de classification.

---

## Contexte et Objectifs

L’objectif principal est de construire un système fiable de détection de spams capable de distinguer :
- les emails légitimes (*non-spam*)
- les emails indésirables (*spam*)

Ce projet s’inscrit dans un cadre **académique** et pédagogique en NLP.

---

## Structure du Projet

- `spam_classification.ipynb` : notebook principal du projet  
- `requirements.txt` : dépendances Python  
- `README.md` : documentation

---

## Prérequis

- Ordinateur sous Windows, macOS ou Linux  
- Python **3.9 ou supérieur**  
- Connexion Internet  

Vérifier que Python est installé :
```bash
python --version
```

ou :
```bash
python3 --version
```

---

## Installation – Pas à pas (Débutant)

### 1. Cloner ou récupérer le projet

Si le projet est sur GitHub :
```bash
git clone <url_du_projet>
cd nom_du_projet
```

Sinon, placer les fichiers dans un dossier et se positionner dedans :
```bash
cd chemin/vers/le/dossier
```

---

### 2. Créer un environnement virtuel (recommandé)

Windows :
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux :
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Télécharger les ressources NLTK (si nécessaire)

Lancer Python :
```bash
python
```

Puis :
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

Quitter Python :
```python
exit()
```

---

## Lancer le Projet

### 5. Démarrer Jupyter Notebook

```bash
jupyter notebook
```

Ou :
```bash
jupyter lab
```

Un navigateur s’ouvre automatiquement.

---

### 6. Exécuter le notebook

- Ouvrir `spam_classification.ipynb`
- Exécuter les cellules **dans l’ordre**
- Lire les sorties et graphiques générés

---

## Pipeline NLP (Résumé)

- Nettoyage des emails (MIME, caractères spéciaux)
- Normalisation du texte
- Tokenisation
- Suppression des stopwords (FR / EN)
- Extraction de features (BoW, TF-IDF)
- Équilibrage des classes
- Entraînement et évaluation des modèles

---

## Modèles Utilisés

- Régression Logistique  
- Naive Bayes  
- SVM  

Métriques :
- Accuracy
- Precision
- Recall
- F1-score


---

## Auteur

Projet académique – NLP & Machine Learning - David DELGADO M1
