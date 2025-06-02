# Utiliser une image de base avec Python 3.9
FROM python:3.9-slim AS base

# Installer Airflow à partir de l'image officielle d'Airflow
FROM apache/airflow:2.10.3

# Définir le répertoire de travail
WORKDIR /opt/airflow

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Mettre à jour pip et installer les dépendances Python à partir de requirements.txt
RUN pip install --upgrade pip
#RUN pip install -r requirements.txt

# Installer yamllint pour la validation YAML
RUN pip install yamllint

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Copier les fichiers nécessaires dans le conteneur (DAGs et logs)
COPY dags /opt/airflow/dags
# COPY logs /opt/airflow/logs
COPY data /opt/airflow/data

# Créer le répertoire logs dans le conteneur
RUN mkdir -p /opt/airflow/logs && chmod -R 755 /opt/airflow/logs

# Exposer le port 8080 pour accéder à l'interface web d'Airflow
EXPOSE 8080

# Définir la commande par défaut pour démarrer Airflow (Webserver)
CMD ["airflow", "webserver"]
