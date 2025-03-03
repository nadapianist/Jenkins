import argparse
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import os
print("Current working directory:", os.getcwd())

from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    # Set MLflow to use SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Configure MLflow experiment
    experiment_name = "churn_prediction_pipeline"
    mlflow.set_experiment(experiment_name)
    train_data = pd.read_csv('/home/nada//ml_project/churn-bigml-80.csv')

    test_data = pd.read_csv('/home/nada//ml_project/churn-bigml-20.csv')
    # Configurer le logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
     
    # Argument parser
    parser = argparse.ArgumentParser(description="Pipeline de Machine Learning pour la prédiction du churn.")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données.")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle.")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle.")
    parser.add_argument("--save", type=str, help="Sauvegarder le modèle dans un fichier.")
    parser.add_argument("--load", type=str, help="Charger un modèle existant.")
    parser.add_argument("--train_path", type=str, required=True, help="Chemin du fichier CSV d'entraînement.")
    parser.add_argument("--test_path", type=str, required=True, help="Chemin du fichier CSV de test.")

    args = parser.parse_args()

    # Préparation des données
    logging.info("🔄 Chargement et préparation des données...")
    X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)

    model = None

    # Si un modèle doit être chargé
    if args.load:
        logging.info(f"📥 Chargement du modèle depuis {args.load}...")
        model = load_model(args.load)

    # Démarrer une expérience MLflow
    with mlflow.start_run():
        if args.train:
            logging.info("🚀 Entraînement du modèle...")
            model, params = train_model(X_train, y_train)

            if model:
                logging.info(f"✅ Modèle entraîné avec succès. Paramètres : {params}")
                
                # ✅ Log model parameters in MLflow
                for param, value in params.items():
                    mlflow.log_param(param, value)

                # ✅ Log the trained model
                mlflow.sklearn.log_model(model, "model")

        # Si un modèle est disponible, évaluation des performances
        if model and args.evaluate:
            logging.info("📊 Évaluation du modèle...")
            accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

            result_message = (
                f"✅ Résultats de l'évaluation :\n"
                f"- Accuracy: {accuracy:.4f}\n"
                f"- Precision: {precision:.4f}\n"
                f"- Recall: {recall:.4f}\n"
                f"- F1-score: {f1:.4f}"
            )
            logging.info(result_message)
            print(result_message)

            # ✅ Log model metrics in MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

        # Sauvegarde du modèle
        if model and args.save:
            logging.info(f"💾 Sauvegarde du modèle dans {args.save}...")
            save_model(model, args.save)

if __name__ == "__main__":
    main()

