pipeline {
    agent any

    environment {
        MLFLOW_URI = "sqlite:///mlflow.db"
        EXPERIMENT_NAME = "churn_prediction_pipeline"
    }

    stages {
        stage('Prepare Data') {
            steps {
                script {
                    echo "🔄 Preparing data..."
                    "python main.py --train_path churn-bigml-80.csv --test_path churn-bigml-20.csv --train --evaluate"
                }
            }
        }

    

        stage('Evaluate Model') {
            when {
                expression { return params.EVALUATE }
            }
            steps {
                script {
                    echo "📊 Evaluating model..."
                    "python main.py --train_path churn-bigml-80.csv --test_path churn-bigml-20.csv --load model.pkl --evaluate"
                }
            }
        }



        stage('Load Model') {
            when {
                expression { return params.LOAD }
            }
            steps {
                script {
                    echo "📥 Loading model..."
                    "python main.py --train_path churn-bigml-80.csv --test_path churn-bigml-20.csv --train --evaluate --save model.pkl"
                }
            }
        }
    }

    post {
        always {
            echo "Cleaning up..."
            // Optional: Add cleanup steps, like removing temporary files.
        }
    }
}
