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
                    sh "python model_pipeline.py --prepare --train_path ${params.TRAIN_PATH} --test_path ${params.TEST_PATH}"
                }
            }
        }

        stage('Train Model') {
            when {
                expression { return params.TRAIN }
            }
            steps {
                script {
                    echo "🚀 Training model..."
                    sh "python model_pipeline.py --train --train_path ${params.TRAIN_PATH} --test_path ${params.TEST_PATH}"
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
                    sh "python model_pipeline.py --evaluate --train_path ${params.TRAIN_PATH} --test_path ${params.TEST_PATH}"
                }
            }
        }

        stage('Save Model') {
            when {
                expression { return params.SAVE }
            }
            steps {
                script {
                    echo "💾 Saving model..."
                    sh "python model_pipeline.py --save 'model_output.pkl' --train_path ${params.TRAIN_PATH} --test_path ${params.TEST_PATH}"
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
                    sh "python model_pipeline.py --load 'model_output.pkl' --train_path ${params.TRAIN_PATH} --test_path ${params.TEST_PATH}"
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
