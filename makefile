.PHONY: install lint format security prepare train test ci push run stop clean build 
IMAGE_NAME=nada230/nada-latrach-4ds1-mlops
TAG=1.0.0 

# Install dependencies
install:
	@pip install -r requirements.txt

# Lint and format checks
lint:
	@flake8 .
	@black --check .

# Auto-format code
format:
	@black .

# Security checks
security:
	@bandit -r .

# Prepare data
train:
	@python main.py --train_path churn-bigml-80.csv --test_path churn-bigml-20.csv --train 

# Train model (depends on prepared data)
evaluate: prepare
	@python main.py --train_path churn-bigml-80.csv --test_path churn-bigml-20.csv --train --evaluate 

# Run tests
save:
	@python main.py --train_path churn-bigml-80.csv --test_path churn-bigml-20.csv --train --evaluate --save model.pkl

# Start FastAPI server
start-api:
	@echo "Starting FastAPI server..."
	. venv/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000 & sleep 3
	@echo "Swagger UI should be available at http://127.0.0.1:8000/docs"
	@echo "Open it in your browser to test the API."

# Build Docker image
build:
	@echo "image construction docker..."
	sudo docker build -t $(IMAGE_NAME):$(TAG) .
	
run:
	@echo "Launch the container..."
	sudo docker run -d -p 8002:8000 --name contain $(IMAGE_NAME):$(TAG) 
	
stop:
	@echo "stop and remove the container..."
	sudo docker stop contain && docker rm contain

push: build
	@echo "push the image into docker HUb..."
	sudo docker push $(IMAGE_NAME):$(TAG) 

docker-clean:
	@echo "clean images and DOcker containers..."
	sudo docker system prune -f 


# Test API via Swagger
test-api: start-api
	@echo "Testing API via Swagger UI..."

# CI pipeline (lint, security, test)
ci: lint security test

# Set default target to run CI steps
.DEFAULT_GOAL := ci

