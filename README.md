# mlops-assignment
This project demonstrates a simple API for serving a pre-trained scikit-learn model (RandomForestClassifier trained on the Iris dataset) using FastAPI, Docker, Docker Compose, and GitHub Actions for CI/CD.  It emphasizes operational best practices such as containerization, automated testing, logging, and monitoring.

## Prerequisites

*   Docker: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
*   Docker Compose (optional but recommended):  Usually included with Docker Desktop or installable separately.
*   Python 3.9+

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone <your_repository_url>
    cd <repository_directory>
    ```

2.  **Build and run the Docker container:**

    **Using Docker Compose (Recommended):**

    ```bash
    docker-compose up --build
    ```

    This command builds the Docker image (if it doesn't exist) and starts the container in detached mode.

    **Without Docker Compose:**

    ```bash
    docker build -t iris-api .
    docker run -p 8000:8000 iris-api
    ```

## Using the API

The API exposes the following endpoints:

*   `/predict`: For making predictions. Accepts a JSON payload with Iris flower measurements.
*   `/health`: A health check endpoint.
*   `/metrics`:  Exposes basic metrics about the API's performance (request count, error rate, average latency).

### Sample Request (Prediction)

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}' http://localhost:8000/predict

Markdown
Expected Response (Prediction):

{"prediction": 0}

Json
(The prediction value will be 0, 1, or 2, corresponding to the predicted Iris species.)

Health Check
curl http://localhost:8000/health

Bash
Expected Response (Health Check):

{"status": "healthy"}

Json
Metrics Endpoint
curl http://localhost:8000/metrics

Bash
Example Response (Metrics):

{
    "total_requests": 15,
    "error_count": 0,
    "average_latency": 0.012
}

Jsontotal_requests: The total number of prediction requests served by the API.

error_count: The number of prediction requests that resulted in an error (HTTP 500).

average_latency: The average time (in seconds) taken to process a prediction request.

CI/CD Pipeline (GitHub Actions)
This project uses GitHub Actions for automated CI/CD. The pipeline is defined in .github/workflows/main.yml and performs the following steps:

Checkout Code: Checks out the code from the repository.

Set up Python: Sets up a Python 3.9 environment.

Install Dependencies: Installs project dependencies from requirements.txt, including fastapi, uvicorn, scikit-learn, joblib, pandas, pytest, and requests.

Linting: Runs flake8 to check for code style issues and potential errors. This helps ensure code quality and maintainability.

Run Unit Tests: Executes unit tests defined in test_main.py to verify the functionality of individual components.

Run Integration Tests: This is a crucial step. It:

Starts the Docker container in detached mode using docker-compose up -d.

Waits for the container to become ready (10 seconds).

Runs pytest with the -m integration flag to execute tests marked as integration tests. These tests send HTTP requests to the /predict endpoint running inside the container to verify the entire deployment stack.

Stops the Docker container using docker-compose down. This ensures that resources are cleaned up.

Build and Push Docker Image: Builds a Docker image and pushes it to Docker Hub (if the workflow is triggered on the main branch). It uses the GitHub Actions secrets DOCKERHUB_USERNAME and DOCKERHUB_TOKEN for authentication. The image is tagged with the Git SHA and latest.

Logging and Monitoring
The API logs inference requests and responses in JSON format using the Python logging module. Each log entry includes:

request_id: A unique ID for the request (if provided in the request headers).

input_data: The input data provided to the /predict endpoint.

prediction: The predicted Iris species.

latency: The time taken to process the request (in seconds).

error: If an error occurred, the error message and traceback are logged.

Example Log Entry (JSON):

{"request_id": "12345", "input_data": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "prediction": 0, "latency": 0.015}

Json
The API also exposes a /metrics endpoint that provides basic metrics about the API's performance (total requests, error count, average latency). These metrics are stored in memory (for simplicity) and are reset when the API restarts.

Integration with Monitoring Systems:

For a production environment, you would integrate these logs and metrics with a dedicated monitoring system. Here's how you could approach it:

Logging:

Use a logging driver (e.g., fluentd, gelf) to forward container logs to a central log aggregation system like the ELK stack (Elasticsearch, Logstash, Kibana) or Splunk.

Configure Logstash to parse the JSON log entries and extract the relevant fields.

Use Kibana to visualize the logs, create dashboards, and set up alerts based on specific events or error patterns.

Metrics:

Use a metrics collector like Prometheus to scrape the /metrics endpoint periodically.

Configure Prometheus to store the metrics data in a time-series database.

Use Grafana to create dashboards and visualize the metrics, allowing you to monitor the API's performance over time and identify potential issues. You could set up alerts in Grafana to notify you of high latency or error rates.

Environment Variables
APP_ENVIRONMENT: (Example) Can be used to configure the application's behavior based on the environment (e.g., development, production). Currently not extensively used, but included to demonstrate understanding of configuration management.

Scaling and Future Improvements
This project can be further improved and scaled in several ways:

Load Balancing: Deploy the API behind a load balancer (e.g., Nginx, HAProxy) with multiple instances of the container running. Kubernetes could be used to orchestrate the deployment and scaling of these instances.

Auto-scaling: Implement auto-scaling based on CPU utilization or request latency using Kubernetes HPA (Horizontal Pod Autoscaler).

Model Versioning: Implement a model versioning system to track different versions of the trained model. This could involve storing models in a separate repository (e.g., AWS S3) and updating the API to load the correct version based on a configuration setting.

A/B Testing: Implement A/B testing by deploying multiple versions of the model and routing traffic to each version based on a defined percentage. This allows for comparing the performance of different models in a real-world setting.

Security: Implement authentication and authorization to secure the API endpoints. This could involve using API keys, JWT tokens, or OAuth 2.0.

Request Validation: Implement request validation to validate the input data against a schema, ensuring that the data is in the correct format and meets the required constraints. Pydantic makes this easy.

Model Retraining: Implement a scheduled model retraining pipeline using tools like Airflow or cron jo