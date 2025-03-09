# Mlops-Assignment
This project showcases fundamental MLOps practices by deploying a simple API to serve a pre-trained scikit-learn model (RandomForestClassifier trained on the Iris dataset). It utilizes FastAPI for the API, Docker for containerization, Docker Compose for local deployment, GitHub Actions for CI/CD, and incorporates basic logging and metrics for monitoring.  The primary goal is to demonstrate a complete, reproducible workflow from model deployment to basic monitoring.

## Key Objectives

This project demonstrates the following key MLOps concepts:

*   **Model Serving:** Exposing a machine learning model as a REST API.
*   **Containerization:** Packaging the API and model into a Docker container for consistent execution across environments.
*   **Infrastructure as Code:** Using Docker Compose to define and manage the application's infrastructure.
*   **CI/CD Pipeline:** Automating the build, testing, and deployment process with GitHub Actions.
*   **Basic Monitoring:** Collecting and exposing basic metrics to track the API's performance and health.

## Prerequisites

*   Docker: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/) - Required for running the containerized API.
*   Docker Compose: Usually included with Docker Desktop or installable separately. Simplifies local deployment.
*   Python 3.9+: Required only if you want to retrain the model locally.

## Getting Started - Step-by-Step

This section provides a detailed, step-by-step guide to getting the API up and running.

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <repository_directory>
    ```

2.  **(Optional) Retrain the Model:**

    The `iris_model.joblib` file contains a pre-trained model. If you wish to experiment with training your own model (e.g., with different parameters), run:

    ```bash
    python model.py
    ```

    This will overwrite the existing `iris_model.joblib` file.  If you skip this step, the pre-trained model will be used.

3.  **Build and Run the Docker Container:**

    **Option 1: Using Docker Compose (Recommended for Simplicity)**

    This is the easiest way to get the API running locally.

    ```bash
    docker-compose up --build
    ```

    This command will:

    *   Build the Docker image based on the `Dockerfile`.
    *   Start the container, mapping port 8000 on your host machine to port 8000 inside the container.
    *   Set the `APP_ENVIRONMENT` environment variable to `production`.
    *   Automatically restart the container if it crashes.

    **Option 2: Using Docker Directly (Alternative)**

    If you prefer not to use Docker Compose, you can build and run the container with these commands:

    ```bash
    docker build -t mlops-api .
    docker run -p 8000:8000 mlops-api (You can tag the image name according to your wish)
    ```

    *   `docker build -t mlops-api .` builds the Docker image and tags it as `mlops-api`.
    *   `docker run -p 8000:8000 mlops-api` runs the container, mapping port 8000.

    After running either of these options, the API will be accessible at `http://localhost:8000`.

## Interacting with the API

The API provides three endpoints:

*   `/predict`:  Predicts the Iris species based on input features.
*   `/health`:  A health check endpoint to verify the API is running.
*   `/metrics`:  Provides basic performance metrics.

### 1. `/predict` Endpoint - Making Predictions

This endpoint accepts a JSON payload containing the four Iris flower measurements: `sepal_length`, `sepal_width`, `petal_length`, and `petal_width`.

**Sample Request (using `curl`):**

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}' http://localhost:8000/predict
```

##Expected Response:

```json
{"prediction": 0}
```

The prediction value (0, 1, or 2) represents the predicted Iris species.


For easier log tracking, you can include an X-Request-ID header with a unique identifier for each request.

````bash
curl -X POST -H "Content-Type: application/json" -H "X-Request-ID: my-unique-id-123" -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}' http://localhost:8000/predict
```

**### 2. `/health` Endpoint - Health Check**


This endpoint returns a simple JSON response indicating the API's health status.

##Sample Request:

````bash
curl http://localhost:8000/health
```

Expected Response:
```json
{"status": "healthy"}
```


**### 3. `/metrics` Endpoint - Performance Metrics**



This endpoint exposes basic metrics about the API's performance.

##Sample Request:

```bash
curl http://localhost:8000/metrics
```

##Example Response:
```json
{
  "total_requests": 10,
  "error_count": 0,
  "average_latency": 0.005
}
```

*   **total_request:**The total number of requests received by the API.

*   **error_count:** The number of requests that resulted in an error.

*   **average_latency:** The average time it takes to process a request (in seconds).

##CI/CD Pipeline - Automated Build, Test, and Deployment

The project includes a GitHub Actions workflow (.github/workflows/main.yml) that automates the following tasks:

1.   **Code Checkout:** Checks out the code from the repository.

2.  **Python Setup:** Sets up Python 3.9+(according the installed version).

3.  **Dependency Installation:**: Installs the required Python packages from requirements.txt.

4.  **Linting:** Uses flake8 to enforce code style and identify potential errors. This ensures code quality and consistency.

5.  **Testing:** Executes unit tests in test_main.py to verify the API's functionality, including validating the model's output.

6.  **Docker Build:** Builds the Docker image.

7.  **Docker Login:** Logs in to Docker Hub using the provided credentials (configured as GitHub secrets).

8.  **Docker Push:** Pushes the Docker image to Docker Hub, tagged with the Git commit SHA and latest.

##Configuration:

   **Github Secrets:** To enable the Docker push step, you need to configure the following secrets in your GitHub repository settings (Settings -> Secrets -> Actions):

        *    DOCKERHUB_USERNAME: Refers to Docker Hub username.
        *    DOCKERHUB_TOKEN: Refers to Docker Hub access token. See Docker Hub documentation for instructions on creating an access token.

##**Triggers:**

The workflow is triggered on:

    -Pushes to the main branch.

    -Pull requests targeting the main branch.

##Logging and Monitoring - Observing the API

The FastAPI application logs inference requests and responses to stdout in JSON format. Each log entry includes:

*   **request_id:** The request ID (if provided).

*   **input_data:** The input features used for prediction.

*   **prediction or error:** The model's prediction or an error message (if an error occurred).

*   **latency:** The time it took to process the request (in seconds).

##Example Log Entry (Successful Prediction):

```json
{"request_id": "my-unique-id-123", "input_data": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "prediction": 0, "latency": 0.004}
```

##Example Log Entry (Error):

```json
{"request_id": "N/A", "input_data": {"sepal_length": 1.0, "sepal_width": 1.0, "petal_length": 1.0, "petal_width": 1.0}, "error": "ValueError('Input X must be non-negative.')", "latency": 0.001}
```

##Integration with Monitoring Systems (Conceptual):

While this project provides basic logging and metrics, a production system would require integration with a dedicated monitoring system. Here's the general approach:

 1.   **Log Aggregation:** Configure Docker to forward stdout logs to a log aggregation service (e.g., Fluentd, Logstash, AWS CloudWatch Logs, Google Cloud Logging).

 2.   **Log Parsing:** The log aggregation service parses the JSON logs to extract relevant fields (e.g., request ID, input features, prediction, error message, latency).

 3.   **Metrics Generation:** The parsed data is used to generate metrics, such as request count, error rate, average latency, and latency percentiles.

 4.   **Visualization and Alerting:** Tools like Grafana, Kibana (part of the ELK stack), or cloud-specific monitoring dashboards are used to visualize the metrics and set up alerts based on predefined thresholds.

 5.   **Expose Metrics (Prometheus):** For Prometheus, you would typically use a Prometheus client library to expose metrics in the required format. This API currently exposes metrics in JSON format via the /metrics endpoint, but a Prometheus exporter would be needed for proper integration.

The provided '/metrics' endpoint offers a starting point for collecting metrics, but it's important to use a more comprehensive solution for a real-world deployment.

#Running Tests - Ensuring API Functionality

To run the unit tests locally, first install the required dependencies:

```bash
pip install -r requirements.txt
pip install pytest
```

Then, execute the tests using pytest:
```bash
pytest test_main.py
```

The tests will verify that the API endpoints are functioning correctly and that the model is producing valid predictions.

##**Potential Next Steps - Scaling and Enhancements**

This project provides a foundation for deploying and monitoring a machine learning model. Here are some potential next steps to enhance the system:

 **Kubernetes Deployment:** Deploy the container to a Kubernetes cluster for increased scalability, resilience, and manageability. Consider using Helm charts to simplify 
 deployment and configuration.

 **Load Balancing:** Implement a load balancer (e.g., Nginx, HAProxy) to distribute traffic across multiple instances of the API server.

 **Security Enhancements:**

    -Implement authentication and authorization to protect the API endpoints.

    -Use HTTPS to encrypt communication.

    -Regularly scan the Docker image for vulnerabilities.

 **Model Monitoring:** Implement a robust model monitoring system to track model performance over time and detect data drift or concept drift. This could involve tracking 
 prediction accuracy, data distributions, and other relevant metrics.

 **A/B Testing:** Implement A/B testing to compare different model versions or API configurations.

 **Advanced Metrics:** Add more granular metrics, such as CPU utilization, memory usage, and network traffic, to gain deeper insights into the API's performance.

 **Retry Mechanism:** Implement a retry mechanism to automatically retry failed requests due to transient errors.

 **Input Validation:** Implement more robust input validation to prevent errors caused by invalid data.
