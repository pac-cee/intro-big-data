# Deployment & Containerization

## Table of Contents
1. [Deployment Options](#deployment-options)
2. [Docker Basics](#docker-basics)
3. [Docker Compose](#docker-compose)
4. [Kubernetes Basics](#kubernetes-basics)
5. [CI/CD Pipelines](#cicd-pipelines)
6. [Monitoring & Logging](#monitoring--logging)
7. [Best Practices](#best-practices)

## Deployment Options

### Traditional Servers
- **Virtual Private Servers (VPS)**: DigitalOcean, Linode, AWS EC2
- **Dedicated Servers**: Full control, more expensive

### Platform as a Service (PaaS)
- **Heroku**: Easy deployment, good for beginners
- **AWS Elastic Beanstalk**: AWS-managed PaaS
- **Google App Engine**: Fully managed serverless platform

### Serverless
- **AWS Lambda**: Run code without managing servers
- **Google Cloud Functions**: Event-driven serverless compute
- **Azure Functions**: Microsoft's serverless offering

## Docker Basics

### Installation
- [Docker Desktop](https://www.docker.com/products/docker-desktop) for Mac/Windows
- Native Docker for Linux

### Basic Commands
```bash
# Build an image
docker build -t myapp .

# Run a container
docker run -p 4000:80 myapp

# List running containers
docker ps

# List all containers
docker ps -a

# Stop a container
docker stop <container_id>

# Remove a container
docker rm <container_id>

# List images
docker images

# Remove an image
docker rmi <image_id>
```

### Dockerfile Example
```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]
```

## Docker Compose

### docker-compose.yml Example
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "4000:80"
    environment:
      - FLASK_ENV=production
    depends_on:
      - redis
    restart: always

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Kubernetes Basics

### Key Concepts
- **Pods**: Smallest deployable units
- **Deployments**: Manage scaling and updates
- **Services**: Network access to pods
- **ConfigMaps & Secrets**: Configuration and secrets

### Basic kubectl Commands
```bash
# Get cluster info
kubectl cluster-info

# Get nodes
kubectl get nodes

# Apply a configuration
kubectl apply -f deployment.yaml

# Get pods
kubectl get pods

# View pod logs
kubectl logs <pod_name>

# Describe a resource
kubectl describe pod <pod_name>

# Delete a resource
kubectl delete -f deployment.yaml
```

### Example Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: LoadBalancer
```

## CI/CD Pipelines

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Login to Docker Hub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_HUB_USERNAME }}
        password: ${{ secrets.DOCKER_HUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v2
      with:
        context: .
        push: true
        tags: username/myapp:latest
    
    - name: Deploy to production
      run: |
        # Add deployment steps here
        echo "Deploying to production..."
```

## Monitoring & Logging

### Tools
- **Prometheus**: Monitoring & alerting
- **Grafana**: Visualization
- **ELK Stack**: Logging (Elasticsearch, Logstash, Kibana)
- **Sentry**: Error tracking

### Basic Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'myapp'
    static_configs:
      - targets: ['localhost:8000']
```

## Best Practices

### Security
- Use minimal base images
- Run as non-root user
- Scan for vulnerabilities
- Use secrets management

### Performance
- Multi-stage builds
- Proper layer caching
- Resource limits
- Health checks

### Operations
- Immutable infrastructure
- Infrastructure as Code (IaC)
- Blue/Green or Canary deployments
- Rollback strategies

## Next Steps
1. Learn about service meshes (Istio, Linkerd)
2. Explore serverless container platforms
3. Study infrastructure as code (Terraform, Pulumi)
4. Implement GitOps workflows
