#!/bin/bash

# Docker setup script for Smart Real Estate AI Platform
set -e

echo "🚀 Setting up Smart Real Estate AI Platform with Docker"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Parse command line arguments
ENVIRONMENT="dev"
BUILD_IMAGES=false
SEED_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --build)
            BUILD_IMAGES=true
            shift
            ;;
        --seed)
            SEED_DATA=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV     Environment to run (dev, prod) [default: dev]"
            echo "  --build       Force rebuild of Docker images"
            echo "  --seed        Seed database with sample data"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Environment: $ENVIRONMENT"

# Set compose files based on environment
COMPOSE_FILES="-f docker-compose.yml"
if [ "$ENVIRONMENT" = "dev" ]; then
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.dev.yml"
elif [ "$ENVIRONMENT" = "prod" ]; then
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.prod.yml"
fi

print_status "Using compose files: $COMPOSE_FILES"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from .env.example"
    cp .env.example .env
    print_warning "Please update .env file with your configuration"
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose $COMPOSE_FILES down

# Build images if requested
if [ "$BUILD_IMAGES" = true ]; then
    print_status "Building Docker images..."
    docker-compose $COMPOSE_FILES build --no-cache
fi

# Start infrastructure services first
print_status "Starting infrastructure services..."
docker-compose $COMPOSE_FILES up -d db redis

# Wait for database to be ready
print_status "Waiting for database to be ready..."
sleep 10

# Run database migrations
print_status "Running database migrations..."
docker-compose $COMPOSE_FILES run --rm gateway python manage.py migrate

# Create superuser if in development
if [ "$ENVIRONMENT" = "dev" ]; then
    print_status "Creating superuser (admin/admin)..."
    docker-compose $COMPOSE_FILES run --rm gateway python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin')
    print('Superuser created successfully')
else:
    print('Superuser already exists')
"
fi

# Seed database if requested
if [ "$SEED_DATA" = true ]; then
    print_status "Seeding database with sample data..."
    docker-compose $COMPOSE_FILES run --rm gateway python manage.py seed_sample_data
fi

# Start all services
print_status "Starting all microservices..."
docker-compose $COMPOSE_FILES up -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 15

# Check service health
print_status "Checking service health..."
docker-compose $COMPOSE_FILES run --rm gateway python manage.py check_services

print_success "🎉 Smart Real Estate AI Platform is now running!"
echo ""
echo "📋 Service URLs:"
echo "  🚪 API Gateway:        http://localhost:8000"
echo "  🏠 Properties Service: http://localhost:8001"
echo "  👥 Contacts Service:   http://localhost:8002"
echo "  🎯 Recommendations:    http://localhost:8003"
echo "  🤖 AI Agents:          http://localhost:8004"
echo "  📊 Analytics:          http://localhost:8005"
echo "  💰 Quotes:             http://localhost:8006"
echo ""
echo "🔧 Management URLs:"
echo "  📚 API Documentation:  http://localhost:8000/api/docs/"
echo "  🌸 Flower (Celery):    http://localhost:5555"

if [ "$ENVIRONMENT" = "dev" ]; then
    echo "  🗄️  PgAdmin:            http://localhost:5050"
    echo "  🔴 Redis Commander:    http://localhost:8081"
fi

echo ""
echo "🚀 Gateway Endpoints:"
echo "  📡 Service Discovery:   http://localhost:8000/gateway/discovery/"
echo "  ❤️  Health Check:       http://localhost:8000/gateway/health/"
echo "  📊 Metrics:            http://localhost:8000/gateway/metrics/"
echo ""
echo "Use 'docker-compose logs -f [service-name]' to view logs"
echo "Use 'docker-compose down' to stop all services"