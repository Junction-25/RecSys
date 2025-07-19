# Real Estate Recommendation System - API Testing Guide

## Overview

This guide provides step-by-step instructions for running and testing the real estate recommendation system API using Postman and curl commands.

## System Architecture

The system follows a two-stage retrieval and ranking architecture where a property is first encoded and used to retrieve candidate buyers from a vector database via approximate nearest neighbor (ANN) search, while simultaneously extracting relevant interaction features through a feature store. These candidates and their enriched features are then fed into a ranking model that produces an ordered list of the most suitable buyers for the given property.

## Prerequisites

1. **Python Environment**: Python 3.8+ with Django 5.0
2. **Dependencies**: All requirements installed (`pip install -r requirements.txt`)
3. **Database**: PostgreSQL with sample data loaded
4. **API Testing Tool**: Postman or curl

## Starting the Server

### 1. Navigate to Project Directory
```bash
cd /Users/faycalamrouche/Desktop/RealEstate-agent
```

### 2. Start Django Development Server
```bash
python manage.py runserver 8000
```

The server will start on `http://localhost:8000`

### 3. Verify Server is Running
```bash
curl http://localhost:8000/api/v1/recommendations/health/
```

Expected response:
```json
{
  "status": "healthy",
  "service": "TikTok-like Recommendation Engine",
  "version": "1.0.0",
  "performance_stats": {
    "total_requests": 1,
    "avg_response_time_ms": 100.5,
    "cache_hit_rate": 0.0,
    "ann_retrieval_time_ms": 15.2,
    "ranking_time_ms": 85.3
  },
  "using_gpu": false,
  "components": {
    "property_encoder": "active",
    "vector_database": "active",
    "deepfm_ranker": "active",
    "colbert_encoder": "active"
  }
}
```

## API Endpoints

### Base URL
```
http://localhost:8000/api/v1/recommendations/
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health/` | GET | System health check |
| `/property/{id}/` | GET | Get recommendations for a specific property |
| `/bulk/` | GET | Get bulk recommendations for multiple properties |

## Testing with Postman

### 1. Import Collection

Create a new Postman collection called "Real Estate Recommendations" with the following requests:

### 2. Health Check Request

**Request Name**: Health Check
- **Method**: GET
- **URL**: `http://localhost:8000/api/v1/recommendations/health/`
- **Headers**: 
  ```
  Content-Type: application/json
  Accept: application/json
  ```

### 3. Single Property Recommendation

**Request Name**: Property Recommendations
- **Method**: GET
- **URL**: `http://localhost:8000/api/v1/recommendations/property/500/`
- **Query Parameters**:
  - `max_results`: `10` (optional, default: 20)
  - `min_score`: `0.3` (optional, default: 0.3)
- **Headers**:
  ```
  Content-Type: application/json
  Accept: application/json
  ```

**Expected Response**:
```json
{
  "property_id": "500",
  "recommendations": [
    {
      "buyer_id": "1234",
      "name": "John Doe",
      "email": "john.doe@example.com",
      "phone": "+1234567890",
      "deepfm_score": 0.85,
      "similarity_score": 0.78,
      "rank": 1,
      "explanation": "Excellent match based on AI analysis. Budget exactly matches preference. Location preferences match well.",
      "interaction_features": {
        "budget_match": 1.0,
        "location_match": 0.85,
        "property_type_match": 1.0,
        "area_match": 0.7,
        "rooms_match": 0.6
      }
    }
  ],
  "total_found": 10,
  "processing_time_ms": 125.5,
  "performance_breakdown": {
    "encoding_time_ms": 5.2,
    "retrieval_time_ms": 15.8,
    "ranking_time_ms": 95.5,
    "feature_enrichment_time_ms": 9.0
  },
  "method": "sota_pipeline",
  "cached": false
}
```

### 4. Bulk Property Recommendations

**Request Name**: Bulk Recommendations
- **Method**: GET
- **URL**: `http://localhost:8000/api/v1/recommendations/bulk/`
- **Query Parameters**:
  - `property_ids`: `500,740,1000` (comma-separated property IDs)
  - `max_results`: `5` (optional, per property)
- **Headers**:
  ```
  Content-Type: application/json
  Accept: application/json
  ```

**Expected Response**:
```json
{
  "bulk_recommendations": {
    "500": {
      "property_id": "500",
      "recommendations": [...],
      "total_found": 5,
      "processing_time_ms": 98.2
    },
    "740": {
      "property_id": "740",
      "recommendations": [...],
      "total_found": 5,
      "processing_time_ms": 102.1
    },
    "1000": {
      "property_id": "1000",
      "recommendations": [...],
      "total_found": 5,
      "processing_time_ms": 87.5
    }
  },
  "total_properties": 3,
  "total_processing_time_ms": 287.8,
  "cached_results": 0
}
```

## Testing with cURL Commands

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/api/v1/recommendations/health/" \
  -H "accept: application/json"
```

### 2. Single Property Recommendation
```bash
curl -X GET "http://localhost:8000/api/v1/recommendations/property/500/?max_results=10&min_score=0.3" \
  -H "accept: application/json"
```

### 3. Bulk Recommendations
```bash
curl -X GET "http://localhost:8000/api/v1/recommendations/bulk/?property_ids=500,740,1000&max_results=5" \
  -H "accept: application/json"
```

### 4. Test Different Properties
```bash
# Test property 740
curl -X GET "http://localhost:8000/api/v1/recommendations/property/740/" \
  -H "accept: application/json"

# Test property 1000
curl -X GET "http://localhost:8000/api/v1/recommendations/property/1000/" \
  -H "accept: application/json"

# Test with different parameters
curl -X GET "http://localhost:8000/api/v1/recommendations/property/500/?max_results=20&min_score=0.5" \
  -H "accept: application/json"
```

## Performance Testing

### Load Testing with Multiple Requests
```bash
# Test concurrent requests (run in separate terminals)
for i in {1..10}; do
  curl -X GET "http://localhost:8000/api/v1/recommendations/property/500/" \
    -H "accept: application/json" &
done
wait
```

### Measure Response Times
```bash
# Using time command
time curl -X GET "http://localhost:8000/api/v1/recommendations/property/500/" \
  -H "accept: application/json" > /dev/null

# Using curl's built-in timing
curl -X GET "http://localhost:8000/api/v1/recommendations/property/500/" \
  -H "accept: application/json" \
  -w "Time: %{time_total}s\n" \
  -o /dev/null -s
```

## Postman Collection Setup

### 1. Create Environment Variables
In Postman, create an environment with:
- `base_url`: `http://localhost:8000`
- `api_version`: `v1`

### 2. Pre-request Scripts
Add this to collection pre-request script:
```javascript
// Set timestamp for request tracking
pm.globals.set("timestamp", new Date().toISOString());
```

### 3. Test Scripts
Add this to test scripts for validation:
```javascript
// Test response status
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

// Test response time
pm.test("Response time is less than 5000ms", function () {
    pm.expect(pm.response.responseTime).to.be.below(5000);
});

// Test response structure
pm.test("Response has required fields", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('recommendations');
    pm.expect(jsonData).to.have.property('processing_time_ms');
});

// Log performance metrics
console.log("Response time:", pm.response.responseTime + "ms");
if (pm.response.json().processing_time_ms) {
    console.log("Server processing time:", pm.response.json().processing_time_ms + "ms");
}
```

## Troubleshooting

### Common Issues

1. **Server Not Starting**
   ```bash
   # Check if port 8000 is in use
   lsof -i :8000
   
   # Kill existing process if needed
   kill <PID>
   ```

2. **Database Connection Issues**
   ```bash
   # Check database status
   python manage.py check --database default
   
   # Run migrations if needed
   python manage.py migrate
   ```

3. **Empty Recommendations**
   - Ensure sample data is loaded in the database
   - Check if embeddings are computed for buyers
   - Verify property exists in database

4. **Slow Response Times**
   - Check server logs for bottlenecks
   - Monitor memory usage
   - Verify optimizations are applied

### Debug Mode
Enable debug logging by setting in Django settings:
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'apps.recommendations': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}
```

## Performance Benchmarks

### Expected Performance (Optimized System)
- **Health Check**: < 50ms
- **Single Property Recommendation**: < 200ms
- **Bulk Recommendations (3 properties)**: < 500ms
- **Concurrent Users (10)**: < 300ms per request

### Performance Monitoring
Monitor these metrics during testing:
- Response time
- Memory usage
- CPU utilization
- Cache hit rate
- Database query count

## Sample Test Scenarios

### 1. Basic Functionality Test
```bash
# Test all endpoints in sequence
curl http://localhost:8000/api/v1/recommendations/health/
curl http://localhost:8000/api/v1/recommendations/property/500/
curl http://localhost:8000/api/v1/recommendations/bulk/?property_ids=500,740
```

### 2. Parameter Validation Test
```bash
# Test with different parameters
curl "http://localhost:8000/api/v1/recommendations/property/500/?max_results=5"
curl "http://localhost:8000/api/v1/recommendations/property/500/?min_score=0.8"
curl "http://localhost:8000/api/v1/recommendations/property/500/?max_results=50&min_score=0.1"
```

### 3. Error Handling Test
```bash
# Test with invalid property ID
curl http://localhost:8000/api/v1/recommendations/property/99999/

# Test with invalid parameters
curl "http://localhost:8000/api/v1/recommendations/property/500/?max_results=-1"
```

This comprehensive guide should help you thoroughly test the recommendation system API and verify all optimizations are working correctly.
