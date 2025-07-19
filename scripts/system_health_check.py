#!/usr/bin/env python
"""
System Health Check and Monitoring Script
Validates all components and provides detailed system status
"""
import os
import sys
import time
import json
import requests
import psutil
from pathlib import Path

class SystemHealthChecker:
    """
    Comprehensive system health monitoring and validation
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.health_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "UNKNOWN",
            "components": {},
            "performance_metrics": {},
            "recommendations": []
        }
    
    def log(self, message, level="INFO"):
        """Log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def check_system_resources(self):
        """Check system resource utilization"""
        self.log("🖥️ Checking System Resources...")
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network stats
            network = psutil.net_io_counters()
            
            resources = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory_percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk_percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            }
            
            # Determine status
            status = "HEALTHY"
            if cpu_percent > 80 or memory_percent > 85 or disk_percent > 90:
                status = "WARNING"
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                status = "CRITICAL"
            
            self.health_report["components"]["system_resources"] = {
                "status": status,
                "metrics": resources
            }
            
            self.log(f"✅ System Resources - {status}")
            self.log(f"   CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
            
            return status != "CRITICAL"
            
        except Exception as e:
            self.log(f"❌ System resources check failed: {e}", "ERROR")
            self.health_report["components"]["system_resources"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
    
    def check_database_connection(self):
        """Check database connectivity and performance"""
        self.log("🗄️ Checking Database Connection...")
        
        try:
            # Set up Django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
            import django
            django.setup()
            
            from django.db import connection
            from django.core.management import execute_from_command_line
            
            # Test database connection
            start_time = time.time()
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            connection_time = (time.time() - start_time) * 1000
            
            # Check table counts
            from apps.properties.models import Property
            from apps.contacts.models import Contact
            
            property_count = Property.objects.count()
            contact_count = Contact.objects.count()
            
            db_metrics = {
                "connection_time_ms": round(connection_time, 2),
                "property_count": property_count,
                "contact_count": contact_count,
                "database_engine": connection.vendor
            }
            
            status = "HEALTHY"
            if connection_time > 100:
                status = "WARNING"
            if connection_time > 500:
                status = "CRITICAL"
            
            self.health_report["components"]["database"] = {
                "status": status,
                "metrics": db_metrics
            }
            
            self.log(f"✅ Database - {status}")
            self.log(f"   Connection: {connection_time:.2f}ms, Properties: {property_count}, Contacts: {contact_count}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Database check failed: {e}", "ERROR")
            self.health_report["components"]["database"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
    
    def check_redis_cache(self):
        """Check Redis cache connectivity and performance"""
        self.log("🔄 Checking Redis Cache...")
        
        try:
            import redis
            
            # Connect to Redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            
            # Test basic operations
            start_time = time.time()
            r.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Test set/get operations
            start_time = time.time()
            r.set("health_check", "test_value", ex=60)
            value = r.get("health_check")
            operation_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = r.info()
            
            cache_metrics = {
                "ping_time_ms": round(ping_time, 2),
                "operation_time_ms": round(operation_time, 2),
                "connected_clients": info.get('connected_clients', 0),
                "used_memory_mb": round(info.get('used_memory', 0) / (1024*1024), 2),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0)
            }
            
            # Calculate hit rate
            hits = cache_metrics["keyspace_hits"]
            misses = cache_metrics["keyspace_misses"]
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
            cache_metrics["hit_rate_percent"] = round(hit_rate * 100, 2)
            
            status = "HEALTHY"
            if ping_time > 10 or operation_time > 50:
                status = "WARNING"
            if ping_time > 50 or operation_time > 200:
                status = "CRITICAL"
            
            self.health_report["components"]["redis_cache"] = {
                "status": status,
                "metrics": cache_metrics
            }
            
            self.log(f"✅ Redis Cache - {status}")
            self.log(f"   Ping: {ping_time:.2f}ms, Hit Rate: {cache_metrics['hit_rate_percent']}%")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Redis check failed: {e}", "ERROR")
            self.health_report["components"]["redis_cache"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
    
    def check_recommendation_engine(self):
        """Check recommendation engine components"""
        self.log("🤖 Checking Recommendation Engine...")
        
        try:
            # Set up Django if not already done
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
            import django
            django.setup()
            
            from apps.recommendations.services import RecommendationEngine, HyperPersonalizedRecommendationEngine
            from apps.recommendations.filters import HybridRecommendationFilter
            from apps.recommendations.colbert_interaction import RealEstateColBERTMatcher
            from apps.recommendations.gemini_explainer import GeminiExplainerService
            
            # Test component initialization
            components = {}
            
            # Test RecommendationEngine
            start_time = time.time()
            rec_engine = RecommendationEngine()
            components["recommendation_engine"] = {
                "status": "HEALTHY",
                "init_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Test HyperPersonalizedRecommendationEngine
            start_time = time.time()
            hyper_engine = HyperPersonalizedRecommendationEngine()
            components["hyper_personalized_engine"] = {
                "status": "HEALTHY",
                "init_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Test HybridRecommendationFilter
            start_time = time.time()
            filter_engine = HybridRecommendationFilter()
            components["filtering_system"] = {
                "status": "HEALTHY",
                "init_time_ms": round((time.time() - start_time) * 1000, 2),
                "default_weights": len(filter_engine.default_weights)
            }
            
            # Test ColBERT matcher
            start_time = time.time()
            colbert_matcher = RealEstateColBERTMatcher()
            components["colbert_interaction"] = {
                "status": "HEALTHY" if colbert_matcher.colbert_engine else "WARNING",
                "init_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Test Gemini explainer
            start_time = time.time()
            gemini_explainer = GeminiExplainerService()
            components["gemini_explainer"] = {
                "status": "HEALTHY" if gemini_explainer.model else "WARNING",
                "init_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            self.health_report["components"]["recommendation_components"] = components
            
            self.log("✅ Recommendation Engine Components")
            for name, info in components.items():
                self.log(f"   {name}: {info['status']} ({info['init_time_ms']}ms)")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Recommendation engine check failed: {e}", "ERROR")
            self.health_report["components"]["recommendation_components"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
    
    def check_api_endpoints(self):
        """Check API endpoint availability and performance"""
        self.log("🌐 Checking API Endpoints...")
        
        base_url = "http://localhost:8000"
        
        endpoints = [
            ("/api/v1/recommendations/", "Recommendations API"),
            ("/api/v1/recommendations/filter-suggestions/", "Filter Suggestions"),
            ("/api/v1/recommendations/explain/", "Explanation API"),
            ("/api/v1/recommendations/colbert-interaction/", "ColBERT Analysis"),
            ("/api/schema/", "API Schema"),
            ("/admin/", "Admin Interface")
        ]
        
        endpoint_results = {}
        
        for endpoint, description in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                status = "HEALTHY"
                if response.status_code in [200, 401, 403]:  # Auth required is OK
                    if response_time > 1000:
                        status = "WARNING"
                    elif response_time > 5000:
                        status = "CRITICAL"
                else:
                    status = "ERROR"
                
                endpoint_results[endpoint] = {
                    "status": status,
                    "response_time_ms": round(response_time, 2),
                    "status_code": response.status_code,
                    "description": description
                }
                
                self.log(f"   {description}: {status} ({response.status_code}, {response_time:.2f}ms)")
                
            except Exception as e:
                endpoint_results[endpoint] = {
                    "status": "ERROR",
                    "error": str(e),
                    "description": description
                }
                self.log(f"   {description}: ERROR - {str(e)}")
        
        self.health_report["components"]["api_endpoints"] = endpoint_results
        
        healthy_endpoints = sum(1 for r in endpoint_results.values() if r.get("status") == "HEALTHY")
        total_endpoints = len(endpoint_results)
        
        self.log(f"✅ API Endpoints: {healthy_endpoints}/{total_endpoints} healthy")
        
        return healthy_endpoints > total_endpoints * 0.8  # 80% healthy threshold
    
    def performance_benchmark(self):
        """Run performance benchmarks"""
        self.log("⚡ Running Performance Benchmarks...")
        
        try:
            # Set up Django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
            import django
            django.setup()
            
            from apps.recommendations.services import RecommendationEngine
            from apps.recommendations.filters import HybridRecommendationFilter
            
            # Benchmark recommendation engine
            rec_engine = RecommendationEngine()
            filter_engine = HybridRecommendationFilter()
            
            # Test data
            sample_recommendations = [
                {
                    'property': {
                        'id': i,
                        'title': f'Test Property {i}',
                        'price': 400000 + (i * 10000),
                        'city': 'Downtown' if i % 2 == 0 else 'Suburbs',
                        'area': 80 + (i * 5),
                        'rooms': 2 + (i % 3),
                        'has_parking': i % 2 == 0
                    },
                    'score': 0.8 + (i * 0.01)
                }
                for i in range(100)
            ]
            
            # Benchmark filtering
            from apps.recommendations.filters import FilterCriteria, FilterOperator
            
            filter_criteria = [
                FilterCriteria('property.price', FilterOperator.BETWEEN, [350000, 550000], 1.0),
                FilterCriteria('property.city', FilterOperator.IN, ['Downtown', 'Central'], 0.9)
            ]
            
            start_time = time.time()
            filtered_results = filter_engine.apply_filters(
                recommendations=sample_recommendations,
                filter_criteria=filter_criteria
            )
            filtering_time = (time.time() - start_time) * 1000
            
            # Benchmark distance calculation
            start_time = time.time()
            for _ in range(1000):
                distance = filter_engine._calculate_haversine_distance(
                    40.7128, -74.0060,  # NYC
                    40.7589, -73.9851   # Central Park
                )
            distance_calc_time = (time.time() - start_time) * 1000 / 1000  # Per calculation
            
            benchmarks = {
                "filtering_100_items_ms": round(filtering_time, 2),
                "distance_calculation_ms": round(distance_calc_time, 4),
                "filtered_results_count": len(filtered_results.filtered_recommendations),
                "filtering_throughput_per_sec": round(100 / (filtering_time / 1000), 0)
            }
            
            self.health_report["performance_metrics"]["benchmarks"] = benchmarks
            
            self.log("✅ Performance Benchmarks")
            self.log(f"   Filtering 100 items: {filtering_time:.2f}ms")
            self.log(f"   Distance calculation: {distance_calc_time:.4f}ms")
            self.log(f"   Filtering throughput: {benchmarks['filtering_throughput_per_sec']}/sec")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Performance benchmark failed: {e}", "ERROR")
            self.health_report["performance_metrics"]["benchmarks"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
    
    def generate_recommendations(self):
        """Generate system recommendations based on health check"""
        self.log("💡 Generating System Recommendations...")
        
        recommendations = []
        
        # Check system resources
        if "system_resources" in self.health_report["components"]:
            resources = self.health_report["components"]["system_resources"]
            if resources["status"] == "WARNING":
                recommendations.append("⚠️ High system resource usage detected. Consider scaling or optimization.")
            elif resources["status"] == "CRITICAL":
                recommendations.append("🚨 Critical system resource usage. Immediate action required.")
        
        # Check database performance
        if "database" in self.health_report["components"]:
            db = self.health_report["components"]["database"]
            if db.get("metrics", {}).get("connection_time_ms", 0) > 100:
                recommendations.append("🗄️ Database connection time is high. Check database performance and indexing.")
        
        # Check cache performance
        if "redis_cache" in self.health_report["components"]:
            cache = self.health_report["components"]["redis_cache"]
            if cache.get("metrics", {}).get("hit_rate_percent", 100) < 80:
                recommendations.append("🔄 Cache hit rate is low. Consider cache warming or TTL optimization.")
        
        # Check API performance
        if "api_endpoints" in self.health_report["components"]:
            endpoints = self.health_report["components"]["api_endpoints"]
            slow_endpoints = [
                ep for ep, info in endpoints.items() 
                if info.get("response_time_ms", 0) > 1000
            ]
            if slow_endpoints:
                recommendations.append(f"🌐 Slow API endpoints detected: {', '.join(slow_endpoints)}")
        
        # General recommendations
        recommendations.extend([
            "📊 Monitor system metrics regularly using this health check",
            "🔄 Set up automated cache warming for better performance",
            "📈 Consider implementing request rate limiting for production",
            "🔒 Ensure proper authentication and authorization in production",
            "📝 Set up comprehensive logging and monitoring"
        ])
        
        self.health_report["recommendations"] = recommendations
        
        for rec in recommendations:
            self.log(f"   {rec}")
        
        return True
    
    def run_full_health_check(self):
        """Run comprehensive system health check"""
        self.log("🏥 Starting Comprehensive System Health Check")
        self.log("="*60)
        
        checks = [
            ("System Resources", self.check_system_resources),
            ("Database Connection", self.check_database_connection),
            ("Redis Cache", self.check_redis_cache),
            ("Recommendation Engine", self.check_recommendation_engine),
            ("API Endpoints", self.check_api_endpoints),
            ("Performance Benchmarks", self.performance_benchmark),
            ("System Recommendations", self.generate_recommendations)
        ]
        
        passed_checks = 0
        
        for check_name, check_function in checks:
            self.log(f"\n🔍 {check_name}")
            self.log("-" * 40)
            
            try:
                if check_function():
                    passed_checks += 1
                    self.log(f"✅ {check_name} - PASSED")
                else:
                    self.log(f"❌ {check_name} - FAILED", "ERROR")
            except Exception as e:
                self.log(f"❌ {check_name} - EXCEPTION: {str(e)}", "ERROR")
            
            self.log("-" * 40)
        
        # Determine overall status
        success_rate = passed_checks / len(checks)
        if success_rate >= 0.9:
            overall_status = "HEALTHY"
        elif success_rate >= 0.7:
            overall_status = "WARNING"
        else:
            overall_status = "CRITICAL"
        
        self.health_report["overall_status"] = overall_status
        self.health_report["success_rate"] = round(success_rate * 100, 1)
        
        # Save health report
        report_path = self.project_root / "system_health_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.health_report, f, indent=2)
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("🏥 SYSTEM HEALTH CHECK COMPLETED")
        self.log("="*60)
        
        self.log(f"📊 Overall Status: {overall_status}")
        self.log(f"✅ Passed Checks: {passed_checks}/{len(checks)} ({success_rate*100:.1f}%)")
        self.log(f"📋 Health Report: {report_path}")
        
        if overall_status == "HEALTHY":
            self.log("\n🎉 System is healthy and ready for production!")
        elif overall_status == "WARNING":
            self.log("\n⚠️ System has some issues but is operational")
        else:
            self.log("\n🚨 System has critical issues requiring immediate attention")
        
        return overall_status == "HEALTHY"


if __name__ == "__main__":
    checker = SystemHealthChecker()
    is_healthy = checker.run_full_health_check()
    
    if is_healthy:
        print("\n🎊 System health check passed!")
        sys.exit(0)
    else:
        print("\n⚠️ System health check found issues")
        sys.exit(1)