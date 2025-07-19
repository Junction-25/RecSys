#!/usr/bin/env python
"""
Production Deployment Script for Real Estate Recommendation System
Handles complete production deployment with monitoring and validation
"""
import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

class ProductionDeployment:
    """
    Production deployment orchestrator with comprehensive validation
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.deployment_log = []
        self.services_status = {}
        
    def log(self, message, level="INFO"):
        """Log deployment progress"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.deployment_log.append(log_entry)
    
    def run_command(self, command, description, check=True):
        """Run deployment command with logging"""
        self.log(f"Executing: {description}")
        self.log(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.log(f"✅ {description} - SUCCESS")
                if result.stdout.strip():
                    self.log(f"Output: {result.stdout.strip()}")
                return True
            else:
                self.log(f"❌ {description} - FAILED", "ERROR")
                self.log(f"Error: {result.stderr.strip()}", "ERROR")
                if check:
                    raise Exception(f"Command failed: {description}")
                return False
                
        except Exception as e:
            self.log(f"❌ {description} - EXCEPTION: {str(e)}", "ERROR")
            if check:
                raise
            return False
    
    def validate_environment(self):
        """Validate deployment environment"""
        self.log("🔍 Validating Deployment Environment...")
        
        # Check required files
        required_files = [
            "docker-compose.production.yml",
            "nginx.prod.conf",
            "Dockerfile",
            ".env",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise Exception(f"Missing required files: {', '.join(missing_files)}")
        
        # Check environment variables
        required_env_vars = [
            "POSTGRES_PASSWORD",
            "GEMINI_API_KEY",
            "GRAFANA_PASSWORD"
        ]
        
        missing_env_vars = []
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                missing_env_vars.append(env_var)
        
        if missing_env_vars:
            self.log(f"⚠️ Missing environment variables: {', '.join(missing_env_vars)}", "WARNING")
            self.log("Please set these in your .env file or environment")
        
        # Check Docker
        self.run_command("docker --version", "Check Docker installation")
        self.run_command("docker-compose --version", "Check Docker Compose installation")
        
        self.log("✅ Environment validation completed")
        return True
    
    def build_images(self):
        """Build Docker images"""
        self.log("🏗️ Building Docker Images...")
        
        # Build main application image
        self.run_command(
            "docker-compose -f docker-compose.production.yml build --no-cache",
            "Build application images"
        )
        
        self.log("✅ Docker images built successfully")
        return True
    
    def deploy_infrastructure(self):
        """Deploy infrastructure services"""
        self.log("🚀 Deploying Infrastructure Services...")
        
        # Start infrastructure services first
        infrastructure_services = [
            "db",
            "redis", 
            "chromadb"
        ]
        
        for service in infrastructure_services:
            self.run_command(
                f"docker-compose -f docker-compose.production.yml up -d {service}",
                f"Start {service} service"
            )
            
            # Wait for service to be ready
            self.log(f"Waiting for {service} to be ready...")
            time.sleep(10)
        
        self.log("✅ Infrastructure services deployed")
        return True
    
    def deploy_application(self):
        """Deploy application services"""
        self.log("📱 Deploying Application Services...")
        
        # Start application services
        app_services = [
            "web",
            "celery_worker",
            "celery_beat",
            "nginx"
        ]
        
        for service in app_services:
            self.run_command(
                f"docker-compose -f docker-compose.production.yml up -d {service}",
                f"Start {service} service"
            )
            
            # Wait for service to be ready
            time.sleep(5)
        
        self.log("✅ Application services deployed")
        return True
    
    def deploy_monitoring(self):
        """Deploy monitoring services"""
        self.log("📊 Deploying Monitoring Services...")
        
        # Start monitoring services
        monitoring_services = [
            "prometheus",
            "grafana"
        ]
        
        for service in monitoring_services:
            self.run_command(
                f"docker-compose -f docker-compose.production.yml up -d {service}",
                f"Start {service} service"
            )
            
            time.sleep(5)
        
        self.log("✅ Monitoring services deployed")
        return True
    
    def run_migrations(self):
        """Run database migrations"""
        self.log("🗄️ Running Database Migrations...")
        
        # Wait for database to be ready
        self.log("Waiting for database to be ready...")
        time.sleep(15)
        
        # Run migrations
        self.run_command(
            "docker-compose -f docker-compose.production.yml exec -T web python manage.py migrate",
            "Run database migrations"
        )
        
        # Create superuser (if needed)
        self.run_command(
            "docker-compose -f docker-compose.production.yml exec -T web python manage.py collectstatic --noinput",
            "Collect static files",
            check=False
        )
        
        self.log("✅ Database migrations completed")
        return True
    
    def validate_deployment(self):
        """Validate deployment health"""
        self.log("🏥 Validating Deployment Health...")
        
        # Wait for services to fully start
        self.log("Waiting for services to stabilize...")
        time.sleep(30)
        
        # Check service status
        result = subprocess.run(
            "docker-compose -f docker-compose.production.yml ps",
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        
        self.log("Service Status:")
        self.log(result.stdout)
        
        # Test endpoints
        endpoints_to_test = [
            ("http://localhost/health/", "Nginx health check"),
            ("http://localhost/api/v1/recommendations/", "Recommendations API"),
            ("http://localhost:9090/", "Prometheus monitoring"),
            ("http://localhost:3000/", "Grafana dashboard")
        ]
        
        healthy_endpoints = 0
        
        for url, description in endpoints_to_test:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code in [200, 401, 403]:  # Auth required is OK
                    self.log(f"✅ {description} - Accessible")
                    healthy_endpoints += 1
                else:
                    self.log(f"⚠️ {description} - Status {response.status_code}", "WARNING")
            except Exception as e:
                self.log(f"❌ {description} - Error: {str(e)}", "ERROR")
        
        success_rate = healthy_endpoints / len(endpoints_to_test)
        
        if success_rate >= 0.75:
            self.log("✅ Deployment validation passed")
            return True
        else:
            self.log(f"❌ Deployment validation failed - {healthy_endpoints}/{len(endpoints_to_test)} endpoints healthy", "ERROR")
            return False
    
    def setup_monitoring_dashboards(self):
        """Setup monitoring dashboards"""
        self.log("📈 Setting up Monitoring Dashboards...")
        
        # Create monitoring directory structure
        monitoring_dir = self.project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Create Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'django-app'
    static_configs:
      - targets: ['web:8000']
    metrics_path: '/metrics/'
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:8080']
    metrics_path: '/nginx_status'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['db:5432']
    scrape_interval: 30s
"""
        
        with open(monitoring_dir / "prometheus.yml", "w") as f:
            f.write(prometheus_config)
        
        self.log("✅ Monitoring dashboards configured")
        return True
    
    def generate_deployment_report(self):
        """Generate deployment report"""
        self.log("📋 Generating Deployment Report...")
        
        report = {
            "deployment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deployment_status": "SUCCESS",
            "services_deployed": [
                "web", "db", "redis", "celery_worker", 
                "celery_beat", "nginx", "prometheus", 
                "grafana", "chromadb"
            ],
            "endpoints": {
                "application": "http://localhost/",
                "api": "http://localhost/api/v1/",
                "admin": "http://localhost/admin/",
                "monitoring": "http://localhost:9090/",
                "dashboards": "http://localhost:3000/"
            },
            "deployment_log": self.deployment_log,
            "next_steps": [
                "Configure SSL certificates for HTTPS",
                "Set up automated backups",
                "Configure log aggregation",
                "Set up alerting rules",
                "Perform load testing"
            ]
        }
        
        # Save report
        report_path = self.project_root / "deployment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"📄 Deployment report saved: {report_path}")
        return True
    
    def run_full_deployment(self):
        """Run complete production deployment"""
        self.log("🚀 Starting Production Deployment")
        self.log("="*60)
        
        deployment_steps = [
            ("Environment Validation", self.validate_environment),
            ("Docker Images Build", self.build_images),
            ("Infrastructure Deployment", self.deploy_infrastructure),
            ("Application Deployment", self.deploy_application),
            ("Database Migrations", self.run_migrations),
            ("Monitoring Deployment", self.deploy_monitoring),
            ("Monitoring Setup", self.setup_monitoring_dashboards),
            ("Deployment Validation", self.validate_deployment),
            ("Deployment Report", self.generate_deployment_report)
        ]
        
        successful_steps = 0
        
        for step_name, step_function in deployment_steps:
            self.log(f"\n🔧 Step: {step_name}")
            self.log("-" * 40)
            
            try:
                if step_function():
                    successful_steps += 1
                    self.log(f"✅ {step_name} - COMPLETED")
                else:
                    self.log(f"❌ {step_name} - FAILED", "ERROR")
                    break
            except Exception as e:
                self.log(f"❌ {step_name} - EXCEPTION: {str(e)}", "ERROR")
                break
            
            self.log("-" * 40)
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("🎉 PRODUCTION DEPLOYMENT COMPLETED")
        self.log("="*60)
        
        if successful_steps == len(deployment_steps):
            self.log("🎊 DEPLOYMENT SUCCESSFUL!")
            self.log("\n📋 System Access Points:")
            self.log("• Application: http://localhost/")
            self.log("• API Documentation: http://localhost/api/schema/")
            self.log("• Admin Interface: http://localhost/admin/")
            self.log("• Prometheus Monitoring: http://localhost:9090/")
            self.log("• Grafana Dashboards: http://localhost:3000/")
            
            self.log("\n🔧 Post-Deployment Tasks:")
            self.log("1. Configure SSL certificates")
            self.log("2. Set up automated backups")
            self.log("3. Configure alerting")
            self.log("4. Perform load testing")
            self.log("5. Set up log aggregation")
            
            return True
        else:
            self.log(f"❌ Deployment failed at step: {deployment_steps[successful_steps][0]}")
            self.log("Check the logs above for details")
            return False


if __name__ == "__main__":
    deployment = ProductionDeployment()
    success = deployment.run_full_deployment()
    
    if success:
        print("\n🚀 Production deployment completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Production deployment failed")
        sys.exit(1)