#!/bin/bash

# Stop any running Django server
echo "Stopping any running Django server..."
pkill -f "python manage.py runserver"

# Remove existing database and migrations
echo "Cleaning up old database and migrations..."
rm -f smart_realestate/db.sqlite3
find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
find . -name "__pycache__" -exec rm -r {} \;

# Create fresh database and apply migrations
echo "Creating new database and applying migrations..."
python manage.py makemigrations
python manage.py migrate

# Import data from JSON files
echo "Importing data from JSON files..."
python manage.py import_data --clear

# Create superuser (optional)
echo "Creating superuser..."
python manage.py createsuperuser --noinput --username admin --email admin@example.com

echo "Database setup complete!"
