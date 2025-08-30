#!/bin/bash
set -e

echo "🔍 Running linting checks..."

echo "Running flake8..."
uv run flake8 backend/

echo "Running Black check..."
uv run black --check backend/

echo "Running isort check..."
uv run isort --check-only backend/

echo "✅ All linting checks passed!"