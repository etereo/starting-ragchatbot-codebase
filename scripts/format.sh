#!/bin/bash
set -e

echo "🧹 Formatting code with Black and isort..."

echo "Running Black formatter..."
uv run black backend/

echo "Organizing imports with isort..."
uv run isort backend/

echo "✅ Code formatting complete!"