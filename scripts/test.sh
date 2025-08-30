#!/bin/bash
set -e

echo "🧪 Running tests..."

uv run pytest backend/tests/ -v

echo "✅ All tests passed!"