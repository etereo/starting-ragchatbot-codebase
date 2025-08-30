#!/bin/bash
set -e

echo "🔎 Running type checking with mypy..."

uv run mypy backend/ --ignore-missing-imports

echo "✅ Type checking complete!"