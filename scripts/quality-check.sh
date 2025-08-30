#!/bin/bash
set -e

echo "🚀 Running complete code quality checks..."

# Run linting
./scripts/lint.sh

# Run type checking
./scripts/typecheck.sh

echo "✅ All quality checks passed! Your code is ready for production."