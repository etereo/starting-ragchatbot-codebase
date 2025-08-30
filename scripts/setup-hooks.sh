#!/bin/bash
set -e

echo "📋 Setting up pre-commit hooks..."

# Install pre-commit if not already installed
uv sync --group dev

# Install the git hook scripts
uv run pre-commit install

echo "✅ Pre-commit hooks installed! Code will be automatically checked before commits."