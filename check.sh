#!/usr/bin/env bash
set -e
echo "=== lint ===" && ruff check src/ tests/
echo "=== import smoke ===" && python -c "import agent; print('import OK')"
echo "=== tests ===" && pytest -x -q
echo "=== ALL CHECKS PASSED ==="
