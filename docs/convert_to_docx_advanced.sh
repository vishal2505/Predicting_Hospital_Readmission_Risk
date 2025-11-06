#!/bin/bash
# Advanced conversion with custom formatting and styles

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Converting TECHNICAL_REPORT.md to DOCX with custom formatting..."
echo "Working directory: $(pwd)"

pandoc TECHNICAL_REPORT.md \
  -o TECHNICAL_REPORT.docx \
  --from markdown \
  --to docx \
  --toc \
  --toc-depth=2 \
  --number-sections \
  --highlight-style=tango \
  --metadata title="Predicting Hospital Readmission Risk for Diabetic Patients" \
  --metadata author="Vishal Mishra (01520511)" \
  --metadata date="November 5, 2025"

echo "✅ Advanced conversion complete: TECHNICAL_REPORT.docx"
echo "Output location: $SCRIPT_DIR/TECHNICAL_REPORT.docx"
