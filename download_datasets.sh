#!/bin/bash
# =============================================================
# download_datasets.sh — Download all external datasets
# =============================================================
# Usage:  chmod +x download_datasets.sh && ./download_datasets.sh
# Prerequisites: pip install huggingface_hub pandas
# =============================================================

set -e
echo "============================================================"
echo "I'VE SEEN THIS BEFORE — Dataset Downloader"
echo "============================================================"

mkdir -p corpus/soviet_sources
mkdir -p corpus/modern_sources/isca_zenodo
mkdir -p corpus/modern_sources/isca_huggingface
mkdir -p corpus/modern_sources/conan

# --- 1. ISCA Zenodo (6,941 tweets, IHRA-labeled, CC BY 4.0) ---
echo -e "\n--- [1/4] ISCA Zenodo Dataset ---"
if [ -f "corpus/modern_sources/isca_zenodo/data.csv" ]; then
    echo "  Already exists, skipping."
else
    echo "  Downloading..."
    curl -L -o "corpus/modern_sources/isca_zenodo/data.csv" \
        "https://zenodo.org/records/7932888/files/DatasetForMachineLearning.csv?download=1" 2>/dev/null \
        && echo "  ✓ Done" \
        || echo "  ⚠ Failed. Download manually from https://zenodo.org/records/7932888"
fi

# --- 2. ISCA HuggingFace (expanded dataset ~11K tweets) ---
echo -e "\n--- [2/4] ISCA HuggingFace Dataset ---"
if [ -f "corpus/modern_sources/isca_huggingface/done.flag" ]; then
    echo "  Already exists, skipping."
else
    python3 -c "
from huggingface_hub import hf_hub_download
for f in ['DatasetForMachineLearning.csv','Antisemitism_dataset.csv']:
    try:
        hf_hub_download('ISCA-IUB/HateSpeechAndBias', f, repo_type='dataset', local_dir='corpus/modern_sources/isca_huggingface')
        print(f'  Downloaded: {f}')
    except Exception as e:
        print(f'  Skipped {f}: {e}')
open('corpus/modern_sources/isca_huggingface/done.flag','w').write('ok')
" 2>/dev/null && echo "  ✓ Done" \
    || echo "  ⚠ Install huggingface_hub: pip install huggingface_hub"
fi

# --- 3. CONAN Counter-Narratives (5,003 HS/CN pairs, CC BY 4.0) ---
echo -e "\n--- [3/4] CONAN Counter-Narratives ---"
if [ -f "corpus/modern_sources/conan/Multitarget-CONAN.csv" ]; then
    echo "  Already exists, skipping."
else
    echo "  Downloading from GitHub..."
    curl -L -o "corpus/modern_sources/conan/Multitarget-CONAN.csv" \
        "https://raw.githubusercontent.com/marcoguerini/CONAN/master/Multitarget-CONAN/Multitarget-CONAN.csv" 2>/dev/null
    curl -L -o "corpus/modern_sources/conan/DIALOCONAN.csv" \
        "https://raw.githubusercontent.com/marcoguerini/CONAN/master/DIALOCONAN/DIALOCONAN.csv" 2>/dev/null
    echo "  ✓ Done"
fi

# --- 4. Ivanov full text ---
echo -e "\n--- [4/4] Ivanov: Caution Zionism (full PDF) ---"
if [ -f "corpus/soviet_sources/ivanov_caution_zionism_1970.pdf" ]; then
    echo "  Already exists, skipping."
else
    curl -L -o "corpus/soviet_sources/ivanov_caution_zionism_1970.pdf" \
        "https://www.marxists.org/subject/jewish/caution-zionism.pdf" 2>/dev/null
    echo "  ✓ Downloaded PDF"
    if command -v pdftotext &>/dev/null; then
        pdftotext "corpus/soviet_sources/ivanov_caution_zionism_1970.pdf" \
            "corpus/soviet_sources/ivanov_caution_zionism_1970_full.txt"
        echo "  ✓ Extracted text"
    else
        echo "  Install poppler for text extraction: sudo apt install poppler-utils"
    fi
fi

echo -e "\n============================================================"
echo "DONE. Files:"
find corpus -type f \( -name "*.csv" -o -name "*.txt" -o -name "*.pdf" \) | sort
echo -e "\nNext: python process_modern_sources.py"
