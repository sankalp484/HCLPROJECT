import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import load_all
from src.validate import validate_all, print_report

if __name__ == "__main__":
    data = load_all()
    results = validate_all(**data)
    print_report(results)