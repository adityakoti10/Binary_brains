"""Run every test script:   python tests/run_all.py"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
failed, skipped = [], []
for t in sorted(here.glob("test_*.py")):
    print(f"\n===== {t.name} =====")
    if t.name == "test_web.py":
        try:
            import fastapi, httpx  # noqa: F401
        except ImportError:
            print("SKIPPED (needs: pip install fastapi httpx python-multipart)")
            skipped.append(t.name)
            continue
    if subprocess.run([sys.executable, str(t)], cwd=here).returncode != 0:
        failed.append(t.name)
print("\n" + ("ALL RUN TEST FILES PASSED" if not failed else "FAILED: " + ", ".join(failed))
      + (f"  (skipped: {', '.join(skipped)})" if skipped else ""))
sys.exit(1 if failed else 0)
