import sys
import os

# Remove the script's directory from sys.path to avoid shadowing 'reverie' package with 'reverie.py'
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
# Also remove CWD if it matches script_dir (just in case)
cwd = os.getcwd()
if cwd == script_dir and cwd in sys.path:
    sys.path.remove(cwd)

# Mimic what rag_interface does (corrected version)
# rag_interface.py is in reverie/backend_server/rag/
# It does: sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
# Relative to THIS script (in reverie/backend_server/), the equivalent is:
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

sys.path.append(target_path)
print(f"Path added: {target_path}")

try:
    import reverie.backend_server.rag.retriever
    print("Import success")
except ImportError as e:
    print(f"Import failed: {e}")
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")

