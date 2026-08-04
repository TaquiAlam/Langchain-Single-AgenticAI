import sys
import os

# Add Backend folder to system path
backend_path = os.path.join(os.path.dirname(__file__), "Backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Execute Backend/app.py
app_py_path = os.path.join(backend_path, "app.py")
with open(app_py_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, app_py_path, "exec"))
