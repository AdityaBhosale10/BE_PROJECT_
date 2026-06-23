import json
from src import main
app = main.create_app()
out = []
for r in app.routes:
    out.append({
        'type': type(r).__name__,
        'has_path': hasattr(r, 'path'),
        'has_routes': hasattr(r, 'routes'),
        'repr': repr(r)[:400]
    })
print(json.dumps(out, indent=2))
