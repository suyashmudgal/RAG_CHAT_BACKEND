# DocChat AI Backend

The complete engineering documentation and architectural guide for DocChat AI is maintained in the root [README.md](../README.md).

For quick startup instructions:
```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Run migrations
alembic upgrade head

# 5. Start dev server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
