.PHONY: dev backend frontend install install-backend install-frontend clean

# ── Start everything ──────────────────────────────────────────────────────────
dev:
	@echo "Starting CalibrAI (backend :8000, frontend :3000)..."
	@trap 'kill %1 %2 2>/dev/null; exit 0' INT TERM; \
	python3 -m uvicorn backend.main:app --reload --port 8000 & \
	(cd frontend && npm run dev) & \
	wait

# ── Individual processes ──────────────────────────────────────────────────────
backend:
	cd /Users/dheeptharai/calibrai && python3 -m uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

# ── Install dependencies ──────────────────────────────────────────────────────
install: install-backend install-frontend

install-backend:
	pip install -r backend/requirements.txt

install-frontend:
	cd frontend && npm install

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f calibrai.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
