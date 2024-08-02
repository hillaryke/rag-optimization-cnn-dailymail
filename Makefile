.PHONY: docker-up docker-down test

up:
	docker compose up -d

down:
	docker compose down

test:
	poetry run python -m unittest discover -s tests -p "test_*.py"

add-dev_%:
	poetry add --group dev $*

setup:
	poetry run python -m main

run-backend:
	poetry run uvicorn backend.app.main:app --reload