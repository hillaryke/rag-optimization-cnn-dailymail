.PHONY: docker-up docker-down test

up:
	docker compose up -d

down:
	docker compose down

test:
	poetry run python -m unittest discover -s tests -p "test_*.py"

add-dev_%:
	poetry add --group dev $*

run:
	poetry run python -m main