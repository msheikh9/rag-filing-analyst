.DEFAULT_GOAL := help

.PHONY: dev down logs index test-backend lint-backend lint-frontend build clean help

dev: ## Start all services (dev mode)
	docker compose --profile full up -d

down: ## Stop all services
	docker compose --profile full down

logs: ## Follow all container logs
	docker compose --profile full logs -f

index: ## Run the indexing script inside the API container
	docker compose exec api python -m scripts.index_sec_dataset

test-backend: ## Run pytest inside the API container
	docker compose exec api python -m pytest tests/ -v

lint-backend: ## Run black --check and isort --check inside the API container
	docker compose exec api black --check .
	docker compose exec api isort --check .

lint-frontend: ## Run npm run lint inside the frontend container
	docker compose exec frontend npm run lint

build: ## Build production images
	docker compose -f docker-compose.prod.yml build

clean: ## Stop containers and remove volumes
	docker compose --profile full down -v

help: ## List all targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
