all:
	docker compose up -d

down:
	docker compose down

exec:
	docker exec -it python bash

clean: down
	yes | docker system prune -a
