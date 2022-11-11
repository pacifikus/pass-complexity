from locust import HttpUser, between, task
from test_utils import generate_query


class LoadTestUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predict(self):
        query = generate_query()
        self.client.get(f"/predict?password={query}")
