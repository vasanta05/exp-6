
import unittest

from app import app
from train_model import train_model


class TestPlacementAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        train_model()

    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")

    def test_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "cgpa": 8.0,
                "aptitude_score": 80,
                "communication_score": 75,
                "internships": 2
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.get_json()["placement_prediction"],
            ["PLACED", "NOT PLACED"]
        )

    def test_missing_fields(self):
        response = self.client.post("/predict", json={"cgpa": 8})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
