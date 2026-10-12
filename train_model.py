
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model():
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    X = []
    y = []

    for cgpa_x10 in range(40, 101):
        cgpa = cgpa_x10 / 10

        for aptitude in (30, 50, 70, 90):
            for communication in (30, 50, 70, 90):
                for internships in range(4):
                    score = (
                        0.4 * cgpa / 10
                        + 0.3 * aptitude / 100
                        + 0.2 * communication / 100
                        + 0.1 * internships / 3
                    )

                    X.append([
                        cgpa, aptitude, communication, internships
                    ])
                    y.append(int(score >= 0.65))

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    model.fit(X_train, y_train)

    joblib.dump(model, output_dir / "placement_model.joblib")
    print("Placement model trained and saved successfully.")


if __name__ == "__main__":
    train_model()
