# Model Card API

[![CI](https://github.com/ratnesh-ml/model-card-api/actions/workflows/test.yml/badge.svg)](https://github.com/ratnesh-ml/model-card-api/actions/workflows/test.yml)

I built this small FastAPI service after noticing that many beginner ML projects end at `model.predict()`. I wanted to practise the less visible work around a prediction: declaring what the model is for, validating inputs, exposing version and health information, and making a simple explanation available to the caller.

The model and data are intentionally synthetic. This is not a student-outcome predictor, and I do not present it as one. The project is my compact exercise in designing an inspectable ML service contract.

## At a glance

| I wanted to practise | What I implemented |
| --- | --- |
| Treating a model as a service | A typed FastAPI surface with health, model-card, prediction, and explanation routes. |
| Making assumptions visible | A `/model-card` endpoint with intended use, feature notes, and limitations. |
| Avoiding opaque inputs | Validation for the three input fields before a prediction is made. |
| Explaining a simple baseline honestly | Linear feature contributions rather than a misleading claim of causal explanation. |
| Keeping the work reproducible | Tests, a Dockerfile, and a GitHub Actions check. |

## The API contract

| Method | Route | What I expect it to do |
| --- | --- | --- |
| `GET` | `/health` | Report liveness and the model version. |
| `GET` | `/model-card` | Show intended use, limitations, and the available features. |
| `POST` | `/predict` | Return a validated demo prediction and confidence band. |
| `POST` | `/explain` | Return linear feature-contribution details for that prediction. |

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
python -m model_card_api
```

Open `http://127.0.0.1:8000/docs`, or check the basic contract directly:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-card
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"study_hours":5,"attendance_rate":0.8,"practice_sessions":4}'
pytest -q
```

To use the container path instead:

```bash
docker build -t model-card-api .
docker run --rm -p 8000:8000 model-card-api
```

## Design decision I wanted to make visible

I kept the example deliberately narrow because a clean service boundary is more useful here than a flashy accuracy claim. Health, model metadata, input validation, prediction, and explanation are different responsibilities; putting them behind explicit routes made me think about how another developer would inspect the system before trusting it.

## What this is not

The classifier is trained on a tiny synthetic dataset. Its confidence band is illustrative, and the explanation is a linear contribution rather than causal evidence. It is not suitable for academic, hiring, or any real-person decision.

If I extended it, I would start with a permission-cleared dataset and then add calibration checks, monitoring, authentication, fairness evaluation, and a reviewed deployment policy.

## Verification and license

`pytest -q` runs the local regression suite. GitHub Actions runs the test workflow on pushes and pull requests. The project is MIT licensed; see [LICENSE](LICENSE).
