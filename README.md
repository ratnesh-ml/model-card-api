# Model Card API

A small FastAPI service that shows what I think an ML endpoint should expose besides a prediction. It includes health status, model metadata, input validation, predictions, and a simple feature-contribution response.

The training data is synthetic and the example task is intentionally harmless. The value of the project is the service contract and documentation, not the accuracy of a made-up student outcome model.

## Endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness and model version |
| GET | `/model-card` | Intended use, limitations, and features |
| POST | `/predict` | Validated prediction response |
| POST | `/explain` | Linear feature contribution details |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
python -m model_card_api
```

Then open `http://127.0.0.1:8000/docs` or try:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-card
curl -X POST http://127.0.0.1:8000/predict       -H 'content-type: application/json'       -d '{"study_hours":5,"attendance_rate":0.8,"practice_sessions":4}'
```

## Run with Docker

```bash
docker build -t model-card-api .
docker run --rm -p 8000:8000 model-card-api
```

## Why this is portfolio-worthy

Many beginner ML projects stop at `model.predict()`. This one makes the boundary visible: the API has a health route, a model card, a version, typed inputs, tests, a Dockerfile, and an explicit warning against real academic decisions.

## Limitations

The model is trained on a tiny synthetic dataset and is not suitable for real decisions. The explanation is a linear model contribution, not a causal explanation. A production iteration would add monitoring, authentication, a real validated dataset, calibration checks, and a more careful fairness review.

## License

MIT. See [LICENSE](LICENSE).
