from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging, json, time, joblib

# opentelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# ===== OTel setup =====
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# ===== logging setup =====
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title="Iris classifier with telemetry")

# service state
app_state = {"is_ready": False, "is_alive": True}

# pydantic schema
class Input(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

model = None

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info("Loading model from models/model.pkl ...")
        model = joblib.load("models/model.pkl")
        app_state["is_ready"] = True
        logger.info("✅ Model loaded.")
    except Exception as e:
        logger.exception(f"❌ Failed to load model: {e}")
        app_state["is_ready"] = False

@app.get("/live_check")
async def live_check():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check")
async def ready_check():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    resp = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    resp.headers["X-Process-Time-ms"] = str(duration)
    return resp

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict")
async def predict(input: Input, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            if model is None:
                raise RuntimeError("Model not loaded")

            features = [[
                input.sepal_length,
                input.sepal_width,
                input.petal_length,
                input.petal_width,
            ]]

            raw_pred = model.predict(features)[0]

            # normalize output
            if isinstance(raw_pred, str):
                species = raw_pred
            else:
                species = SPECIES_MAP.get(int(raw_pred), str(raw_pred))

            latency = round((time.time() - start_time) * 1000, 2)

            payload = {
                "status": "success",
                "predicted_label": species,
                "species": species,
                "latency_ms": latency,
                "trace_id": trace_id,
            }

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": features,
                "output": payload,
                "latency_ms": latency,
                "status": "success"
            }))

            return payload

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")

