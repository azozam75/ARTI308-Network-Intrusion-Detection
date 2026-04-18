from fastapi import FastAPI

app = FastAPI(title="NIDS API")

@app.get("/")
def root():
    return {"status": "ok"}
