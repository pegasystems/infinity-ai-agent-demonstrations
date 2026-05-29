"""Run the DeepEval Pega REST API server."""

import uvicorn

from api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "run_api:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
    )
