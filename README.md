# Yorgan

### Installation (Dev Mode)

To install the development environment (including the package in editable mode):

```bash
pixi install -e dev
```

### Run fastapi server.

You can run the server using the configured pixi task:

```bash
pixi run server
```

Or manually:

```bash
uvicorn app.main:app --port 8000
```
or just:
```bash
fastapi dev
```

### Run agent
currently due to bug in google adk we cant mount the agents directly. So we run agents server separately

```
cd app/agents
adk web --port 8001 --allow_origins="*"
```
