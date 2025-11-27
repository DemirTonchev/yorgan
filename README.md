# Yorgan

### Run fast api 

```bash
uvicorn app.main:app --port 8000
```
or just (seems a bit buggy):
```bash
fastapi dev
```

### Run agent

```
cd app/agents
adk web --port 8001 --allow_origins="*"
```
