web: gunicorn server:app
```

To:
```
web: gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --worker-class sync
```

### Why These Options?

- `--bind 0.0.0.0:$PORT` - Explicitly bind to Render's dynamic port
- `--workers 1` - Use single worker (important for free tier memory limits)
- `--timeout 300` - Give 5 minutes for model loading and predictions
- `--worker-class sync` - Use synchronous workers (best for TensorFlow)

---

## ✅ **Quick Summary - What You Need to Do:**

### 1. **Update 2 Files:**

**File 1: `server.py`** - Replace with the CORS-fixed code I provided earlier

**File 2: `Procfile`** - Update to:
```
web: gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --worker-class sync