# 🚀 Quick Setup Guide

## Option 1: Docker (Recommended - 5 minutes)

### Prerequisites
- Docker Desktop installed and running
- 4GB RAM available
- 10GB disk space

### Steps
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd defect-detection-cv

# 2. Start all services
docker-compose up -d

# 3. Verify services are running
docker-compose ps

# 4. View logs (optional)
docker-compose logs -f backend

# 5. Access the application
open http://localhost:3000
```

### Services Started
- ✅ Frontend: http://localhost:3000
- ✅ Backend API: http://localhost:8000
- ✅ API Docs: http://localhost:8000/docs
- ✅ Prometheus: http://localhost:9090
- ✅ Grafana: http://localhost:3001 (admin/admin)

---

## Option 2: Local Development (15 minutes)

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start backend
python3 -m uvicorn app.api.main:app --reload --port 8000
```

### Frontend Setup (new terminal)
```bash
cd frontend
npm install
npm start
```

---

## Quick Test

### 1. Upload Test Image
```bash
# Download test image
curl -o test_defect.png https://via.placeholder.com/640

# Test detection
curl -X POST "http://localhost:8000/detect?visualize=true" \
  -F "file=@test_defect.png" \
  -o result.json

# View result
cat result.json
```

### 2. Check Metrics
```bash
# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

---

## Stopping Services

### Docker
```bash
docker-compose down
```

### Local
Press `Ctrl+C` in both terminal windows

---

## Troubleshooting

### Docker Issues
```bash
# Reset everything
docker-compose down -v
docker-compose up --build

# Check logs
docker-compose logs backend
docker-compose logs frontend
```

### Port Conflicts
If ports are in use, modify `docker-compose.yml`:
```yaml
ports:
  - "3001:80"  # Change 3000 to 3001
  - "8001:8000"  # Change 8000 to 8001
```

### Model Not Loading
```bash
# Verify model exists
ls -la models/production/model.onnx

# Check volume mount
docker-compose exec backend ls -la /app/models/production/
```

---

## Next Steps

1. ✅ Upload images at http://localhost:3000
2. ✅ View API docs at http://localhost:8000/docs
3. ✅ Monitor metrics at http://localhost:9090
4. ✅ Create dashboards at http://localhost:3001

---

**Need help? Open an issue on GitHub!**
