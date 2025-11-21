import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processFile = (file) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleDetect = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ error: 'Failed to detect defects' });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="App">
      {/* Animated Background */}
      <div className="background-animation">
        <div className="circle circle-1"></div>
        <div className="circle circle-2"></div>
        <div className="circle circle-3"></div>
      </div>

      {/* Header */}
      <header className="App-header">
        <div className="logo-container">
          <div className="logo-icon">👁️</div>
          <div className="logo-text">
            <h1>Defect Detection AI</h1>
            <p className="subtitle">Manufacturing Quality Control • Real-Time Analysis</p>
          </div>
        </div>
        
        <div className="tech-badges">
          <span className="badge">YOLOv8</span>
          <span className="badge">ONNX</span>
          <span className="badge">FastAPI</span>
          <span className="badge">React</span>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-container">
        
        {/* Upload Section */}
        <div className="content-card upload-card">
          <div className="card-header">
            <h2>📤 Upload Image</h2>
            <p>Drag & drop or click to select an image</p>
          </div>

          <div 
            className={`drop-zone ${isDragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => document.getElementById('file-input').click()}
          >
            {!preview ? (
              <div className="drop-zone-content">
                <div className="upload-icon">📁</div>
                <p className="drop-text">Drop your image here</p>
                <p className="drop-subtext">or click to browse</p>
                <p className="supported-formats">PNG, JPG, JPEG supported</p>
              </div>
            ) : (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="preview-image" />
                <button className="reset-button" onClick={(e) => {
                  e.stopPropagation();
                  handleReset();
                }}>
                  ✕ Clear
                </button>
              </div>
            )}
          </div>

          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />

          <button
            onClick={handleDetect}
            disabled={!selectedFile || loading}
            className={`detect-button ${loading ? 'loading' : ''}`}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <span className="button-icon">��</span>
                Detect Defects
              </>
            )}
          </button>
        </div>

        {/* Results Section */}
        {result && (
          <div className={`content-card results-card ${result.has_defect ? 'defect-result' : 'good-result'}`}>
            <div className="card-header">
              <h2>📊 Analysis Results</h2>
            </div>

            <div className="status-banner">
              <div className="status-icon">
                {result.has_defect ? '❌' : '✅'}
              </div>
              <div className="status-text">
                <h3>{result.has_defect ? 'Defect Detected' : 'Quality Passed'}</h3>
                <p>{result.has_defect ? 'Issue found in product' : 'No defects detected'}</p>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-icon">📈</div>
                <div className="metric-content">
                  <span className="metric-label">Confidence Score</span>
                  <span className="metric-value">{result.confidence?.toFixed(2)}</span>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">📋</div>
                <div className="metric-content">
                  <span className="metric-label">Status</span>
                  <span className="metric-value status-badge">
                    {result.status}
                  </span>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">📐</div>
                <div className="metric-content">
                  <span className="metric-label">Image Size</span>
                  <span className="metric-value">
                    {result.image_size?.width} × {result.image_size?.height}
                  </span>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">📄</div>
                <div className="metric-content">
                  <span className="metric-label">Filename</span>
                  <span className="metric-value filename">{result.filename}</span>
                </div>
              </div>
            </div>

            <div className="action-buttons">
              <button className="secondary-button" onClick={handleReset}>
                🔄 Analyze Another
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Stats Footer */}
      <footer className="stats-footer">
        <div className="stat-item">
          <span className="stat-icon">⚡</span>
          <div className="stat-text">
            <span className="stat-value">16.7ms</span>
            <span className="stat-label">Inference Time</span>
          </div>
        </div>
        <div className="stat-item">
          <span className="stat-icon">💾</span>
          <div className="stat-text">
            <span className="stat-value">11.7MB</span>
            <span className="stat-label">Model Size</span>
          </div>
        </div>
        <div className="stat-item">
          <span className="stat-icon">🎯</span>
          <div className="stat-text">
            <span className="stat-value">92%+</span>
            <span className="stat-label">Accuracy</span>
          </div>
        </div>
        <div className="stat-item">
          <span className="stat-icon">🚀</span>
          <div className="stat-text">
            <span className="stat-value">Real-time</span>
            <span className="stat-label">Detection</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
