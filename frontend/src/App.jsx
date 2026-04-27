import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Activity, AlertCircle, CheckCircle2, Loader2, FileImage, Zap, Brain } from 'lucide-react'
import axios from 'axios'
import './App.css'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ''

// ── Subcomponents ──────────────────────────────────────────────────────────────

function Header() {
  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="logo">
          <div className="logo-icon">
            <Activity size={22} strokeWidth={2.5} />
          </div>
          <div className="logo-text">
            <span className="logo-name">PneumoScan</span>
            <span className="logo-badge">AI</span>
          </div>
        </div>

      </div>
    </header>
  )
}

function DropZone({ onFileAccepted, isLoading }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) onFileAccepted(acceptedFiles[0])
  }, [onFileAccepted])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] },
    maxFiles: 1,
    disabled: isLoading,
  })

  return (
    <div
      {...getRootProps()}
      className={`dropzone ${isDragActive ? 'dropzone--active' : ''} ${isLoading ? 'dropzone--disabled' : ''}`}
      id="xray-dropzone"
    >
      <input {...getInputProps()} id="xray-file-input" />
      <div className="dropzone-content">
        <div className={`dropzone-icon ${isDragActive ? 'dropzone-icon--active' : ''}`}>
          {isDragActive ? <FileImage size={40} /> : <Upload size={40} />}
        </div>
        <p className="dropzone-title">
          {isDragActive ? 'Drop X-Ray here' : 'Drag & Drop Chest X-Ray'}
        </p>
        <p className="dropzone-sub">or click to browse — JPEG / PNG supported</p>
      </div>
    </div>
  )
}

function ImagePreview({ file }) {
  return (
    <motion.div
      className="preview-container"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <img
        src={URL.createObjectURL(file)}
        alt="Chest X-Ray preview"
        className="preview-img"
      />
      <div className="preview-meta">
        <FileImage size={14} />
        <span className="mono">{file.name}</span>
        <span className="text-muted">({(file.size / 1024).toFixed(1)} KB)</span>
      </div>
    </motion.div>
  )
}

function ResultCard({ result }) {
  const isNormal = result.label === 'NORMAL'
  const confPct  = (result.confidence * 100).toFixed(1)

  return (
    <motion.div
      className={`result-card ${isNormal ? 'result-card--normal' : 'result-card--pneumonia'}`}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      id="prediction-result"
    >
      {/* Status badge */}
      <div className="result-header">
        <div className={`result-icon ${isNormal ? 'result-icon--normal' : 'result-icon--pneumonia'}`}>
          {isNormal
            ? <CheckCircle2 size={32} strokeWidth={2} />
            : <AlertCircle  size={32} strokeWidth={2} />
          }
        </div>
        <div>
          <p className="result-label-text">Prediction</p>
          <h2 className={`result-label ${isNormal ? 'text-normal' : 'text-pneumonia'}`}>
            {result.label}
          </h2>
        </div>
        <div className="result-timing mono text-muted">
          {result.inference_time_ms.toFixed(1)} ms
        </div>
      </div>

      {/* Confidence bar */}
      <div className="confidence-section">
        <div className="confidence-row">
          <span className="text-secondary">Confidence</span>
          <span className={`mono ${isNormal ? 'text-normal' : 'text-pneumonia'}`}>
            {confPct}%
          </span>
        </div>
        <div className="confidence-track">
          <motion.div
            className={`confidence-fill ${isNormal ? 'confidence-fill--normal' : 'confidence-fill--pneumonia'}`}
            initial={{ width: 0 }}
            animate={{ width: `${confPct}%` }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
          />
        </div>
      </div>

      {/* LLM Report */}
      {result.report && (
        <div className="report-section">
          <p className="report-title"><Brain size={14} /> AI Diagnostic Report</p>
          <p className="report-body">{result.report}</p>
        </div>
      )}

      {!result.report && (
        <p className="report-placeholder text-muted mono">
          LLM report — available in Phase 4
        </p>
      )}
    </motion.div>
  )
}

// ── Main App ────────────────────────────────────────────────────────────────────

export default function App() {
  const [file,      setFile]      = useState(null)
  const [result,    setResult]    = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error,     setError]     = useState(null)

  const handleFileAccepted = (f) => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setIsLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const { data } = await axios.post(`${BACKEND_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      })
      setResult(data)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Unknown error'
      setError(`Analysis failed: ${msg}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <Header />

      <main className="app-main">
        {/* Hero */}
        <section className="hero">
          <motion.h1
            className="hero-title gradient-text"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y:  0 }}
            transition={{ duration: 0.6 }}
          >
            Chest X-Ray Pneumonia Detection
          </motion.h1>
          <motion.p
            className="hero-sub text-secondary"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            Upload a PA chest radiograph for instant AI-powered analysis
          </motion.p>
        </section>

        {/* Upload area */}
        <div className="upload-grid">
          <div className="upload-column">
            <DropZone onFileAccepted={handleFileAccepted} isLoading={isLoading} />

            <AnimatePresence>
              {file && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <ImagePreview file={file} />

                  <div className="action-row">
                    <button
                      id="analyze-btn"
                      className="btn btn-primary"
                      onClick={handleAnalyze}
                      disabled={isLoading}
                    >
                      {isLoading
                        ? <><div className="spinner" /> Analyzing…</>
                        : <><Activity size={16} /> Analyze X-Ray</>
                      }
                    </button>
                    <button
                      id="reset-btn"
                      className="btn btn-ghost"
                      onClick={handleReset}
                      disabled={isLoading}
                    >
                      Reset
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  className="error-banner"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  id="error-banner"
                >
                  <AlertCircle size={16} />
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Result */}
          <div className="result-column">
            <AnimatePresence mode="wait">
              {result ? (
                <ResultCard key="result" result={result} />
              ) : (
                <motion.div
                  key="placeholder"
                  className="result-placeholder glass"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Activity size={48} className="text-muted" strokeWidth={1} />
                  <p className="text-muted">Prediction will appear here</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>


    </div>
  )
}
