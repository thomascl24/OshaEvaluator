# OSHA Compliance Evaluator

An AI-powered web application that automatically evaluates instruction manuals for OSHA compliance using Natural Language Inference (NLI) models.

## Overview

This application uses advanced RoBERTa-based models to analyze instruction manual steps against OSHA regulations and provide compliance evaluations. The system combines vector search with semantic similarity to identify relevant regulations and classify compliance relationships.

## Project Structure

```
.
├── osha_app/
│   ├── src/
│   │   ├── main.py          # FastAPI main application
│   │   ├── model.py         # ML models and data processing
│   │   └── nli_subapp.py    # NLI prediction endpoints
│   └── tests/               # Test files
├── client/                  # React frontend application
├── server/                  # Node.js/Express backend
├── Dockerfile              # Container configuration
├── poetry.lock             # Python dependency lock file
├── pyproject.toml          # Python project configuration
└── README.md               # This file
```

## Features

- **File Upload**: Drag-and-drop interface for JSON instruction manuals
- **AI Analysis**: RoBERTa-large model for natural language inference
- **Vector Search**: Qdrant integration for regulation similarity search
- **Real-time Results**: Clean table display with expandable text
- **Responsive Design**: Mobile-first layout with modern styling

## API Endpoints

- `GET /app/health` - Health check endpoint
- `POST /app/predict` - Upload JSON file and get compliance predictions

## Response Format

```json
{
  "predictions": ["entailment", "neutral", "contradiction"],
  "premises": ["Instruction step text..."],
  "hypotheses": ["OSHA regulation text..."]
}
```

## Technologies

### Backend
- **FastAPI**: Python web framework
- **PyTorch Lightning**: ML framework
- **RoBERTa**: Transformer model for NLI
- **Qdrant**: Vector database for embeddings
- **Redis**: Caching layer

### Frontend
- **React**: UI framework with TypeScript
- **Tailwind CSS**: Utility-first styling
- **Vite**: Build tool and development server
- **TanStack Query**: Data fetching and caching

## Installation

1. Install Python dependencies:
```bash
poetry install
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

## Docker Deployment

Build and run with Docker:
```bash
docker build -t osha-evaluator .
docker run -p 8000:8000 osha-evaluator
```

## Model Information

- **Base Model**: RoBERTa-Large (355M parameters)
- **Training**: SNLI dataset for natural language inference
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Store**: Qdrant for regulation embeddings

## Evaluation Types

- **Entailment**: Instruction step requires compliance with regulation
- **Contradiction**: Instruction step conflicts with regulation  
- **Neutral**: Instruction step is unrelated to regulation

## License

This project is for educational and research purposes.