# OSHA Compliance Evaluator

## Overview

This is a full-stack web application that provides AI-powered OSHA compliance evaluation for instruction manuals. The system uses Natural Language Inference (NLI) models to automatically identify relevant safety regulations and evaluate compliance requirements.

## System Architecture

The application follows a modern full-stack architecture with the following key components:

### Frontend Architecture
- **Framework**: React with TypeScript and Vite for development tooling
- **UI Library**: Shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with custom CSS variables for theming
- **Routing**: Wouter for client-side routing
- **State Management**: React Query (TanStack Query) for server state management
- **Build Tool**: Vite with TypeScript support

### Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **Database**: PostgreSQL with Drizzle ORM
- **Database Provider**: Neon Database (serverless PostgreSQL)
- **Session Management**: PostgreSQL session store with connect-pg-simple

### Data Storage Solutions
- **Primary Database**: PostgreSQL via Neon Database serverless
- **ORM**: Drizzle ORM with TypeScript-first approach
- **Schema Management**: Drizzle Kit for migrations and schema management
- **Vector Storage**: Qdrant for storing and querying document embeddings (based on attached Python files)

### AI/ML Integration
- **NLI Models**: RoBERTa-based models for Natural Language Inference
- **Embeddings**: HuggingFace embeddings for document similarity search
- **Vector Database**: Qdrant for semantic search capabilities
- **Model Framework**: PyTorch Lightning with transformers library

## Key Components

### Frontend Components
1. **File Upload System**: Drag-and-drop interface for JSON instruction manual uploads
2. **Results Display**: Expandable text components for viewing evaluation results
3. **Navigation**: Clean navigation with routing between Home and About pages
4. **UI Components**: Comprehensive set of reusable components from Shadcn/ui

### Backend Components
1. **Express Server**: Main application server with middleware setup
2. **Storage Interface**: Abstracted storage layer with in-memory implementation
3. **Route Registration**: Modular route handling system
4. **Development Tools**: Vite integration for hot reloading in development

### Shared Components
1. **Schema Definitions**: Drizzle schema with Zod validation
2. **Type Safety**: Shared TypeScript types between frontend and backend

## Data Flow

1. **File Upload**: Users upload JSON instruction manuals through the frontend
2. **Processing**: Files are sent to the AI processing service (FastAPI backend)
3. **NLI Analysis**: Each instruction step is analyzed against OSHA regulations using RoBERTa models
4. **Vector Search**: Relevant regulations are retrieved using semantic similarity search
5. **Results Display**: Compliance evaluations are returned and displayed with expandable text components

## External Dependencies

### Core Dependencies
- **@neondatabase/serverless**: Serverless PostgreSQL connection
- **drizzle-orm**: TypeScript ORM for database operations
- **@tanstack/react-query**: Data fetching and caching
- **@radix-ui/***: Headless UI components
- **tailwindcss**: Utility-first CSS framework

### AI/ML Dependencies (Python Service)
- **transformers**: HuggingFace transformers for NLI models
- **pytorch-lightning**: ML framework for model training
- **qdrant-client**: Vector database client
- **langchain**: LLM application framework

### Development Dependencies
- **vite**: Build tool and development server
- **typescript**: Type safety and development experience
- **drizzle-kit**: Database migrations and introspection

## Authentication and Authorization

Currently, the application uses a basic user schema with username/password authentication. The system is prepared for session-based authentication using PostgreSQL session storage, though the full authentication flow is not yet implemented.

## Deployment Strategy

### Development
- **Frontend**: Vite development server with hot module replacement
- **Backend**: Node.js with tsx for TypeScript execution
- **Database**: Neon Database serverless PostgreSQL

### Production
- **Build Process**: Vite builds the frontend, esbuild bundles the backend
- **Runtime**: Node.js production server serving built assets
- **Database**: Production Neon Database instance
- **Environment**: Replit-optimized with cartographer plugin for development

### Configuration
- Environment variables for database connection (DATABASE_URL)
- Tailwind configuration for design system consistency
- TypeScript configuration with path mapping for clean imports

## Project Structure

The application follows a hybrid architecture combining React frontend with FastAPI backend:

### FastAPI Backend Structure
```
osha_app/
├── src/
│   ├── main.py          # FastAPI main application with lifespan management
│   ├── model.py         # ML models, data processing, and NLI classes
│   └── nli_subapp.py    # NLI prediction endpoints and API routes
└── tests/               # Test files (future implementation)
```

### Frontend Structure
- React application in `client/` directory
- Node.js/Express development server in `server/` directory
- Shared TypeScript schemas in `shared/` directory

## Changelog

```
Changelog:
- July 02, 2025. Initial setup
- July 02, 2025. Restructured FastAPI backend to match user's directory structure
  - Moved Python files to osha_app/src/ directory
  - Updated Dockerfile to use new module path
  - Created comprehensive README with project documentation
  - Maintained compatibility with existing React frontend
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```