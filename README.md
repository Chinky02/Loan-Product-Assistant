# Bank of Maharashtra Loan Product Assistant

A comprehensive RAG-based (Retrieval-Augmented Generation) system that helps users get information about Bank of Maharashtra's loan products through an intelligent conversational interface.

## Project Setup

### Prerequisites
- Python 3.8+
- Git
- Ollama (for local LLM inference)

### 1. Clone and Setup Environment
```bash
git clone <repository-url>
cd Loan-Product-Assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama
```bash
# Install Ollama from https://ollama.com/
# Download and run the desired model
ollama pull llama3
```

### 4. Build Knowledge Base
```bash
python scripts/build_knowledge_base.py --processed_dir data/processed --output_dir knowledge_base
```

### 5. Run the Application
```bash
# Start the API server
uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --reload

# Access the API at http://localhost:8000
# Interactive documentation at http://localhost:8000/docs
```

### 6. Test the System
```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Tell me about the Maha Super Housing Loan Scheme","top_k":5}'
```

## Architectural Decisions

### Libraries Chosen

#### Web Scraping
- **Beautiful Soup 4**: Selected for robust HTML parsing and extraction from Bank of Maharashtra's website
- **Requests**: Reliable HTTP library for fetching web pages with proper headers and timeout handling
- **Rationale**: These libraries provide stable, well-documented solutions for web scraping with good error handling capabilities

#### Data Processing
- **Pandas**: For structured data manipulation and cleaning
- **JSON**: Native Python library for handling structured data storage
- **Rationale**: JSON format provides flexibility for storing hierarchical loan product data while remaining human-readable

#### RAG Pipeline
- **FAISS (Facebook AI Similarity Search)**: Vector database for efficient similarity search
- **Sentence Transformers**: For generating high-quality embeddings
- **Ollama**: Local LLM inference without API costs
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Rationale**: This combination provides a cost-effective, locally-hosted RAG solution with excellent performance

### Data Strategy

#### Text Chunking Approach
- **Chunk Size**: 800 characters with 120-character overlap
- **Strategy**: Sliding window chunking to preserve context across boundaries
- **Preprocessing**: Whitespace normalization and JSON flattening
- **Rationale**: 
  - 800 characters balances semantic coherence with embedding model constraints
  - 120-character overlap ensures important information isn't lost at chunk boundaries
  - JSON flattening makes hierarchical loan data searchable at granular levels

#### Data Structure
```
data/
├── raw/           # Original HTML files from web scraping
├── processed/     # Cleaned JSON files with structured loan data
└── scripts/       # Data processing and scraping scripts
```

### Model Selection

#### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Rationale**: 
  - Fast inference (384-dimensional embeddings)
  - Good performance on semantic similarity tasks
  - Lightweight enough for local deployment
  - Strong multilingual capabilities for handling Hindi/Marathi content

#### Language Model
- **Primary**: Llama 3 via Ollama
- **Fallback**: Automatic fallback to alternative models
- **Rationale**:
  - Local inference eliminates API costs and latency
  - Strong reasoning capabilities for financial domain
  - Privacy-preserving (no data sent to external APIs)
  - Ollama provides easy model management and switching

### AI Tools Used

#### Development Tools
- **GitHub Copilot**: Code completion and documentation generation
- **ChatGPT/Claude**: Architecture planning and code review
- **Rationale**: These tools accelerated development while maintaining code quality


## Challenges Faced

### 1. Dynamic Web Content
- **Challenge**: Bank websites often use JavaScript to load content dynamically
- **Solution**: Implemented robust HTML parsing with multiple fallback selectors to handle various page structures

### 2. Inconsistent Data Formats
- **Challenge**: Loan product pages had varying HTML structures and data organization
- **Solution**: Created flexible extraction functions with multiple parsing strategies and error handling

### 3. Text Chunking Optimization
- **Challenge**: Balancing chunk size for semantic coherence vs. retrieval relevance
- **Solution**: Experimented with different chunk sizes (400, 600, 800, 1000 characters) and settled on 800 with overlap

### 4. Embedding Model Performance
- **Challenge**: Finding the right balance between model size, speed, and accuracy
- **Solution**: Benchmarked multiple sentence-transformer models and chose all-MiniLM-L6-v2 for optimal performance

### 5. Local LLM Setup
- **Challenge**: Making the system accessible without requiring expensive API keys
- **Solution**: Integrated Ollama for local LLM inference with automatic model management

### 6. Multilingual Content
- **Challenge**: Bank website contains content in English, Hindi, and Marathi
- **Solution**: Used multilingual embedding models and preserved original language content in the knowledge base

## Potential Improvements

### If Given More Time

#### 1. Enhanced Data Collection
- **Web Crawling**: Implement automated monitoring for new loan products
- **Multi-source Integration**: Scrape from multiple bank websites for comparative analysis
- **Real-time Updates**: Scheduled scraping to keep loan information current

#### 2. Advanced RAG Techniques
- **Hybrid Search**: Combine dense embeddings with sparse keyword search (BM25)
- **Query Expansion**: Implement query rewriting for better retrieval
- **Re-ranking**: Add cross-encoder models for improved context relevance

#### 3. User Experience Enhancements
- **Web Interface**: Build React/Vue.js frontend for better user interaction
- **Chat Memory**: Implement conversation history and context awareness
- **Multi-modal Support**: Add support for loan application forms and documents

#### 4. Production Readiness
- **Containerization**: Docker setup for easy deployment
- **Monitoring**: Add logging, metrics, and health checks
- **Caching**: Implement Redis for query caching and session management
- **Authentication**: Add user authentication and rate limiting

#### 5. Advanced Analytics
- **Query Analytics**: Track popular questions and improve responses
- **A/B Testing**: Test different RAG configurations for optimal performance
- **Feedback Loop**: Implement user feedback collection for continuous improvement

#### 6. Model Improvements
- **Fine-tuning**: Train domain-specific embeddings on banking/finance corpus
- **Custom LLM**: Fine-tune Llama models specifically for banking domain
- **Evaluation Framework**: Implement automated evaluation metrics for response quality

#### 7. Data Quality
- **Entity Recognition**: Extract and structure key loan parameters (interest rates, eligibility, etc.)
- **Data Validation**: Implement automated checks for data consistency and completeness
- **Structured Outputs**: Generate structured responses with key loan details

#### 8. Integration Capabilities
- **API Extensions**: Add endpoints for loan eligibility checking and application status
- **Third-party Integration**: Connect with loan application systems
- **Mobile App**: Develop mobile application for better accessibility

---

## Current System Features

- **Modular Architecture**: Clean separation of concerns across services
- **Robust Error Handling**: Comprehensive error handling with logging
- **Automatic Fallbacks**: Model fallback support for reliability
- **Interactive Documentation**: FastAPI automatic API documentation
- **Health Monitoring**: Detailed health checks and status reporting
- **Cost-effective**: Uses local models with no API costs
- **Scalable Design**: Architecture supports easy scaling and enhancement
