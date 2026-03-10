# AI Research Paper Reviewer

A comprehensive application for reviewing research papers using AI agents.

## Project Structure

- `backend/` - Python FastAPI backend
  - `main.py` - FastAPI application
  - `agents.py` - Multi-agent review system
  - `pdf_parser.py` - PDF text extraction and parsing
  - `arxiv_search.py` - ArXiv API integration
  - `llm_client.py` - LLM interface (local and mock)
- `frontend/` - Web frontend
  - `index.html` - Main HTML page
  - `style.css` - Styles
  - `app.js` - JavaScript logic
- `tests/` - Unit tests
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # make sure python-multipart is installed for FastAPI file uploads
   ```

2. For local LLM, install Ollama and pull models:
   ```bash
   ollama pull llama3.2:3b
   ```

3. For cloud LLM (optional), sign up at https://ollama.ai/cloud and configure access

3. Run the backend:
   ```bash
   cd backend
   python main.py
   ```

4. Open `frontend/index.html` in a browser.

## LLM Options

The application supports three LLM modes:

1. **Mock LLM** (default for testing): Returns deterministic responses instantly
2. **Local LLM**: Uses Ollama with local models (llama3.2:3b, mistral, etc.)
3. **Cloud LLM**: Uses Ollama Cloud models (requires Ollama account and cloud access)

### Setting up Ollama Cloud (Detailed Steps)

1. **Sign up for Ollama Cloud**:
   - Go to https://ollama.ai/cloud
   - Create an account (free tier available)
   - Verify your email

2. **Install Ollama Desktop App**:
   - Download from https://ollama.ai/download
   - Install and sign in with your cloud account

3. **Configure Cloud Access**:
   - Open Ollama app
   - Go to Settings/Preferences
   - Sign in with your Ollama Cloud account
   - Enable cloud features

4. **Set Environment Variable** (optional, for API key):
   ```powershell
   # Set this in your system environment variables or in a .env file
   $env:OLLAMA_CLOUD_API_KEY = "your-api-key-here"
   ```

5. **Pull Cloud Models**:
   ```powershell
   # These models are available through cloud:
   ollama pull llama3.1:70b
   ollama pull llama3.1:405b
   ollama pull mixtral:8x7b
   ollama pull codellama:34b
   ```

6. **Test Cloud Model**:
   ```powershell
   ollama run llama3.1:70b
   # Type a message to test
   ```

### Troubleshooting Cloud Access

- **"Model not found"**: Make sure you're signed into Ollama Cloud in the app
- **"Access denied"**: Check your cloud subscription/credits
- **"Network error"**: Ensure internet connection and Ollama app is running
- **API Key Issues**: Some cloud models may require API key configuration

### Alternative: Use Local Models
If cloud setup is complex, use local models instead:
```powershell
ollama pull llama3.2:3b  # Free, runs locally
ollama pull mistral      # Another good local option
```