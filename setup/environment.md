## Local LLM Setup

**Ollama** runs on the host machine (not containerised) at http://localhost:11434.

### Model choice: llama3.2:3b
Hardware constraints make larger models (llama3 8B, mistral 7B) impractical. llama3.2:3b (~2GB) runs comfortably within these limits while remaining within the llama3 model family.

### Why Ollama runs on the host, not in Docker
Containerising Ollama prevents hardware acceleration. The backend and frontend
are containerised; LLM inference is not. This is a deliberate architectural
decision.

### Starting Ollama
Ollama starts automatically on Windows after installation (system tray icon).
To start manually if needed:
`ollama serve`