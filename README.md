# Analyst AI

A data analysis assistant powered by LangGraph and OpenAI, inspired by Julius AI. This application allows users to upload CSV and XLSX files for analysis and interact with the data through natural language conversations.

## Features

- File Upload: Support for CSV and XLSX files
- Interactive Analysis: Chat with your data using natural language
- Visualization: Generate plots and charts based on your queries
- LangGraph ReACT Architecture: Reasoning and Action framework for intelligent analysis
- Human-in-the-Loop: Verification and feedback system for critical analyses

## Project Structure

```
Analyst_AI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── routes.py
│   ├── agents/
│   │   └── analyst_agent.py
│   ├── utils/
│   └── tests/
└── frontend/
    └── (React + Tailwind components)
```

## Setup

### Backend Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

## Usage

1. Open the application in your browser (default: http://localhost:5173)
2. Enter your OpenAI API key in the left sidebar
3. Upload a CSV or XLSX file
4. Start chatting with your data!

## API Endpoints

- `POST /api/init`: Initialize a new analysis session
- `POST /api/upload/{session_id}`: Upload a data file
- `POST /api/chat/{session_id}`: Send a message to the analyst
- `DELETE /api/session/{session_id}`: End an analysis session

## Development

- Backend uses FastAPI with async support
- Frontend built with React and Tailwind CSS
- LangGraph for ReACT architecture
- Python data analysis libraries (pandas, seaborn, plotly)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 