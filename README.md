# ✨ Idea Spark AI

A light, colorful AI assistant built with **Streamlit** and **Google Gemini Flash**.

## Features
- 💬 **Chat**: streaming chatbot that remembers the conversation
- 🛠️ **Writing Tools**: 23 tools (summarize, rewrite, fix grammar, quiz, flashcards, translate and more)
- 📎 **File upload**: PDF, Word (.docx), TXT, MD, CSV, PNG, JPG (up to 15 MB)
- 🔁 Automatic retry and backup models when Google is busy or rate-limited

## Run locally
1. Get a free API key at https://aistudio.google.com (Get API key)
2. Install Python 3.10+, then:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Paste your key in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io, click **Create app**, choose your repo, branch `main`, file `app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Click **Deploy**.

## Notes
- Model names change often. See https://ai.google.dev/gemini-api/docs/models and use the
  "Custom model name" box in the sidebar if one stops working.
- Never commit your API key. `.gitignore` already excludes `.streamlit/secrets.toml` and `.env`.
