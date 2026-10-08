# 🗨️ LangGraph + Streamlit Chatbot

A modern chatbot built with **LangGraph** (backend workflow/state management) and **Streamlit** (frontend UI). It integrates with the **Groq API** for fast inference and supports **short-term memory** to maintain context across turns.

---

## 🚀 Features

- Interactive chat UI with avatars (`st.chat_message`)
- Short-term memory (rolling context window)
- Backend powered by LangGraph workflows
- Frontend built in Streamlit for instant deployment
- Uses **Groq API key** for model inference
- Clean separation of backend (`langgraph_backend.py`) and frontend (`streamlit_fronted.py`)
- Streaming responses (token-by-token typing effect)
- Streaming responses (token-by-token typing effect)
- Multi-chat sidebar with conversation history
- Persistent chat history using SQLite

---
## 📸 Demo

![Chatbot Demo](https://res.cloudinary.com/dnfkkxlvi/image/upload/v1791482035/Screenshot_2026-10-08_232008_qy16wm.png)

---

## 📂 Project Structure

```text
Chatbot/
├── __pycache__/            # Python cache files
├── venv/                   # Virtual environment (ignored in git)
├── .gitignore              # Ignore venv, .env, etc.
├── langgraph_backend.py    # LangGraph workflow logic
├── streamlit_fronted.py    # Streamlit UI
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd Chatbot
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run streamlit_fronted.py
```

---

## 🧠 Memory

- **Short-term memory:** Maintains a rolling window of recent conversation turns
- **Extendable to long-term memory** with vector stores (FAISS, Chroma, etc.)

---

## ⚙️ .gitignore

Already included to keep sensitive files safe:

```text
venv/
.env
__pycache__/
*.log
```

---

## 📜 requirements.txt

A ready-to-use template for dependencies:

```text
streamlit
langgraph
groq
python-dotenv
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.