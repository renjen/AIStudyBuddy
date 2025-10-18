#AI Study Buddy — Chat with your notes (RAG)

Upload PDFs or notes and ask questions. Uses LangChain + Chroma + OpenAI. Shows sources with page numbers. Works locally or with Docker.

<p align="left">
  <a href="https://github.com/<your-username>/<repo-name>/issues">Report a bug</a> ·
  <a href="https://github.com/<your-username>/<repo-name>/issues">Request a feature</a>
</p>

##Features
- Upload **PDF/TXT/MD** and index into Chroma (persisted to `.chroma/`)
- Ask questions; answers include **citations** (file + page)
- Adjustable **chunk size/overlap** and **top-k retrieval**
- Runs with **Streamlit** (no backend setup)
- Optional **Docker** for one-command start

##Quickstart

###Local
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # put your OpenAI key inside
streamlit run app.py
