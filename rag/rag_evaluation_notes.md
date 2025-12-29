# 📘 Evaluating RAG Systems — Clear Notes

## 🔍 Goal of the Lesson
This lesson explains **how to evaluate Retrieval-Augmented Generation (RAG) systems** using a principled framework called the **RAG Triad**, implemented with **TruLens**.

> 🎯 **Goal:** Detect **hallucinations**, **retrieval failures**, and **answer quality issues** in RAG pipelines.

---

## 🧱 Core Concept: **RAG Triad**
The **RAG Triad** evaluates the **three key stages** of a RAG system:

1. **Retrieval**
2. **Grounding**
3. **Generation**

### ✅ The Three Metrics
- **Context Relevance**
- **Groundedness**
- **Answer Relevance**

These metrics are implemented as **feedback functions** in **TruLens**.

---

## 🛠️ Tools & Setup

### 🔑 Prerequisites
- **TruLens-eval**
- **LlamaIndex**
- **OpenAI API key**

### 🧰 Key Components
- **Tru object (`Tru`)** – manages the evaluation database
- Records **prompts, responses, intermediate steps, and scores**

---

## 📄 Document Processing & Retrieval

### 📚 Data Source
- PDF: *Building a Career in AI* by **Andrew Ng**

### 🔧 Indexing Setup
- Single merged document
- **Sentence Index**
- **bge-small-v1.5 embedding model**
- **GPT-3.5 Turbo** (temperature = 0.1)

### 🔍 Retrieval Engine
- **Sentence Window Query Engine**

---

## ❓ Example Query
**“How do you create your AI portfolio?”**

The system returns:
- Final answer
- Retrieved context
- Metadata

Evaluation is required to verify **trustworthiness**.

---

## 🔺 The RAG Triad — Detailed Breakdown

### 1️⃣ **Answer Relevance**
Checks whether the **final answer** addresses the **user query**.
- Score: **0–1**
- May include **supporting reasoning**

**Failure Mode:** Answer is true but unrelated.

---

### 2️⃣ **Context Relevance**
Evaluates whether retrieved documents are relevant to the query.
- Scores each retrieved chunk
- Computes a **mean relevance score**

**Failure Mode:** Retrieval pulls irrelevant context.

---

### 3️⃣ **Groundedness**
Checks whether the answer is **supported by retrieved context**.
- Sentence-level scoring
- Scores averaged

**Failure Mode (Hallucination):** Answer uses model knowledge not present in context.

---

## 🔁 Evaluation & Iteration Workflow
1. Start with basic RAG
2. Evaluate using **RAG Triad**
3. Identify failure modes
4. Improve retrieval
5. Re-evaluate

---

## 🪟 Sentence Window RAG Tuning
- **Small window** → insufficient context
- **Large window** → irrelevant information

Goal: **Balanced window size**

---

## 🧪 Feedback Function Types

### 🤖 LLM-Based
- GPT-3.5 / GPT-4
- Semantic and scalable

### 👤 Human Evaluation
- ~80% agreement with LLM judges

### 📊 Traditional NLP Metrics
- ROUGE, BLEU (syntactic, limited for RAG)

---

## 🗃️ Recording & Results

### 🧾 TruRecorder
- Logs inputs, outputs, scores, latency, and cost
- Stored in **JSON format**

### 📊 Streamlit Dashboard
- Leaderboard view
- Record-level inspection

**Example Insight:**
Low groundedness when statements lack supporting retrieved context.

---

## 🧠 Open-Book Exam Analogy
- **Context Relevance:** Opened the right page
- **Groundedness:** Used information from that page
- **Answer Relevance:** Answered the question asked

---

## 📝 Key Takeaways
- **RAG Triad = Context Relevance + Groundedness + Answer Relevance**
- **Groundedness** is key to detecting hallucinations
- Evaluation is **iterative**
- Sentence-window RAG improves grounding
