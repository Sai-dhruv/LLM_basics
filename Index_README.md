# Index_README — Generative AI Training (Step-by-Step)

Use this as a **teaching index** for juniors. It follows the flow in your book screenshots and turns it into a **clear learning path**, with what to teach + mini‑tasks.

---

## How to run this training
- **Audience:** Juniors (0–3 yrs), software/data engineers
- **Format:** 10–12 sessions (60–90 mins each) + hands‑on
- **Rule:** Every session ends with **one small demo** + **one quiz** + **one homework**.

---

## Step-by-step Index (Teaching Plan)

### 0) Kickoff + Setup (Session 0)
**Goal:** Everyone can run a notebook and call an LLM API safely.
- Install: Python, Jupyter, venv/conda, Git
- Explain: API keys, secrets, cost, rate limits
- Safety basics: PII, internal docs, prompt injection awareness
**Hands-on:** “Hello LLM” notebook (prompt → response)

---

## 1) Generative AI Use Cases, Fundamentals, and Project Life Cycle
### 1.1 Use Cases and Tasks
- Text: summarization, Q&A, extraction, classification, rewrite
- Code: code generation, tests, refactor hints
- Business: support bots, search assistants, report automation
**Hands-on:** Pick 1 use case for your team and write a 1‑page problem statement.

### 1.2 Foundation Models and Model Hubs
- What is a foundation model?
- Model hubs: open models vs hosted models
- Selection criteria: quality, latency, cost, context window, safety
**Hands-on:** Compare 2 models on the same prompt; note differences.

### 1.3 Generative AI Project Life Cycle
- Discover → prototype → evaluate → deploy → monitor → improve
- Data readiness + governance
**Hands-on:** Draw a “GenAI project life cycle” diagram for your office use case.

### 1.4 Generative AI on AWS
- High-level AWS building blocks (storage, compute, IAM, networking)
- Where GenAI fits: RAG, fine‑tuning, inference
**Hands-on:** Identify which AWS services your org already uses and where GenAI would plug in.

### 1.5 Why Generative AI on AWS?
- Security/IAM, scale, managed services, compliance
**Hands-on:** List 5 security controls you’d enforce in production.

### 1.6 Building Generative AI Applications on AWS
- Reference architecture: app → orchestration → retrieval/tools → model
**Hands-on:** Sketch an architecture (web/API + vector DB + LLM + logging).

---

## 2) Prompt Engineering and In-Context Learning
### 2.1 Prompts and Completions
- Prompt = instruction + context + examples + output format
- Completions and deterministic vs creative outputs
**Hands-on:** Convert a vague prompt into a structured one.

### 2.2 Tokens
- Tokens vs words; why tokens affect cost + latency
- Context window basics
**Hands-on:** Estimate token usage for a few prompts (roughly).

### 2.3 Prompt Engineering
- Role prompting, constraints, formatting, tone
- Few-shot prompting
**Hands-on:** Create a “prompt template” for summarizing meeting notes.

### 2.4 Prompt Structure
**Instruction**
- Clear task, constraints, output schema

**Context**
- Provide only needed information; remove noise
**Hands-on:** Build a prompt that outputs **JSON** only (with keys you define).

### 2.5 In-Context Learning with Few-Shot Inference
- Zero-shot vs one-shot vs few-shot
- When examples help
**Hands-on:** Add 2 examples and compare output quality.

### 2.6 In-Context Learning Gone Wrong
- Overfitting to examples, contradictions, ambiguous instructions
**Hands-on:** Show a failure example and fix it.

### 2.7 In-Context Learning Best Practices
- Keep examples representative; validate edge cases
**Hands-on:** Write 5 “test prompts” to validate your prompt.

### 2.8 Prompt-Engineering Best Practices
- Be explicit, constrain output, define audience, add refusal rules
**Hands-on:** Add a guardrail: “If info missing, ask a question.”

### 2.9 Inference Configuration Parameters
- Temperature, top_p, max_tokens, stop, frequency/presence penalties
**Hands-on:** Demonstrate temperature 0 vs 1 with same prompt.

---

## 3) Large-Language Foundation Models
### 3.1 What is an LLM (High level)
- Pretraining + next-token prediction
- Why they “sound right” even when wrong
**Hands-on:** Explain LLMs in 5 sentences (juniors practice).

### 3.2 Tokenizers
- BPE/SentencePiece concept, token boundaries
**Hands-on:** Show how a sentence becomes tokens (simple demo).

### 3.3 Embedding Vectors
- What embeddings are, similarity search, cosine distance
- How embeddings power retrieval/RAG
**Hands-on:** Create embeddings for 10 sentences; retrieve nearest neighbors.

### 3.4 Transformer Architecture (Conceptual)
- Inputs & context window
- Embedding layer
- Encoder / Decoder (conceptual differences)
- Self-attention
- Softmax output
**Hands-on:** Whiteboard “attention” with a small sentence.

### 3.5 Types of Transformer-Based Foundation Models
- Encoder-only, decoder-only, encoder-decoder
- When to use which (classification vs generation)
**Hands-on:** Map tasks to model types.

### 3.6 Pretraining Datasets
- Web/text/code; quality and bias considerations
**Hands-on:** Discuss why data quality matters.

### 3.7 Scaling Laws + Compute-Optimal Models
- Bigger data/model/compute trends
- Tradeoffs (cost vs quality)
**Hands-on:** Explain scaling tradeoffs with a simple analogy.

---

## 4) Memory and Compute Optimizations
### 4.1 Memory Challenges
- Why large models need lots of VRAM
- Batch size, sequence length, KV cache
**Hands-on:** Explain KV cache in plain English.

### 4.2 Data Types and Numerical Precision
- fp16, bfloat16, fp8, int8 (what they mean)
**Hands-on:** Quick “precision vs size” table exercise.

### 4.3 Quantization
- What it is and why it helps inference
**Hands-on:** Compare a quantized model vs full precision (conceptually).

### 4.4 Optimizing Self-Attention
- FlashAttention (why it’s faster)
- Grouped-Query Attention (why it reduces memory)
**Hands-on:** Explain these as “engineering optimizations” not math.

### 4.5 Distributed Computing
- DDP vs FSDP (high level)
- Performance and scaling
- Distributed computing on AWS (SageMaker, Trainium/Neuron)
**Hands-on:** Draw “data parallel” vs “sharded” picture.

---

## 5) Fine-Tuning and Evaluation
### 5.1 Instruction Fine-Tuning
- What changes vs just prompting
- Example families: Llama 2-Chat, Falcon-Chat, FLAN-T5
**Hands-on:** Decide: prompt vs fine-tune for a given requirement.

### 5.2 Instruction Dataset
- Multitask datasets (FLAN-style)
- Prompt templates
- Converting your data into instruction format
**Hands-on:** Create 20 instruction pairs (input/output) from your office use case.

### 5.3 Instruction Fine-Tuning on AWS
- Studio / JumpStart / HuggingFace Estimator (conceptual)
**Hands-on:** Outline training steps + required artifacts.

### 5.4 Evaluation
- Metrics: accuracy-like, faithfulness/groundedness, latency, cost
- Benchmarks and datasets
**Hands-on:** Create an evaluation checklist for your app.

---

## 6) Parameter-Efficient Fine-Tuning (PEFT)
### 6.1 Full Fine-Tuning vs PEFT
- When full FT is expensive and risky
- PEFT updates fewer params
**Hands-on:** Explain PEFT with “adapter” analogy.

### 6.2 LoRA and QLoRA
- LoRA fundamentals
- Rank, target modules/layers
- Applying LoRA
- Merging adapters vs keeping separate
- Full FT vs LoRA performance
- Prompt tuning / soft prompts
**Hands-on:** Decide LoRA rank + target modules for a toy experiment (no training required).

---

## 7) RLHF — Reinforcement Learning from Human Feedback
### 7.1 Human Alignment: Helpful, Honest, Harmless
- Why alignment is needed
**Hands-on:** Create 5 “good vs bad” response examples.

### 7.2 RLHF Overview
- Preference data → reward model → policy optimization (conceptual)
**Hands-on:** Build a tiny preference dataset (A vs B answers).

### 7.3 Train a Custom Reward Model (Conceptual steps)
- Collect data with human-in-the-loop
- Ground Truth workflows
**Hands-on:** Design a labeling guide (what makes an answer good).

### 7.4 PPO + RLHF (High level)
- Why PPO is used
**Hands-on:** Explain “optimize behavior with feedback” in simple terms.

### 7.5 Mitigate Reward Hacking + Evaluate RLHF Model
- Reward hacking examples
- Qualitative vs quantitative evaluation
- Before/after comparisons
**Hands-on:** Identify 3 reward-hacking risks for your use case.

---

## 8) Model Deployment Optimizations
### 8.1 Model optimizations for inference
- Latency vs cost vs quality tradeoffs
**Hands-on:** Define SLOs: p95 latency + cost per request.

### 8.2 Pruning
- Remove weights/neurons (concept)
**Hands-on:** Discuss when pruning is worth it.

### 8.3 Post-Training Quantization with GPTQ
- PTQ concepts
**Hands-on:** “What can go wrong?” checklist (accuracy drop, edge cases).

### 8.4 Distillation
- Small student learns from big teacher
**Hands-on:** Identify where distillation fits (mobile/edge/low cost).

### 8.5 Large Model Inference Container
- Packaging, dependencies, reproducibility
**Hands-on:** Outline a container checklist (model files, health checks, logs).

### 8.6 AWS Inferentia
- Purpose-built inference hardware (concept)
**Hands-on:** When to choose specialized inference hardware.

### 8.7 Model Update and Deployment Strategies
- A/B testing
- Shadow deployment
- Monitoring
- Autoscaling policies
**Hands-on:** Draw rollout plan for v1 → v2.

---

## 9) Context-Aware Reasoning Apps using RAG and Agents
### 9.1 LLM limitations
- Hallucination
- Knowledge cutoff
**Hands-on:** Show a hallucination example and how to detect it.

### 9.2 Retrieval-Augmented Generation (RAG)
- External sources of knowledge
- RAG workflow: load → chunk → embed → retrieve → generate
**Hands-on:** Build a mini RAG on 3 internal docs (local files).

### 9.3 Document loading
- PDFs, docs, HTML, Confluence, etc.
**Hands-on:** Load 2 file types and unify text.

### 9.4 Chunking
- Why chunking matters (recall vs precision)
- Chunk size/overlap
**Hands-on:** Try 2 chunk sizes and compare retrieval.

### 9.5 Retrieval + reranking
- Vector search
- Reranking (why)
**Hands-on:** Add reranking and measure better top-3 results.

### 9.6 Prompt augmentation
- Put retrieved context into the prompt safely
**Hands-on:** Add citations + “answer only from context” rule.

### 9.7 RAG orchestration & implementation
- Chains, routers, tools
- Retrieval chains
- MMR reranking (diversity)
**Hands-on:** Create a “router” that chooses between FAQ vs RAG.

---

## 10) Multimodal Foundation Models
### 10.1 Use cases
- Image understanding, captioning, VQA, document AI
**Hands-on:** Caption 3 images and extract key fields.

### 10.2 Multimodal prompt engineering best practices
- Provide clear instructions, specify output format
**Hands-on:** Create a prompt to extract a table from an image (conceptually).

### 10.3 Image generation and enhancement
- Text-to-image
- Image editing/enhancement
- Inpainting/outpainting/depth-to-image
**Hands-on:** Create 3 prompts: generate, edit, inpaint.

### 10.4 Image captioning and Visual Question Answering
- When to use which
**Hands-on:** Ask 5 questions about an image; verify answers.

### 10.5 Model evaluation (multimodal)
- Quality + safety + bias + accuracy
**Hands-on:** Make a rubric for image output review.

### 10.6 Diffusion fundamentals (high level)
- Forward vs reverse diffusion
- U-Net role (conceptual)
**Hands-on:** Explain diffusion in a simple story analogy.

### 10.7 Stable Diffusion architectures (high level)
- SD2: text encoder, U-Net, decoder, cross-attention, scheduler
- SDXL: base + refiner
**Hands-on:** Identify what each component does in 1 line.

---

## 11) Controlled Generation and Fine-Tuning with Stable Diffusion
### 11.1 ControlNet
- Controlling pose/edges/depth for consistent generation
**Hands-on:** Pick a control signal (canny/pose) and explain its purpose.

### 11.2 Fine-tuning for images
- DreamBooth
- DreamBooth + PEFT-LoRA
- Textual inversion
**Hands-on:** Compare when to use DreamBooth vs LoRA.

### 11.3 Human alignment (vision)
- Safety, copyright, brand policy, content moderation
**Hands-on:** Create a “safe image generation policy” for office.

---

## 12) Amazon Bedrock — Managed Service for Generative AI
### 12.1 Bedrock foundation models
- Titan, partner models (concept)
**Hands-on:** List which models you’d use for: chat, embeddings, images.

### 12.2 Bedrock inference APIs
- Invoke model, streaming, guardrails (concept)
**Hands-on:** Define required logs: prompt, response, latency, tokens, user id.

### 12.3 Common Bedrock use cases
- Generate SQL code
- Summarize text
- Embeddings
- Fine-tuning
- Agents
- Multimodal models (text↔image)
**Hands-on:** Build a “Bedrock use-case matrix” (task → model → input/output).

### 12.4 Data privacy and network security
- IAM roles, VPC endpoints (concept), encryption, audit trails
**Hands-on:** Draft a checklist for production approval.

### 12.5 Governance and monitoring
- Evaluation, red teaming, drift monitoring, cost control
**Hands-on:** Define what to monitor (quality, hallucination rate, cost).

---

## Capstone (Final Project)
**Goal:** Build a small “Glean-style” internal assistant (safe demo).
- Sources: 3–10 documents (local), optional Slack/Jira mock data
- Features: RAG + citations + basic routing + evaluation checklist
**Deliverables:**
1) Architecture diagram  
2) Prompt templates  
3) RAG pipeline notebook  
4) Evaluation sheet (10 questions)  
5) Short demo recording (2–3 mins)

---

## Quick Checklist for You (Trainer)
- [ ] Prepared slides or whiteboard notes for each session
- [ ] Sample prompts + failure examples
- [ ] Small dataset/docs for RAG labs
- [ ] A rubric for evaluating answers (groundedness + relevance)
- [ ] A safe-use policy for internal data

---

## Suggested session mapping (optional)
1) Sessions 1–2: Chapters 1–2  
2) Sessions 3–4: Chapter 3 (LLM basics + embeddings)  
3) Session 5: Chapter 4 (optimizations)  
4) Sessions 6–7: Chapters 5–6 (fine-tuning + PEFT)  
5) Session 8: Chapter 7 (RLHF)  
6) Session 9: Chapter 8 (deployment)  
7) Sessions 10–11: Chapter 9 (RAG + agents)  
8) Session 12: Chapters 10–12 (multimodal + Bedrock) + Capstone demo

---

If you want, tell me:
- Your team’s **real use case** (ex: internal policy Q&A, Jira summarizer, Slack digest),
and I’ll generate **session slides + lab notebooks** for it.
