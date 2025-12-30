# GEN_AI_Definitions

This README provides **short (2–3 lines), simple definitions** for core **Generative AI concepts**.  
Designed for **office juniors (0–3 years experience)** to quickly understand fundamentals.

---

## 1. Generative AI Fundamentals

### Generative AI
Generative AI refers to models that can **create new content** such as text, code, images, or summaries.  
These models learn patterns from large datasets and generate human-like outputs.

### Use Cases and Tasks
Generative AI is used for **chatbots, summarization, Q&A, code generation, and automation**.  
Each task focuses on reducing manual effort and improving productivity.

### Foundation Models
Foundation models are **large pre-trained models** trained on massive datasets.  
They can be adapted for many tasks using prompts, fine-tuning, or retrieval.

### Model Hubs
Model hubs store and distribute **pre-trained AI models**.  
They help teams discover, compare, and reuse models easily.

### Generative AI Project Life Cycle
A GenAI project follows **problem definition → prototype → evaluate → deploy → monitor**.  
Continuous feedback improves quality and safety.

---

## 2. Prompt Engineering and In-Context Learning

### Prompts and Completions
A prompt is the **instruction or input**, and completion is the **model’s output**.  
Clear prompts lead to better responses.

### Tokens
Tokens are the **small units of text** processed by LLMs.  
Token count affects cost, speed, and context size.

### Prompt Engineering
Prompt engineering is the practice of **writing effective instructions** for models.  
It improves results without changing the model.

### Prompt Structure
A good prompt includes **instruction, context, examples, and output format**.  
Structured prompts reduce ambiguity.

### In-Context Learning
In-context learning allows models to **learn from examples in the prompt**.  
No retraining is required.

### Zero-Shot Inference
Zero-shot inference means **no examples are provided**.  
The model relies only on its prior knowledge.

### Few-Shot Inference
Few-shot inference includes **multiple examples** in the prompt.  
It improves accuracy and consistency.

---

## 3. Large Language Models (LLMs)

### Large Language Models
LLMs are trained to **predict the next word or token** in a sequence.  
They power chatbots, summarization, and reasoning tasks.

### Tokenizers
Tokenizers convert text into **tokens understood by the model**.  
Different tokenizers affect performance and cost.

### Embeddings
Embeddings are **numeric representations of meaning**.  
They enable semantic search and retrieval.

### Transformer Architecture
Transformers use **self-attention** to understand relationships in text.  
They process data efficiently in parallel.

### Self-Attention
Self-attention lets the model **focus on important words** in context.  
It is the key innovation behind transformers.

---

## 4. Memory and Compute Optimizations

### Memory Challenges
Large models require **high GPU memory**.  
Optimization is needed to reduce cost.

### Numerical Precision
Lower precision formats reduce memory usage.  
They balance accuracy and performance.

### Quantization
Quantization converts model weights to **lower precision formats**.  
It speeds up inference and saves memory.

### FlashAttention
FlashAttention optimizes attention computation.  
It improves speed and reduces memory usage.

### Distributed Training
Distributed training spreads work across **multiple GPUs or machines**.  
It enables training large models efficiently.

---

## 5. Fine-Tuning and Evaluation

### Fine-Tuning
Fine-tuning adapts a model to **specific tasks or domains**.  
It improves relevance and accuracy.

### Instruction Fine-Tuning
Instruction fine-tuning teaches models to **follow instructions better**.  
It improves helpfulness and consistency.

### Evaluation
Evaluation measures **quality, relevance, groundedness, and latency**.  
It ensures models perform as expected.

---

## 6. Parameter-Efficient Fine-Tuning (PEFT)

### PEFT
PEFT fine-tunes **only a small number of parameters**.  
It is faster and cheaper than full fine-tuning.

### LoRA
LoRA adds **small adapter layers** to a model.  
The base model remains unchanged.

### QLoRA
QLoRA combines **quantization with LoRA**.  
It enables fine-tuning on limited hardware.

---

## 7. Reinforcement Learning from Human Feedback (RLHF)

### RLHF
RLHF aligns model behavior using **human feedback**.  
It improves safety and usefulness.

### Reward Model
A reward model scores responses based on quality.  
It guides the main model during training.

### PPO
PPO is an algorithm used to **update models safely**.  
It prevents unstable learning.

---

## 8. Model Deployment Optimizations

### Pruning
Pruning removes **unnecessary weights** from models.  
It reduces model size.

### Distillation
Distillation trains a **smaller model from a larger one**.  
It lowers inference cost.

### A/B Testing
A/B testing compares **two model versions** in production.  
It validates improvements.

---

## 9. Retrieval-Augmented Generation (RAG) and Agents

### Hallucination
Hallucination occurs when a model **generates incorrect facts**.  
RAG helps reduce hallucinations.

### Retrieval-Augmented Generation (RAG)
RAG combines **document retrieval with generation**.  
Models answer using external knowledge.

### Chunking
Chunking splits documents into **smaller pieces**.  
It improves retrieval accuracy.

### Agents
Agents are LLMs that **decide actions and use tools**.  
They enable dynamic workflows.

---

## 10. Multimodal Models

### Multimodal Models
Multimodal models handle **text, images, and vision tasks**.  
They enable image understanding and generation.

### Diffusion Models
Diffusion models generate images by **gradually removing noise**.  
They power Stable Diffusion.

### Stable Diffusion
Stable Diffusion is a **text-to-image generation model**.  
It creates images from natural language prompts.

---

## 11. Controlled Image Generation

### ControlNet
ControlNet provides **fine-grained control** over image generation.  
It uses edges, depth, or pose information.

### DreamBooth
DreamBooth fine-tunes models for **specific subjects or styles**.  
It personalizes image generation.

---

## 12. Amazon Bedrock

### Amazon Bedrock
Amazon Bedrock is a **managed GenAI service on AWS**.  
It provides access to multiple foundation models.

### Bedrock APIs
Bedrock APIs allow **secure model invocation**.  
No infrastructure management is required.

### Governance and Security
Bedrock supports **IAM, logging, encryption, and monitoring**.  
It is designed for enterprise environments.

---

✅ End of Definitions
