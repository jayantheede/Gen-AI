# LLM'S in Gen AI 

**LLM** in **Generative AI** stands for **Large Language Model**.

**what is an LLM?**
> An LLM is an AI model that understands and generates human-like text using deep learning.
>
**How LLMs Work ?**
LLMs are built using **Transformer neural networks**.

They work by:

1. Reading your input (prompt)
    
2. Understanding the context and intent
    
3. Predicting the **next best word** repeatedly
    
4. Producing meaningful sentences, paragraphs, or conversations

# Tokens

**what is Token in Gen  Ai ?**
  A token is a piece of text (word, part of a word, number, or symbol) used by a Generative AI model to process language.

## What Counts as a Token?

A token can be:

- A **whole word** → `doctor`
    
- A **part of a word** → `appoint` + `ment`
    
- A **number** → `2026`
    
- A **symbol** → `@`, `#`, `!`
    
- Even **spaces or punctuation** (in some models)


**Examples of Tokens?**

Sentence:

> **"Generative AI is powerful"**

Possible tokenization:

`Generative | AI | is | powerful`

→ **4 tokens**

# Transformers

**What Is Transformers in Gen AI ?**
  A Transformer is a deep learning model that processes entire sequences of data at once using attention, rather than reading word-by-word.

## Why Transformers Are Important

Before transformers:

- Models read text **sequentially** (slow and poor at long context)
    
- Hard to remember long sentences
    

With transformers:

- Models read **all words at the same time**
    
- Understand **long-range relationships**
    
- Scale to massive datasets
  
	
#  What Is Parallelism in Transformers?

**Parallelism** in Transformers means the model can **process all tokens (words) at the same time**, instead of one-by-one.

> This is one of the biggest advantages of Transformers in Generative AI.
> 
# Simple Definition

> Parallelism is the ability of a Transformer to handle an entire sentence simultaneously rather than sequentially.

## Why Parallelism Matters

Before Transformers (RNNs / LSTMs):

- Text was processed **word after word**
    
- Slow training
    
- Hard to scale
    
- Poor long-context understanding
    

With Transformers:

- All tokens are processed **in parallel**
    
- Much faster training
    
- Better context understanding
    
- Efficient use of GPUs/TPUs

## Example (Simple)

Sentence:

> **“The patient booked an appointment today.”**

### Sequential Model (Old):

`The → patient → booked → an → appointment → today`

One word at a time

### Transformer (Parallel):

`The | patient | booked | an | appointment | today`

All words at once



## How Parallelism Works in Transformers

### 1️⃣ Tokenization

Sentence is split into tokens.

### 2️⃣ Embeddings

All tokens are converted into vectors **simultaneously**.

### 3️⃣ Self-Attention (Parallel)

Each token:

- Looks at **every other token at the same time**
    
- Computes relationships in parallel
    

This is possible because:

- No dependency on previous token outputs
    
- Matrix operations run efficiently on GPUs

