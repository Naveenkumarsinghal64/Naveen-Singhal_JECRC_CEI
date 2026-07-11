# 🧠 Single Agent Systems & Agent Pipelines

## 📌 Project Overview

This project implements a **Single-Agent Smart Assistant** that understands user queries, routes tasks based on user intent, uses the appropriate tools, and returns structured JSON responses.

The assistant demonstrates the core concepts of **Single Agent Systems**, including conditional routing, tool integration, and basic error handling.

---

## 🚀 Features

- Understands user queries
- Conditional routing based on intent
- Calculator Tool integration
- Keyword Extraction Tool integration
- General response handling
- Structured JSON output
- Basic error handling

---

## 🛠️ Tools Used

### 1. Calculator Tool
Evaluates mathematical expressions using Python.

**Example:**
```
Input:
Calculate 20 + 5

Output:
25
```

---

### 2. Keyword Extraction Tool

Extracts up to five unique keywords from a given sentence.

**Example:**
```
Input:
Extract keywords from Artificial Intelligence is transforming industries

Output:
["artificial", "intelligence", "transforming", "industries"]
```

---

## 🤖 Agent Routing Logic

The agent uses simple rule-based conditional routing:

- If the query contains **"calculate"** → Calculator Tool
- If the query contains **"keywords"** → Keyword Extraction Tool
- Otherwise → General Response

---

## 📦 Output Format

```json
{
  "type": "calculation | keywords | general | error",
  "result": "..."
}
```

---

## 🧪 Sample Test Cases

### Calculation

```
Input:
Calculate 20 + 5

Output:
{
  "type": "calculation",
  "result": "25"
}
```

### Keyword Extraction

```
Input:
Extract keywords from Artificial Intelligence is transforming industries

Output:
{
  "type": "keywords",
  "result": ["artificial", "intelligence", "transforming", "industries"]
}
```

### General Query

```
Input:
What is Machine Learning?

Output:
{
  "type": "general",
  "result": "This is a general response."
}
```

---

## 📂 Project Structure

```
Week-8-Single-Agent-Systems/
│── Single_Agent_Pipeline.ipynb
│── single_agent_pipeline.py
│── Single_Agent_Quiz.pdf
│── README.md
```

---

## 💻 Technologies Used

- Python 3
- Jupyter Notebook
- VS Code

---

## 📚 Concepts Covered

- Single Agent Systems
- Agent Pipelines
- Conditional Routing
- Tool Integration
- JSON Output
- Error Handling

---

## 👨‍💻 Author

**Naveen Singhal**

