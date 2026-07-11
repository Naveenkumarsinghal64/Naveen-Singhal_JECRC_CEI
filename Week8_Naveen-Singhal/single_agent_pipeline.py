"""# 🧠 Single Agent Pipeline Project

## Problem Statement
Build a **Single-Agent Smart Assistant** that:
- Understands user queries
- Routes tasks based on intent
- Uses tools when required
- Returns structured JSON output

### The agent should handle:
- Math queries → Calculator Tool
- Keyword extraction → Keyword Tool
- General queries → Direct response

---
### 🛠️ What You Need to Implement
- Agent logic
- Conditional routing
- Tool integration
- Basic error handling

### 🚀 Bonus
- Improve routing
- Add logging
- Add more tools
"""

# 🛠️ TOOL 1: Calculator

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Error in calculation"

# 🛠️ TOOL 2: Keyword Extractor

def extract_keywords(text: str) -> list:
    """Extract keywords from text."""
    try:
        words = text.split()
        keywords = list(set([w.lower() for w in words if len(w) > 4]))
        return keywords[:5]
    except Exception:
        return []

"""## 🤖 Implement Agent Logic Below

👉 Use conditional routing:
- If query contains "calculate" → use calculator
- If query contains "keywords" → use keyword extractor
- Else → general response
"""

# 🤖 AGENT FUNCTION

def agent(query: str):
    query_lower = query.lower()

    try:

        # Calculator Routing
        if "calculate" in query_lower:

            expression = query_lower.replace("calculate", "").strip()

            result = calculator(expression)

            return {
                "type": "calculation",
                "result": result
            }

        # Keyword Routing
        elif "keywords" in query_lower:

            text = query.replace("Extract keywords from", "")
            text = text.replace("keywords", "").strip()

            result = extract_keywords(text)

            return {
                "type": "keywords",
                "result": result
            }

        # General Routing
        else:

            return {
                "type": "general",
                "result": "This is a general response."
            }

    except Exception as e:

        return {
            "type": "error",
            "result": str(e)
        }

"""## 📦 Expected Output Format

```
{
  "type": "calculation / keywords / general / error",
  "result": ...
}
```
"""

# 🧪 Test Cases

queries = [
    "Calculate 20 + 5",
    "Extract keywords from Artificial Intelligence is transforming industries",
    "What is machine learning?"
]

for q in queries:
    print("Query:", q)
    print("Response:", agent(q))
    print("-" * 50)

# 🎯 Interactive Mode

while True:
    user_input = input("Enter query (type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    print("Response:", agent(user_input))