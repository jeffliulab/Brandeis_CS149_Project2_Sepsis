# AI Agent with Sepsis Analysis (ML Final Project)


## How Sepsis Analysis Tool integrated in AI Agent?

The Sepsis Analysis tool and the open-manus tool are in parallel relationship.

```vbnet

1. User sends a message to the AI assistant
   ↓
2. LLM analyzes the user’s prompt and decides **which toolset** to use,
   then emits a **hidden directive** in the form [[TOOLS:…]]:
     • FALSE   → no tools needed  
     • TRUE    → use the OPEN_MANUS toolset  
     • SEPSIS  → use the SEPSIS pipeline toolset  
   (Note: this is choosing a toolset, not individual commands.)
   ↓
3. Branch on the chosen toolset:
      ┌────────────────────────────┐
      │ Toolset = FALSE            │
      │  → skip all tools, go to 6 │
      └────────────────────────────┘
      ┌────────────────────────────┐
      │ Toolset = TRUE             │
      │  → invoke OPEN_MANUS       │
      └────────────────────────────┘
      ┌────────────────────────────┐
      │ Toolset = SEPSIS           │
      │  → invoke SEPSIS TOOL SETS │
      │    subprocess pipeline     │
      └────────────────────────────┘
   ↓
4. Tool execution & logging:
     • OPEN_MANUS → call `Manus.run(content)`  
     • SEPSIS     → run scripts 1_load … 7_predict via `subprocess`  
     • Capture each step’s stdout/stderr as **console logs**  
   ↓
5. Automated summary via LLM:
     • Build a “summary prompt” containing  
       – the **original user request**  
       – the **console logs** from step 4  
       – instructions to summarize:  
         1. Tools used & why  
         2. Outcomes achieved  
         3. Key insights  
         4. Next steps/recommendations  
     • Stream that prompt to the LLM and collect the **final_summary**
   ↓
6. Return to user:


```


## AgentKit Framework

**AgentKit** is an extensible AI Agent infrastructure framework designed for developers to rapidly build, customize, and deploy tool-augmented agents. It includes example functionalities focused on **financial** and **pharmaceutical** domains to demonstrate real-world applications.

This SEPSIS TOOL is integrated in this framework. It uses the AgentKit V0.14.

---


## 📁 Project Structure

```
AgentKit/
│
├── app/            # Main entry point with Gradio-based frontend interface
│
├── key/            # API key configuration for LLM (e.g., OpenAI or DeepSeek)
│
├── sepsis/            # Sepsis Tools
│
└── open_manus/     # Open-manus Tools
```

---

## 🚀 Quick Start

**1. Clone the Repository**

```bash
git clone {the github repo url}.git
cd this repo root folder
```

**2. Install Requirements**

Recommend using conda to manage pip packages.

```bash
pip install -r requirements.txt
```

**3. Configure Your API Keys**

Create or modify the file `key/key.py` with your API credentials:

```python
MAIN_APP_KEY = "your-openai-or-deepseek-api-key"
MAIN_APP_URL = "https://api.deepseek.com/v1"  # or OpenAI base URL
```

Also modify the file `key/manus_config/config.toml`

**4. Run the Application**

```bash
python3 -m app
```

Recommend using vscode to run, vscode can guide you open the browser, also you can visit [http://localhost:7860](http://localhost:7860) in your browser.

**5. Ask Sepsis Task**

You can ask the agent to finish the sepsis task, or chat with anything you want.

The AI Agent will only do the sepsis task when you ask to.

---

