# 🏥 Multi-Agent Medical Report Analysis with Hallucination Control

An intelligent multi-agent AI system that analyzes medical reports using specialized LLM-powered agents (Cardiologist, Psychologist, Pulmonologist) running in parallel, synthesizes findings through a Multidisciplinary Team agent, and incorporates hallucination control mechanisms to ensure clinically grounded outputs.

> ⚠️ **Disclaimer:** This system is for educational and research purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 📌 Table of Contents

- [Overview](#overview)
- [System Architecture](#-system-architecture)
- [End-to-End Pipeline](#-end-to-end-pipeline)
- [Hallucination Control](#-hallucination-control)
- [Agent Interaction Flow](#-agent-interaction-flow)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Sample Output](#-sample-output)
- [Future Roadmap](#-future-roadmap)
- [Contributors](#-contributors)

---

## Overview

Medical reports are often long, unstructured, and manually reviewed — making the process time-consuming and error-prone. Single "black-box" AI systems can provide diagnoses but often lack transparent reasoning and may hallucinate medications, lab values, or patient history.

This project addresses these challenges by:

- Deploying **domain-specific specialist agents** that analyze reports from their area of expertise
- Running agents **concurrently** using Python's `ThreadPoolExecutor` for reduced latency
- Synthesizing specialist findings into a **unified multidisciplinary diagnosis**
- Applying a **5-layer hallucination defense strategy** to ensure output reliability
- Maintaining **human-in-the-loop** safety for clinical oversight

---

## 🏗 System Architecture

```mermaid
graph LR
    A["🖥️ User Interface<br/><br/>• File Upload<br/>• Progress Steps<br/>• Specialist Tabs<br/>• Final Diagnosis"] -->|"Upload Report"| B["⚙️ Flask Backend<br/><br/>• /analyze endpoint<br/>• Orchestrates<br/>  7-step pipeline"]
    B -->|"Trigger Pipeline"| C["🔬 Analysis Pipeline<br/><br/>1. Read Report<br/>2. Intelligent Triage<br/>3. Specialist Agents ⚡<br/>4. Hallucination Check<br/>5. Team Synthesis<br/>6. Cross-Agent Validation<br/>7. Report Generation"]
    C -->|"Return Results"| A

    style A fill:#1e3a5f,stroke:#4a9eff,stroke-width:2px,color:#ffffff
    style B fill:#2d4a2d,stroke:#66bb6a,stroke-width:2px,color:#ffffff
    style C fill:#4a1a5e,stroke:#ce93d8,stroke-width:2px,color:#ffffff
```

---

## 🔄 End-to-End Pipeline

```mermaid
graph TD
    A["📄 Medical Report<br/>PDF / TXT Input"] --> B["📥 Step 1: Report Ingestion<br/>Read & parse input file"]
    B --> C["🔍 Step 2: Intelligent Triage<br/>Determine which specialists are needed"]
    C --> D["⚡ Step 3: Parallel Specialist Execution"]

    D --> E["🫀 Cardiologist Agent"]
    D --> F["🧠 Psychologist Agent"]
    D --> G["🫁 Pulmonologist Agent"]

    E --> H["🛡️ Step 4: Hallucination Detector<br/>Validate evidence, values, meds & confidence"]
    F --> H
    G --> H

    H --> I["👥 Step 5: Multidisciplinary Team Agent<br/>Synthesize specialist findings into unified diagnosis"]
    I --> J["✅ Step 6: Cross-Agent Consistency Validator<br/>Check contradictions & flag inconsistencies"]
    J --> K["📊 Step 7: Report Generator<br/>PDF + JSON + TXT with scores & metadata"]

    style A fill:#0d47a1,stroke:#42a5f5,stroke-width:2px,color:#ffffff
    style B fill:#1565c0,stroke:#42a5f5,stroke-width:2px,color:#ffffff
    style C fill:#1565c0,stroke:#42a5f5,stroke-width:2px,color:#ffffff
    style D fill:#e65100,stroke:#ff9800,stroke-width:2px,color:#ffffff
    style E fill:#b71c1c,stroke:#ef5350,stroke-width:2px,color:#ffffff
    style F fill:#4a148c,stroke:#ce93d8,stroke-width:2px,color:#ffffff
    style G fill:#1b5e20,stroke:#66bb6a,stroke-width:2px,color:#ffffff
    style H fill:#f57f17,stroke:#ffee58,stroke-width:2px,color:#000000
    style I fill:#283593,stroke:#7986cb,stroke-width:2px,color:#ffffff
    style J fill:#2e7d32,stroke:#66bb6a,stroke-width:2px,color:#ffffff
    style K fill:#0d47a1,stroke:#42a5f5,stroke-width:2px,color:#ffffff
```

---

## 🛡 Hallucination Control

The system implements a **5-layer defense strategy** to minimize AI hallucination:

```mermaid
graph TD
    subgraph PREVENTION["🟢 PREVENTION LAYERS"]
        L1["<b>Layer 1: Prompt Engineering</b><br/>• Anti-hallucination rules in prompts<br/>• Force evidence citation<br/>• Explicit confidence levels<br/>• Self-check for assumptions"]
        L2["<b>Layer 2: Temperature Control</b><br/>• temperature=0 for factual output<br/>• Reduces creative/random responses<br/>• Prioritizes deterministic generation"]
    end

    subgraph DETECTION["🟡 DETECTION LAYERS"]
        L3["<b>Layer 3: Output Validation</b><br/>• Evidence grounding ≥30% overlap<br/>• Physiological plausibility check<br/>• Invented meds & test value detection<br/>• Confidence calibration scoring"]
        L4["<b>Layer 4: Cross-Agent Checking</b><br/>• Compare outputs across specialists<br/>• Flag contradictions<br/>• Inconsistent vitals detection<br/>• Risk score calculation"]
    end

    subgraph MITIGATION["🔴 MITIGATION LAYER"]
        L5["<b>Layer 5: Human Review</b><br/>• Very Low 0-20%: Quick review<br/>• Low 20-40%: Standard review<br/>• Medium 40-60%: Careful review<br/>• High 60-80%: Detailed review<br/>• Very High 80-100%: Regenerate"]
    end

    L1 --> L2 --> L3 --> L4 --> L5

    style PREVENTION fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style DETECTION fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#e65100
    style MITIGATION fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c
    style L1 fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px,color:#000000
    style L2 fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px,color:#000000
    style L3 fill:#fff9c4,stroke:#f57f17,stroke-width:1px,color:#000000
    style L4 fill:#fff9c4,stroke:#f57f17,stroke-width:1px,color:#000000
    style L5 fill:#ffcdd2,stroke:#c62828,stroke-width:1px,color:#000000
```

### Hallucination Risk Score Formula

```mermaid
graph LR
    A["Critical Errors<br/>× 0.3 each"] --> C["🎯 Risk Score<br/>min 1.0, errors + warnings"]
    B["Warnings<br/>× 0.05 each"] --> C
    C --> D{"Risk Level"}
    D -->|"< 0.1"| E["🟢 Very Low"]
    D -->|"< 0.3"| F["🔵 Low"]
    D -->|"< 0.5"| G["🟡 Medium"]
    D -->|"< 0.7"| H["🟠 High"]
    D -->|"≥ 0.7"| I["🔴 Very High"]

    style C fill:#1565c0,stroke:#42a5f5,stroke-width:2px,color:#ffffff
    style D fill:#6a1b9a,stroke:#ce93d8,stroke-width:2px,color:#ffffff
    style E fill:#2e7d32,stroke:#66bb6a,stroke-width:1px,color:#ffffff
    style F fill:#1565c0,stroke:#42a5f5,stroke-width:1px,color:#ffffff
    style G fill:#f57f17,stroke:#ffee58,stroke-width:1px,color:#000000
    style H fill:#e65100,stroke:#ff9800,stroke-width:1px,color:#ffffff
    style I fill:#b71c1c,stroke:#ef5350,stroke-width:1px,color:#ffffff
```

---

## 🤝 Agent Interaction Flow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant M as ⚙️ Main.py
    participant C as 🫀 Cardiologist
    participant P as 🧠 Psychologist
    participant L as 🫁 Pulmonologist
    participant T as 👥 MDT Agent

    U->>M: Upload Medical Report
    M->>M: Load & Parse Report

    par Parallel Execution
        M->>C: Analyze cardiac workup
        M->>P: Assess mental health
        M->>L: Evaluate respiratory
    end

    C-->>M: Cardiac findings
    P-->>M: Psychological assessment
    L-->>M: Pulmonary evaluation

    M->>T: Pass all specialist reports
    T->>T: Synthesize & cross-validate
    T-->>M: Unified diagnosis (3 conditions)

    M->>M: Generate final report
    M-->>U: 📊 Final Diagnosis (TXT/PDF/JSON)
```

---

## 📁 Project Structure

```
📦 multi-agent-medical-analysis
├── 🐍 Main.py                    # Orchestrator — runs agents concurrently & generates final diagnosis
├── 🤖 Agents.py                  # Agent classes (Cardiologist, Psychologist, Pulmonologist, MDT)
├── 📂 Medical Reports/
│   └── 📄 Medical Report - Michael Johnson - Panic Attack Disorder.txt
├── 📂 results/
│   └── 📄 final_diagnosis.txt    # Generated diagnosis output
├── 🔑 apikey.env                 # Google Gemini API key (not committed)
├── 📋 requirements.txt           # Python dependencies
├── 🚫 .gitignore
└── 📖 README.md
```

---

## 🛠 Tech Stack

```mermaid
graph LR
    subgraph LLM["🧠 LLM Layer"]
        A["Google Gemini<br/>2.0 Flash"]
    end

    subgraph ORCH["⚙️ Orchestration"]
        B["LangChain"]
        C["LangChain-Google-GenAI"]
        D["PromptTemplate"]
    end

    subgraph EXEC["⚡ Execution"]
        E["ThreadPoolExecutor<br/>Concurrent Futures"]
        F["Flask Backend"]
    end

    subgraph OUTPUT["📊 Output"]
        G["ReportLab PDF"]
        H["JSON"]
        I["TXT"]
    end

    A --> B --> E --> G
    A --> C --> E --> H
    A --> D --> F --> I

    style LLM fill:#1a237e,stroke:#5c6bc0,stroke-width:2px,color:#ffffff
    style ORCH fill:#004d40,stroke:#26a69a,stroke-width:2px,color:#ffffff
    style EXEC fill:#e65100,stroke:#ff9800,stroke-width:2px,color:#ffffff
    style OUTPUT fill:#4a148c,stroke:#ce93d8,stroke-width:2px,color:#ffffff
```

| Category | Technology |
|----------|------------|
| **LLM** | Google Gemini 2.0 Flash |
| **Orchestration** | LangChain, LangChain-Google-GenAI |
| **Concurrency** | Python `concurrent.futures.ThreadPoolExecutor` |
| **Prompt Management** | LangChain `PromptTemplate` |
| **Backend** | Flask (for web interface variant) |
| **Report Generation** | ReportLab (PDF), JSON |
| **Environment Management** | python-dotenv |
| **Language** | Python 3.10+ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/multi-agent-medical-analysis.git
   cd multi-agent-medical-analysis
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**

   Create an `apikey.env` file in the project root:

   ```env
   GOOGLE_API_KEY = "your-google-gemini-api-key"
   ```

### Run the Analysis

```bash
python Main.py
```

The final diagnosis will be saved to `results/final_diagnosis.txt`.

---

## 📋 Sample Output

**Input:** Medical report for a 29-year-old male presenting with chest pain, palpitations, shortness of breath, dizziness, and sweating.

**Final Diagnosis (synthesized from 3 specialist agents):**

```mermaid
graph TD
    subgraph DX1["🔴 Diagnosis 1"]
        A["<b>Panic Disorder with Agoraphobia</b><br/><br/>Intense chest pain, palpitations, and<br/>impending doom coupled with anxiety<br/>history and high-stress occupation<br/>strongly suggest panic attacks"]
    end

    subgraph DX2["🟠 Diagnosis 2"]
        B["<b>Microvascular Angina with<br/>Possible Vasospastic Component</b><br/><br/>Chest pain despite normal coronary<br/>arteries suggests microvascular<br/>dysfunction warranting further<br/>cardiac investigation"]
    end

    subgraph DX3["🟡 Diagnosis 3"]
        C["<b>GERD-Related Respiratory<br/>Exacerbation</b><br/><br/>Despite PPI management, acid reflux<br/>may contribute to or mimic respiratory<br/>symptoms, especially in the<br/>context of anxiety"]
    end

    style DX1 fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000000
    style DX2 fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000
    style DX3 fill:#fffde7,stroke:#f57f17,stroke-width:2px,color:#000000
```

---

## 🗺 Future Roadmap

```mermaid
timeline
    title Development Roadmap
    section Short-Term
        Patient-friendly explanation agent
        : Treatment timeline generator
        : Second-opinion mode
    section Medium-Term
        Patient history integration
        : Drug interaction checking
        : Research literature agent
    section Long-Term
        Predictive risk analytics
        : Emergency triage levels
        : Multi-language support & voice interface
```

### 🔒 Production Readiness Considerations

```mermaid
graph TD
    subgraph SECURITY["🔒 Security & Compliance Roadmap"]
        A["🔐 Encrypt Files at Rest<br/>AES-256"] --> B["🌐 HTTPS/TLS<br/>Everywhere"]
        B --> C["🗑️ Automatic Deletion<br/>Retention Policies"]
        C --> D["👤 User Authentication<br/>& Role-Based Access"]
        D --> E["🕵️ Anonymization<br/>De-identify PHI"]
        E --> F["🏥 HIPAA-Ready Setup<br/>BAA & Audit Logs"]
    end

    style SECURITY fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style A fill:#e8eaf6,stroke:#3949ab,stroke-width:1px,color:#000000
    style B fill:#e8eaf6,stroke:#3949ab,stroke-width:1px,color:#000000
    style C fill:#e8eaf6,stroke:#3949ab,stroke-width:1px,color:#000000
    style D fill:#e8eaf6,stroke:#3949ab,stroke-width:1px,color:#000000
    style E fill:#e8eaf6,stroke:#3949ab,stroke-width:1px,color:#000000
    style F fill:#e8eaf6,stroke:#3949ab,stroke-width:1px,color:#000000
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
