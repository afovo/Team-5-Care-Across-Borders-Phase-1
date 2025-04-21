# Team-5-Care-Across-Borders-Phase-1
An agentic AI dialogue platform for Care Across Borders' customer discovery phase

## Prerequisites

- Python 3.9+ installed
- An OpenAI API key with access to GPT-4
- Basic knowledge of command line interfaces

## Installation Steps

### 1. Set Up Project Environment

```bash
# Create a new directory for the project
mkdir care-across-borders
cd care-across-borders

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Create project structure
mkdir -p data config
```

### 2. Install Required Packages

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pydantic streamlit
```

### 3. Create Configuration Files

Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

Create a `config/settings.py` file:

```python
import os
from enum import Enum

class Language(str, Enum):
    ENGLISH = "English"
    TWI = "Twi"
    GA = "Ga"

class StakeholderRole(str, Enum):
    CLINIC_MANAGER = "Clinic Manager"
    PHYSICIAN = "Physician"
    HEALTH_WORKER = "Community Health Worker"

# Default settings
DEFAULT_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.7
```

### 4. Set Up Knowledge Base

Create `data/ghana_healthcare_context.txt` with the content provided in the main code.

### 5. Create Main Application Files

1. Copy the `HealthcareAgent` class code to `healthcare_agent.py`
2. Copy the Streamlit UI implementation to `app.py`

### 6. Launch the Application

```bash
streamlit run app.py
```

## Project Overview

This application simulates conversations with healthcare stakeholders in Ghana for Care Across Borders, an early-stage telemedicine venture. The system is designed to support the customer discovery phase by enabling simulated dialogues with clinic managers, physicians, and community health workers.

## Features

- **Multi-language Support**: Seamlessly switch between English, Twi, and Ga
- **Role-based Conversations**: Simulate discussions with different healthcare stakeholders
- **Contextual Understanding**: Adapts to different clinic settings and resource levels
- **Report Generation**: Creates comprehensive reports after conversations
- **Knowledge Base Integration**: Includes information about Ghana's healthcare context

## Technical Architecture

The application is built using:

- **LangChain**: For agent orchestration and tool integration
- **OpenAI's GPT-4**: As the underlying language model
- **FAISS**: For efficient vector storage and retrieval
- **Streamlit**: For the user interface

## System Components

### Healthcare Agent

The core `HealthcareAgent` class manages:
- Language translation
- Conversation memory
- Knowledge retrieval
- Report generation
- Role and context switching

### Tools

The agent uses several specialized tools:
1. **TranslateToEnglish**: Translates Twi or Ga to English
2. **TranslateFromEnglish**: Translates English to Twi or Ga  
3. **GhanaHealthcareInfo**: Retrieves contextual information about Ghana's healthcare system
4. **SwitchLanguage**: Changes the conversation language
5. **GenerateReport**: Creates summary reports from conversations

### Context Management

The system maintains a `ConversationContext` that includes:
- Current language
- Stakeholder role
- Clinic location
- Available resources
- Internet connectivity

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install langchain langchain-openai faiss-cpu streamlit pydantic
   ```
3. Create a file named `ghana_healthcare_context.txt` with the provided knowledge base content
4. Set your OpenAI API key as an environment variable or in Streamlit secrets
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage Guide

### Starting a Conversation

1. Configure the conversation settings in the sidebar:
   - Select the stakeholder role
   - Choose the conversation language
   - Set the clinic location, resources, and connectivity
   - Click "Apply Settings"

2. Begin the conversation by typing in the chat input

3. The agent will:
   - Ask questions about current healthcare practices
   - Explore pain points and challenges
   - Assess openness to digital solutions
   - Adapt to the selected language and role

### Generating Reports

After a sufficient conversation:

1. Click "Generate Report" in the sidebar
2. View the comprehensive report in the Report tab
3. The report includes:
   - Stakeholder profile
   - Current practices assessment
   - Digital readiness evaluation
   - Partnership potential analysis

## Understanding Community Health Workers in Ghana

The system is specifically designed to account for the reality of community health workers:

- **Educational Background**: Typically hold high school diplomas without formal post-secondary education
- **Working Environment**: Operate in remote CHPS (Community-based Health Planning and Services) zones
- **Communication Methods**: Rely primarily on handwritten communication, though most have mobile phones
- **Language Considerations**: May prefer Twi or Ga over English in certain contexts

## Extending the System

The modular design allows for:

- Adding additional languages
- Expanding the knowledge base
- Creating new stakeholder roles
- Developing more specialized tools
- Customizing the report format
