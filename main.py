# Healthcare Discovery Agent - LangChain Implementation
# For Care Across Borders Telemedicine Venture

import os
import json
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from io import StringIO
# Define the available languages

class Language(str, Enum):
    ENGLISH = "English"
    TWI = "Twi"
    GA = "Ga"

# Define the healthcare stakeholder roles
class StakeholderRole(str, Enum):
    CLINIC_MANAGER = "Clinic Manager"
    PHYSICIAN = "Physician"
    HEALTH_WORKER = "Community Health Worker"
    NURSE = "Nurse"
    DISTRICT_HEALTH_OFFICER = "District Health Officer"

# Define a model for the conversation context
class ConversationContext(BaseModel):
    language: Language = Field(default=Language.ENGLISH)
    role: StakeholderRole = Field(default=StakeholderRole.CLINIC_MANAGER)
    clinic_location: str = Field(default="Rural community near Accra")
    clinic_resources: str = Field(default="Limited")
    internet_connectivity: str = Field(default="Intermittent")

# Function to create the knowledge base for Ghana healthcare context
def create_knowledge_base(openai_api_key):
    # Sample text with information about Ghana's healthcare system
    with open("ghana_healthcare_context.txt", "w") as f:
        f.write("""
# Ghana Healthcare Context

## Community Health Workers
1. Typically members of the community who have received basic first-aid and referral training.
2. Generally hold a high school diploma, without formal post-secondary education.
3. Operate in remote CHPS (Community-based Health Planning and Services) zones.
4. Rely on handwritten communication, though most have mobile phones (some smartphones).

## Languages in Ghana (Pilot Region)
- Twi, Ga, and English are the main languages used.

## Healthcare System Structure
- Ghana operates a tiered healthcare system with tertiary hospitals in major cities.
- District hospitals serve as referral points for smaller clinics.
- Community-based Health Planning and Services (CHPS) compounds serve rural areas.
- Many rural clinics lack reliable electricity and internet connectivity.

## Common Challenges
- Limited record-keeping systems, often paper-based.
- Difficulty tracking patient follow-ups.
- Communication barriers between different levels of care.
- Limited specialist access in rural areas.
- Transportation difficulties for patients requiring referrals.

## Skin Conditions
- Fungal infections (like ringworm/tinea capitis) are common, especially in humid conditions.
- Scabies outbreaks occur, particularly in schools.
- Buruli ulcer is endemic in some areas.
- Other common conditions include impetigo, burns from domestic accidents, and diabetic ulcers.
        """)
    
    # Create interview transcript files
    create_interview_transcripts()
    
    # Load and process the knowledge base
    loader = TextLoader("ghana_healthcare_context.txt")
    docs = loader.load()
    
    # Load interview transcripts
    nurse_loader = TextLoader("nurse_interview.txt")
    dhmt_loader = TextLoader("dhmt_interview.txt")
    chw_loader = TextLoader("chw_interview.txt")
    
    nurse_docs = nurse_loader.load()
    dhmt_docs = dhmt_loader.load()
    chw_docs = chw_loader.load()
    
    # Combine all documents
    all_docs = docs + nurse_docs + dhmt_docs + chw_docs
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = vectorstore.as_retriever()
    
    return retriever

def create_interview_transcripts():
    """Create text files with interview transcripts for knowledge base"""
    
    # Nurse interview transcript
    with open("nurse_interview.txt", "w") as f:
        f.write("""
# Interview with Nurse in Health Center, Eastern Region
Location: Health Center, Eastern Region
Language: Conducted in Twi, translated into English
Respondent Profile: Female, late 30s, Registered Nurse, 10+ years in service

## Daily Work
- Starts at 7:30am, patients line up by 8am
- OPD runs until 2pm
- Sees 40-50 clients daily, more on market days
- Common cases: children, pregnant women, minor injuries, skin issues (especially during humid/rainy season)

## Skin Condition Management
- Asks basic questions: duration, medications used, other affected persons
- Examines lesions personally
- Manages clear cases (fungal, scabies) at the facility
- Refers unusual cases, ulcers, severe infections, necrotic/cancerous appearances
- Does not guess on complicated cases

## Confidence Level
- Confident within limits
- Limited dermatology training
- Familiar with fungal infections, bacterial infections, basic dermatitis
- Refers mixed infections or rare diseases
- Balances patient care with scope of practice

## Second Opinions
- Calls senior colleagues or district hospital doctors when unsure
- Sometimes sends pictures via WhatsApp
- Responses may be delayed due to colleagues' busy schedules

## CHW Referrals
- Takes them seriously but re-examines everything
- Notes that some CHWs refer for minor issues while others wait too long

## App Requirements
- Must be offline-first due to unreliable network
- Clear red flag alerts for necrosis or deep infections
- Photo timeline to track healing
- Easy note-taking interface, not extensive forms
- Treatment recommendations must be based on GHS Standard Treatment Guidelines
- Prioritizes human clinical judgment over app recommendations

## Concerns
- App crashing
- Taking too long during OPD
- Providing incorrect information
- Lack of follow-up support after launch

## Technical Context
- Phone network available 60-70% of the time, unreliable in afternoons
- Phone charging only available at home, not at clinic
- Patients require clear consent process for photos and privacy protection

## Motivation
- Desires formal recognition: training points, certificates, consideration during appraisals
        """)
    
    # District Health Officer interview transcript
    with open("dhmt_interview.txt", "w") as f:
        f.write("""
# Interview with District Health Officer, Volta Region
Location: District Health Directorate, Volta Region
Language: Conducted in English (with occasional Twi phrases)
Respondent Profile: Male, early 40s, Deputy District Director of Public Health, 12 years' experience

## Role Description
- Oversees public health programs (disease surveillance, maternal/child health, school health, community outreach)
- Supervises CHOs, health centers, and works with facility in-charges to implement MOH and GHS programs

## Skin Conditions Context
- Very common, especially fungal infections, scabies outbreaks in schools, burns from domestic accidents
- Often underreported as system prioritizes malaria, TB, maternal deaths
- Impacts productivity but receives little programmatic attention

## Current Handling Protocol
- Frontline workers refer serious cases to next level (health center nurse or district hospital)
- CHWs not licensed to diagnose
- Nurses manage simple infections but should escalate complex cases
- Documentation expected but informal referrals (verbal, WhatsApp) are common
- Frontline workers lack confidence in dermatology due to limited training

## App Requirements
- Must be simple, fast, and not create extra reporting burden
- Offline mode essential
- Triage risk assessment to guide referrals and reduce unnecessary overload
- Case storage with later syncing for rural zones
- Linkage to nurse supervisors for accountability
- Dashboard access for DHMT to monitor cases and plan interventions

## Priority Conditions
- Tinea capitis (ringworm of the scalp)
- Scabies
- Impetigo
- Buruli ulcer (endemic in some areas)
- Diabetic ulcers (for referral)
- Burns (small domestic injuries)

## Scaling Considerations
- Must embed into training packages
- Align with GHS systems
- Improve referrals
- Reduce unnecessary patient transport
- Increase early detection

## Concerns
- Data ownership: Ministry and GHS must retain control
- Sustainability: Maintenance after project ends
- Duplication: Must not replace clinical judgment or overload staff
        """)
    
    # CHW interview transcript
    with open("chw_interview.txt", "w") as f:
        f.write("""
# Interview with Community Health Worker, Central Region
Location: CHPS compound, Central Region
Language: Originally in Twi, translated into English
Respondent Profile: Woman, early 30s, high school graduate, 5 years' experience

## Daily Work
- Wakes at 5:30am, begins visits by 7am
- Visits 5-10 homes depending on distance
- Checks on pregnant women, newborns, follows up on fever or skin rash cases

## Skin Issue Management
- Advises clinic visits for minor rashes or ringworm
- Calls supervisor for serious cases (sores, swelling)
- Sometimes takes pictures via WhatsApp for consultation
- Not supposed to diagnose but community members often expect this
- Refers cases with bleeding, infection, or unfamiliar presentations
- Follows up if condition not improving after two visits

## Record Keeping
- Uses paper register submitted monthly to health center
- Uses Child Welfare Card for mothers and babies
- Sometimes keeps personal notes

## Technology Use
- Uses phone for photos, calling nurses, and WhatsApp
- Network poor in some areas
- Battery life is a limitation during all-day movement

## App Requirements
- Must be very simple to use
- Clear indication of whether to refer immediately or monitor at home
- Photo storage until network available
- Note-taking capability for contextual information (e.g., prior herbal treatment)
- Voice capabilities for faster comprehension
- Visual examples of conditions
- No prescription functionality (outside CHW scope)

## Community Acceptance
- Would accept phone photos if purpose is clearly explained
- Privacy and confidentiality must be guaranteed (no sharing)
- Community would trust if explanation is clear and GHS-approved

## Language
- Primarily uses Twi with patients
- Some speak Fante, but Twi is widely understood
- English only used for reports

## Adoption Barriers
- Would not use without supervisor approval or GHS endorsement
- Difficult user experience
- Battery limitations
- Constant internet requirement
- Community trust issues

## Technical Context
- Network available roughly 50% of the time (stronger in mornings)
- Charging only available at home, no power banks
- Phone battery management is a daily challenge

## Motivation
- Desires certificate or small allowance
- Would appreciate recognition at DHMT meetings
        """)

def extract_example_dialogues():
    """Create structured example dialogues based on interview transcripts"""
    
    # These would be formatted dialogues based on the interviews
    # that show authentic interaction patterns for each stakeholder role
    
    example_dialogues = {
        "NURSE": [
            {"role": "human", "content": "Can you describe a normal working day for you?"},
            {"role": "assistant", "content": "I start around 7:30. By 8 o'clock, patients have already lined up. OPD runs till about 2pm. I usually see 40 to 50 clients a day, more if it's market day. Mostly children, pregnant women, minor injuries, and plenty of skin issues, especially when it's humid or raining."},
            {"role": "human", "content": "When someone comes with a skin condition, how do you usually manage it?"},
            {"role": "assistant", "content": "I ask the basic questions: how long, any medication used, any other person affected. Then I check the lesion myself. If it's clear, fungal, scabies, I manage at the facility. But if it's unusual, ulcers, severe infections, anything looking necrotic or cancerous, I refer. I don't guess on complicated cases. It's not worth the risk."}
        ],
        "DISTRICT_HEALTH_OFFICER": [
            {"role": "human", "content": "Can you briefly tell me about your role in the district health system?"},
            {"role": "assistant", "content": "I oversee public health programs, including disease surveillance, maternal and child health, school health, and community outreach. I supervise the CHOs, the health centers, and work with facility in-charges to implement MOH and GHS programs."},
            {"role": "human", "content": "In your experience, how common are skin-related health issues at the district level?"},
            {"role": "assistant", "content": "Very common. Especially fungal infections, scabies outbreaks in schools, some burns from domestic accidents. We see dermatological conditions, but they are often underreported because the system prioritizes malaria, TB, maternal deaths. Skin disease is quietly draining productivity, but it doesn't get much programmatic attention."}
        ],
        "HEALTH_WORKER": [
            {"role": "human", "content": "Can you tell me about a normal day in your work as a community health worker?"},
            {"role": "assistant", "content": "Hmm, a normal day… usually I wake up early, around 5:30, and by 7 I am moving to the houses. I visit maybe 5 to 10 homes depending on the distance. I check on the pregnant women, newborns, sometimes follow up on someone with fever or skin rash."},
            {"role": "human", "content": "What do you do when you see a skin issue?"},
            {"role": "assistant", "content": "It depends. If it's just small rash or ringworm, I advise them to go to the clinic. Some will go. Some will ask for cream. If it looks bad, like sores, or swollen, I call my supervisor. Sometimes I take a picture and send it to her on WhatsApp."}
        ]
    }
    
    # Save to JSON file
    with open("example_dialogues.json", "w") as f:
        json.dump(example_dialogues, f, indent=2)

# Main Healthcare Agent Class
class HealthcareAgent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, api_key=openai_api_key)
        self.context = ConversationContext()
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.retriever = create_knowledge_base(openai_api_key)
        self.example_dialogues = self._load_example_dialogues()
        self.agent_executor = self._create_agent()
    
    def _load_example_dialogues(self):
        """Load example dialogues or create them if they don't exist"""
        try:
            with open("example_dialogues.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Create and then load example dialogues
            extract_example_dialogues()
            with open("example_dialogues.json", "r") as f:
                return json.load(f)
        
    def _create_agent(self):
        # Create language translation tools
        translate_to_english_tool = Tool(
            name="TranslateToEnglish",
            func=self._translate_to_english,
            description="Translates text from Twi or Ga to English"
        )
        
        translate_from_english_tool = Tool(
            name="TranslateFromEnglish",
            func=self._translate_from_english,
            description="Translates text from English to Twi or Ga"
        )
        
        # Create a retrieval tool for healthcare context
        retrieval_tool = Tool(
            name="GhanaHealthcareInfo",
            func=self._query_healthcare_context,
            description="Provides information about Ghana's healthcare system, community health workers, and local context"
        )
        
        # Create a tool to switch languages
        switch_language_tool = Tool(
            name="SwitchLanguage",
            func=self._switch_language,
            description="Changes the conversation language between English, Twi, and Ga"
        )
        
        # Create a tool to generate summary report
        generate_report_tool = Tool(
            name="GenerateReport",
            func=self._generate_report,
            description="Generates a summary report of the conversation insights"
        )
        
        # Create a tool to retrieve authentic dialogue examples
        example_tool = Tool(
            name="GetAuthenticDialogueExamples",
            func=self._get_dialogue_examples,
            description="Retrieves authentic dialogue examples from real healthcare workers in Ghana"
        )

        # Define the agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._generate_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        agent = create_openai_tools_agent(
            self.llm,
            [translate_to_english_tool, translate_from_english_tool, 
             retrieval_tool, switch_language_tool, generate_report_tool,
             example_tool],
            prompt
        )
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=[translate_to_english_tool, translate_from_english_tool, 
                   retrieval_tool, switch_language_tool, generate_report_tool,
                   example_tool],
            verbose=True,
            memory=self.memory
        )
        
    def _get_dialogue_examples(self, role_key: str = None) -> str:
        """Retrieve authentic dialogue examples based on stakeholder role"""
        if not role_key:
            role_key = self.context.role.upper()
        else:
            role_key = role_key.upper()
            
        # Map from StakeholderRole enum values to example dialogue keys
        role_mapping = {
            "CLINIC_MANAGER": "DISTRICT_HEALTH_OFFICER",  # Using DHO examples for clinic managers
            "PHYSICIAN": "NURSE",  # Using nurse examples for physicians
            "HEALTH_WORKER": "HEALTH_WORKER",
            "NURSE": "NURSE",
            "DISTRICT_HEALTH_OFFICER": "DISTRICT_HEALTH_OFFICER"
        }
        
        # Get the correct key for the example dialogues
        mapped_role = role_mapping.get(role_key, "HEALTH_WORKER")
        
        # Get examples for the role
        examples = self.example_dialogues.get(mapped_role, [])
        
        if not examples:
            return "No examples available for this role."
            
        # Format the examples as a conversational transcript
        formatted_examples = "Authentic dialogue examples:\n\n"
        for message in examples:
            speaker = "User" if message["role"] == "human" else "Healthcare Worker"
            formatted_examples += f"{speaker}: {message['content']}\n\n"
            
        return formatted_examples
    
    def _generate_system_prompt(self) -> str:
        role_descriptions = {
            StakeholderRole.CLINIC_MANAGER: "The clinic manager oversees operations, makes administrative decisions, handles finances, and coordinates staff. They're concerned with efficiency, resource allocation, and overall clinic performance.",
            StakeholderRole.PHYSICIAN: "The physician provides direct patient care, makes diagnoses, prescribes treatments, and manages patient follow-ups. They're concerned with patient outcomes, treatment options, and clinical resources.",
            StakeholderRole.HEALTH_WORKER: "The community health worker provides basic care, conducts health education, and refers patients to clinics when needed. They typically have a high school diploma, operate in remote areas, and use handwritten communication systems.",
            StakeholderRole.NURSE: "The nurse provides direct patient care in health centers, manages diagnoses for common conditions, and refers complex cases. They see 40-50 patients daily, work from around 7:30am to 2pm, and manage a variety of conditions including skin issues.",
            StakeholderRole.DISTRICT_HEALTH_OFFICER: "The district health officer oversees public health programs including disease surveillance, maternal and child health, school health, and community outreach. They supervise health centers and CHWs, and implement Ministry of Health and Ghana Health Service programs."
        }
        
        language_examples = {
            Language.ENGLISH: "",
            Language.TWI: "Examples of expressions in Twi: 'Ɛte sɛn?' (How are you?), 'Me din de...' (My name is...), 'Yɛfrɛ me...' (I am called...), 'Medaase' (Thank you), 'Aane' (Yes), 'Daabi' (No), 'Mepa wo kyɛw' (Please)",
            Language.GA: "Examples of expressions in Ga: 'Tɛ wɔ ŋɔ?' (How are you?), 'Miiŋwɛmi ji...' (My name is...), 'Tsɔɔmɔ mi...' (Show me...), 'Oyiwaladon' (Thank you), 'Hɛɛ' (Yes), 'Daabi' (No), 'Ofaine' (Please)"
        }
        
        # Get role-specific conversation style guidance based on transcripts
        conversation_style = {
            StakeholderRole.HEALTH_WORKER: """
    - Use simpler language and shorter sentences
    - Occasionally use expressions like "hmm" and "ooo" for authenticity
    - Be more tentative in assertions - "Maybe..." "It depends..."
    - Reference supervisors and authority figures often
    - Express practical concerns about battery life, network issues
    - Show awareness of community perceptions and trust issues
    - Emphasize visual learning and voice instructions over text
    - Mention time constraints and home visit logistics
            """,
            StakeholderRole.NURSE: """
    - Be direct and practical in responses
    - Emphasize clinical assessment process - asking questions, examining patients
    - Express confidence within defined scope of practice
    - Reference risk assessment - "I don't guess on complicated cases"
    - Mention high patient volumes (40-50 daily)
    - Show concern for resource limitations and time constraints
    - Emphasize alignment with Ghana Health Service guidelines
    - Reference WhatsApp communications for second opinions
            """,
            StakeholderRole.DISTRICT_HEALTH_OFFICER: """
    - Speak with authority about public health programs and district-level concerns
    - Reference system-level perspectives and programmatic priorities
    - Mention oversight responsibilities for CHWs and health centers
    - Emphasize data collection, reporting, and monitoring
    - Reference Ghana Health Service (GHS) and Ministry of Health (MOH) guidelines
    - Discuss resource allocation challenges across facilities
    - Mention supervision visits and performance metrics
            """,
            StakeholderRole.CLINIC_MANAGER: """
    - Focus on operational concerns like staffing, supplies, and facility management
    - Reference budget constraints and resource allocation decisions
    - Discuss coordination between different healthcare workers
    - Mention record-keeping systems and administrative processes
    - Show concern for clinic efficiency and patient throughput
    - Reference reporting requirements to district health offices
            """,
            StakeholderRole.PHYSICIAN: """
    - Emphasize clinical expertise and diagnostic process
    - Reference medical terminology appropriate for Ghanaian context
    - Show concern for patient outcomes and follow-up
    - Mention limitations in available treatments or diagnostics
    - Reference referral processes for complex cases
    - Discuss collaboration with nurses and community health workers
            """
        }
        
        # Get the appropriate style guide for the current role
        style_guide = conversation_style.get(self.context.role, "")
        
        # Construct the system prompt
        system_prompt = f"""
    You are a simulation of a {self.context.role.value} working in a healthcare setting in Ghana.

    ROLE DESCRIPTION:
    {role_descriptions[self.context.role]}

    SETTING:
    - Location: {self.context.clinic_location}
    - Resources: {self.context.clinic_resources}
    - Internet connectivity: {self.context.internet_connectivity}

    LANGUAGE:
    You will communicate in {self.context.language.value}.
    {language_examples[self.context.language]}

    CONVERSATION STYLE:
    {style_guide}

    FOCUS ON SKIN CONDITIONS:
    You should emphasize experience with common skin conditions in Ghana, including fungal infections, scabies, impetigo, Buruli ulcer, diabetic ulcers, and burns.

    CONTEXT AWARENESS:
    - Respond in a way that reflects the realities of healthcare in Ghana
    - Consider resource limitations and technological constraints
    - Be authentic to the communication patterns of your role
    - Remember that community health workers are not licensed to diagnose
    - Reference Ghana Health Service Standard Treatment Guidelines when appropriate

    YOUR GOAL:
    Help the user understand the perspective, needs, and challenges of a {self.context.role.value} in Ghana, particularly related to diagnosing and managing skin conditions.
    """
        
        return system_prompt
    def _translate_to_english(self, text: str) -> str:
        """Translate from Twi or Ga to English"""
        if self.context.language == Language.ENGLISH:
            return text
        
        # Using the LLM for translation
        translation_prompt = f"Translate this {self.context.language} text to English: {text}"
        result = self.llm.invoke(translation_prompt)
        return result.content

    def _translate_from_english(self, text: str) -> str:
        """Translate from English to Twi or Ga"""
        if self.context.language == Language.ENGLISH:
            return text
        
        # Using the LLM for translation
        translation_prompt = f"Translate this English text to {self.context.language}: {text}"
        result = self.llm.invoke(translation_prompt)
        return result.content

    def _query_healthcare_context(self, query: str) -> str:
        """Query the healthcare context knowledge base"""
        # Create retrieval chain
        retrieval_chain = create_history_aware_retriever(
            self.llm, 
            self.retriever, 
            "Answer the following question about healthcare in Ghana using the retrieved documents: {input}"
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            self.llm,
            ChatPromptTemplate.from_template(
                "Answer the following question based on the context provided:\n\n{context}\n\nQuestion: {input}"
            )
        )
        
        # Create retrieval chain
        chain = create_retrieval_chain(retrieval_chain, document_chain)
        
        # Run chain
        result = chain.invoke({"input": query, "chat_history": self.memory.chat_memory.messages})
        return result["answer"]

    def _switch_language(self, language: str) -> str:
        """Switch the conversation language"""
        try:
            new_language = Language(language)
            self.context.language = new_language
            
            # Regenerate the agent with the new language setting
            self.agent_executor = self._create_agent()
            
            return f"Language switched to {new_language}"
        except ValueError:
            return f"Invalid language: {language}. Available languages: {[l.value for l in Language]}"

    def _generate_report(self, _: str = "") -> str:
        """Generate a report summarizing the conversation insights"""
        report_prompt = """
Based on our conversation, generate a detailed report that covers:

1. STAKEHOLDER PROFILE
   - Role and responsibilities
   - Clinic setting and resources
   - Key challenges mentioned

2. CURRENT PRACTICES
   - Follow-up care processes
   - Referral systems
   - Patient record-keeping methods
   - Skin condition management approach
   
3. DIGITAL READINESS
   - Current technology usage
   - Openness to new digital tools
   - Potential barriers to adoption
   - Specific feedback on skin condition diagnosis app
   
4. PARTNERSHIP POTENTIAL
   - Interest level in Care Across Borders solution
   - Alignment with current needs
   - Recommended next steps
   - Key features that would drive adoption

5. LANGUAGE AND COMMUNICATION
   - Preferred language for patient interaction
   - Preferred language for documentation
   - Communication patterns between healthcare levels

Provide specific quotes or examples from the conversation to support your findings.
Focus particularly on insights about skin condition management and diagnosis.
"""
        
        # Inject all conversation history
        chat_history = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" 
                                 for m in self.memory.chat_memory.messages])
        
        result = self.llm.invoke(f"{report_prompt}\n\nCONVERSATION TRANSCRIPT:\n{chat_history}")
        return result.content

    def set_stakeholder_role(self, role: StakeholderRole):
        """Set the stakeholder role for the conversation"""
        self.context.role = role
        # Regenerate the agent with the new role
        self.agent_executor = self._create_agent()
        
    def set_clinic_location(self, location: str):
        """Set the clinic location"""
        self.context.clinic_location = location
        
    def set_clinic_resources(self, resources: str):
        """Set the clinic resources level"""
        self.context.clinic_resources = resources
        
    def set_internet_connectivity(self, connectivity: str):
        """Set the internet connectivity level"""
        self.context.internet_connectivity = connectivity
    
    def run(self, user_input: str) -> str:
        """Run the agent with user input"""
        result = self.agent_executor.invoke({"input": user_input})
        return result["output"]


# Streamlit UI Implementation
def create_streamlit_app():
    st.set_page_config(page_title="Care Across Borders - Healthcare Discovery Agent", layout="wide")
    
    st.title("Care Across Borders")
    st.subheader("Healthcare Stakeholder Simulation Agent")
    
    # Initialize session state
    if "agent" not in st.session_state:
        # Get API key
        openai_api_key = os.getenv("OPENAI_API_KEY", None)
        if not openai_api_key:
            openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if not openai_api_key:
                st.warning("Please enter your OpenAI API key to continue.")
                return
                
        st.session_state.agent = HealthcareAgent(openai_api_key)
        st.session_state.messages = []
        st.session_state.report_generated = False
        st.session_state.interview_transcripts = []
        st.session_state.selected_transcript = None
    
    # Sidebar for configuration
    st.sidebar.title("Conversation Settings")
    
    # Role selection
    role = st.sidebar.selectbox(
        "Stakeholder Role",
        [role.value for role in StakeholderRole],
        index=0
    )
    
    # Language selection
    language = st.sidebar.selectbox(
        "Conversation Language",
        [lang.value for lang in Language],
        index=0
    )
    
    # Location and resources
    location = st.sidebar.text_input("Clinic Location", "Rural community near Accra")
    resources = st.sidebar.selectbox("Clinic Resources", ["Very Limited", "Limited", "Moderate", "Well-equipped"])
    connectivity = st.sidebar.selectbox("Internet Connectivity", ["None", "Intermittent", "Stable but slow", "Good"])
    
    # Apply settings button
    if st.sidebar.button("Apply Settings"):
        st.session_state.agent.set_stakeholder_role(StakeholderRole(role))
        st.session_state.agent._switch_language(language)
        st.session_state.agent.set_clinic_location(location)
        st.session_state.agent.set_clinic_resources(resources)
        st.session_state.agent.set_internet_connectivity(connectivity)
        st.sidebar.success("Settings applied!")
        
        # Add system message about the new context
        system_msg = f"[System: Now speaking with a {role} in {language}. Location: {location}, Resources: {resources}, Connectivity: {connectivity}]"
        st.session_state.messages.append({"role": "system", "content": system_msg})
    
    # Generate report button
    if st.sidebar.button("Generate Report"):
        with st.spinner("Generating report..."):
            report = st.session_state.agent._generate_report()
            st.session_state.report = report
            st.session_state.report_generated = True
            st.session_state.messages.append({"role": "system", "content": "[Report generated. You can view it in the Report tab.]"})
    
    # View interview transcripts section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Reference Interview Transcripts")
    
    # Display interview transcript buttons
    transcript_options = [
        "Nurse Interview", 
        "District Health Officer Interview", 
        "Community Health Worker Interview"
    ]
    
    selected_transcript = st.sidebar.radio(
        "Select transcript to view:",
        transcript_options,
        index=0
    )
    
    if selected_transcript != st.session_state.selected_transcript:
        st.session_state.selected_transcript = selected_transcript
        # Load the appropriate transcript
        if selected_transcript == "Nurse Interview":
            with open("nurse_interview.txt", "r") as f:
                st.session_state.current_transcript = f.read()
        elif selected_transcript == "District Health Officer Interview":
            with open("dhmt_interview.txt", "r") as f:
                st.session_state.current_transcript = f.read()
        elif selected_transcript == "Community Health Worker Interview":
            with open("chw_interview.txt", "r") as f:
                st.session_state.current_transcript = f.read()
    
    # Tabs for chat, report, and transcript
    chat_tab, report_tab, transcript_tab = st.tabs(["Conversation", "Report", "Interview Transcript"])
    
    with chat_tab:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "ai":
                st.chat_message("assistant").write(message["content"]) 
            elif message["role"] == "system":
                st.info(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Get AI response
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(prompt)
                st.session_state.messages.append({"role": "ai", "content": response})
                st.chat_message("assistant").write(response)
    
    with report_tab:
        if st.session_state.report_generated:
            st.markdown(st.session_state.report)
        else:
            st.info("No report generated yet. Complete your conversation and click 'Generate Report' in the sidebar.")
    
    with transcript_tab:
        if hasattr(st.session_state, 'current_transcript'):
            st.markdown(st.session_state.current_transcript)
        else:
            st.info("Select a transcript from the sidebar to view here.")

if __name__ == "__main__":
    create_streamlit_app()