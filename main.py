# Healthcare Discovery Agent - LangChain Implementation
# For Care Across Borders Telemedicine Venture

import os
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

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

# Define a model for the conversation context
class ConversationContext(BaseModel):
    language: Language = Field(default=Language.ENGLISH)
    role: StakeholderRole = Field(default=StakeholderRole.CLINIC_MANAGER)
    clinic_location: str = Field(default="Rural community near Accra")
    clinic_resources: str = Field(default="Limited")
    internet_connectivity: str = Field(default="Intermittent")

# Function to create the knowledge base for Ghana healthcare context
def create_knowledge_base():
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
        """)
    
    # Load and process the knowledge base
    loader = TextLoader("ghana_healthcare_context.txt")
    docs = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    return retriever

# Main Healthcare Agent Class
class HealthcareAgent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, api_key=openai_api_key)
        self.context = ConversationContext()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.retriever = create_knowledge_base()
        self.agent_executor = self._create_agent()
        
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
             retrieval_tool, switch_language_tool, generate_report_tool],
            prompt
        )
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=[translate_to_english_tool, translate_from_english_tool, 
                   retrieval_tool, switch_language_tool, generate_report_tool],
            verbose=True,
            memory=self.memory
        )
    
    def _generate_system_prompt(self) -> str:
        role_descriptions = {
            StakeholderRole.CLINIC_MANAGER: "The clinic manager oversees operations, makes administrative decisions, handles finances, and coordinates staff. They're concerned with efficiency, resource allocation, and overall clinic performance.",
            StakeholderRole.PHYSICIAN: "The physician provides direct patient care, makes diagnoses, prescribes treatments, and manages patient follow-ups. They're concerned with patient outcomes, treatment options, and clinical resources.",
            StakeholderRole.HEALTH_WORKER: "The community health worker provides basic care, conducts health education, and refers patients to clinics when needed. They typically have a high school diploma, operate in remote areas, and use handwritten communication systems."
        }
        
        language_examples = {
            Language.ENGLISH: "",
            Language.TWI: "Examples of greetings in Twi: 'Ɛte sɛn?' (How are you?), 'Me din de...' (My name is...)",
            Language.GA: "Examples of greetings in Ga: 'Tɛ wɔ ŋɔ?' (How are you?), 'Miiŋwɛmi ji...' (My name is...)"
        }
        
        system_prompt = f"""
You are an AI designed for Care Across Borders, an early-stage telemedicine venture connecting doctors and patients across countries with AI-powered translation and locally adapted health protocols.

You are currently simulating a conversation with a {self.context.role} in a healthcare facility in Ghana.

Current context:
- Location: {self.context.clinic_location}
- Resources: {self.context.clinic_resources}
- Internet connectivity: {self.context.internet_connectivity}
- Current language: {self.context.language}

Role information:
{role_descriptions[self.context.role]}

{language_examples[self.context.language]}

Your goal is to gather information about:
1. How they handle follow-up care
2. Their referral processes
3. Patient record-keeping methods
4. Their openness to digital health solutions

Ask thoughtful, open-ended questions. Show empathy and understanding of local challenges. Adapt your communication style to match their role and responsibilities.

If the conversation is not in English, use the translation tools to communicate effectively.

At the end of the conversation, be prepared to generate a report summarizing the key insights gathered.
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
   
3. DIGITAL READINESS
   - Current technology usage
   - Openness to new digital tools
   - Potential barriers to adoption
   
4. PARTNERSHIP POTENTIAL
   - Interest level in Care Across Borders solution
   - Alignment with current needs
   - Recommended next steps

Provide specific quotes or examples from the conversation to support your findings.
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
    st.set_page_config(page_title="Care Across Borders - Healthcare Discovery Agent")
    
    st.title("Care Across Borders")
    st.subheader("Healthcare Stakeholder Simulation Agent")
    
    # Initialize session state
    if "agent" not in st.session_state:
        # Get API key
        openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not openai_api_key:
            openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if not openai_api_key:
                st.warning("Please enter your OpenAI API key to continue.")
                return
                
        st.session_state.agent = HealthcareAgent(openai_api_key)
        st.session_state.messages = []
        st.session_state.report_generated = False
    
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
    
    # Tabs for chat and report
    chat_tab, report_tab = st.tabs(["Conversation", "Report"])
    
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

if __name__ == "__main__":
    create_streamlit_app()