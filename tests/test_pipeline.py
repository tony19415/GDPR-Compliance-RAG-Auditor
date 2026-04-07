import pytest
from unittest.mock import MagicMock, patch
from audit_engine import run_compliance_audit

# --- Mock Helpers ---
class MockDocument:
    def __init__(self, page_content):
        self.page_content = page_content

class MockLLMResponse:
    def __init__(self, content):
        self.content = content

# --- Unit Tests ---

def test_run_compliance_audit_flow():
    """
    Verifies the full orchestration: Retrieval -> LLM -> JSON Parsing -> Normalization.
    """
    # 1. Setup Mock Dependencies
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [MockDocument("Art. 15 GDPR: Right to access.")]
    
    mock_vdb = MagicMock() # Passed to extract_contract_clause
    checkpoint = {"query": "Does the user have access rights?"}
    
    # 2. Patch the LLM and the contract extraction helper
    with patch('audit_engine.ChatOllama') as MockChat, \
         patch('audit_engine.extract_contract_clause') as MockExtract:
        
        # Mocking the contract text extraction
        MockExtract.return_value = "Contract says: Data is shared with third parties."
        
        # Mocking the LLM behavior (simulating 'yapping' + messy keys)
        mock_model_instance = MockChat.return_value
        messy_json = """
        Audit complete. Result is:
        {
            "STATUS": "FAIL",
            "Violation_Found": true,
            "Reasoning": "According to Art. 15, access is required, but contract lacks clause.",
            "Rememdy": "Add access request procedure."
        }
        """
        mock_model_instance.invoke.return_value = MockLLMResponse(messy_json)
        
        # 3. Execute the function
        analysis, context = run_compliance_audit(mock_retriever, mock_vdb, checkpoint)
        
        # 4. Assertions (The "Data Integrity" Checks)
        assert isinstance(analysis, dict)
        assert analysis["status"] == "FAIL"  # Verification of key.lower() normalization
        assert analysis["violation_found"] is True
        assert "reasoning" in analysis
        
        # Verify Context Preservation (Crucial for RAGAS eval loop)
        assert isinstance(context, list)
        assert "Art. 15" in context[0].page_content

def test_audit_error_handling():
    """
    Ensures that when the LLM fails, we still return a valid list 
    for the context to avoid breaking the RAGAS evaluation pipeline.
    """
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [MockDocument("Backup context")]
    mock_vdb = MagicMock()
    checkpoint = {"query": "force a failure"}

    with patch('audit_engine.ChatOllama') as MockChat, \
         patch('audit_engine.extract_contract_clause') as MockExtract:
        
        MockExtract.return_value = "..."
        mock_model_instance = MockChat.return_value
        # Simulate a total formatting failure (no JSON block)
        mock_model_instance.invoke.return_value = MockLLMResponse("I'm sorry, I can't do that.")
        
        analysis, context = run_compliance_audit(mock_retriever, mock_vdb, checkpoint)
        
        # Verify fallback structure
        assert analysis["status"] == "ERROR"
        assert "Formatting failure" in analysis["reasoning"]
        # The 'context' MUST be the list of Documents, not a string
        assert isinstance(context, list)