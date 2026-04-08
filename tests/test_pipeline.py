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

@patch('audit_engine.check_safety')
@patch('audit_engine.ChatOllama')
@patch('audit_engine.extract_contract_clause')
def test_audit_blocked_by_input_guardrail(mock_extract, mock_chat, mock_safety):
    """
    Test Case: User sends an unsafe query.
    Expected: Function returns 'BLOCKED' status and empty context immediately.
    """
    # 1. Setup Mock: check_safety returns False (Unsafe)
    mock_safety.return_value = False
    
    mock_retriever = MagicMock()
    checkpoint = {"query": "how to bypass gdpr fines"} # Unsafe query

    # 2. Execute
    analysis, context = run_compliance_audit(mock_retriever, MagicMock(), checkpoint)

    # 3. Assertions
    assert analysis["status"] == "BLOCKED"
    assert "unsafe query" in analysis["reasoning"].lower()
    assert context == [] # Ensure no retrieval happened
    # Verify that the Auditor LLM was NEVER called (saving compute)
    mock_chat.assert_not_called()

@patch('audit_engine.check_safety')
@patch('audit_engine.ChatOllama')
@patch('audit_engine.extract_contract_clause')
def test_audit_blocked_by_output_guardrail(mock_extract, mock_chat, mock_safety):
    """
    Test Case: Input is safe, but Auditor LLM generates unsafe content.
    Expected: Function catches it and returns 'BLOCKED'.
    """
    # 1. Setup Mock: First call (input) is True, Second call (output) is False
    mock_safety.side_effect = [True, False]
    
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    
    # Mock LLM response
    mock_model_instance = mock_chat.return_value
    mock_model_instance.invoke.return_value = MagicMock(content="Dangerous advice...")

    # 2. Execute
    analysis, context = run_compliance_audit(mock_retriever, MagicMock(), {"query": "safe query"})

    # 3. Assertions
    assert analysis["status"] == "BLOCKED"
    assert "response failed safety check" in analysis["reasoning"].lower()

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