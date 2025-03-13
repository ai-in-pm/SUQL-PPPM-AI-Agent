import unittest
import os
import sys
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.data_extraction_agent import DataExtractionAgent
from agents.insights_generation_agent import InsightsGenerationAgent
from agents.risk_management_agent import RiskManagementAgent
from database.db_manager import DatabaseManager

class TestIntegratedAgents(unittest.TestCase):
    
    def setUp(self):
        # Create a test database manager with in-memory SQLite database
        self.db_manager = DatabaseManager(':memory:')
        
        # Set up the database schema
        self.db_manager._initialize_database()
        
        # Create all agents with the same database manager for integration testing
        self.data_extraction_agent = DataExtractionAgent(db_manager=self.db_manager)
        self.insights_agent = InsightsGenerationAgent(db_manager=self.db_manager)
        self.risk_agent = RiskManagementAgent(db_manager=self.db_manager)
        
        # Initialize all agents
        self.data_extraction_agent.initialize()
        self.insights_agent.initialize()
        self.risk_agent.initialize()
        
        # Create a test project
        project = {
            'id': 'test_project_integration',
            'name': 'Test Integration Project',
            'description': 'This is a test project for integrated agent testing',
            'metadata': json.dumps({
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'budget': 200000,
                'priority': 'High'
            })
        }
        self.db_manager.add_project(project)
    
    @patch('agents.data_extraction_agent.OpenAI')
    @patch('agents.insights_generation_agent.Anthropic')
    @patch('agents.risk_management_agent.OpenAI')
    def test_end_to_end_workflow(self, mock_risk_openai, mock_insights_anthropic, mock_data_openai):
        # Mock the OpenAI client for Data Extraction Agent
        mock_data_completion = MagicMock()
        mock_data_completion.choices = [MagicMock(message=MagicMock(content=json.dumps([
            {"task": "Task 1", "status": "Complete", "progress": 100, "owner": "John"},
            {"task": "Task 2", "status": "In Progress", "progress": 50, "owner": "Alice"},
            {"task": "Task 3", "status": "Not Started", "progress": 0, "owner": "Bob"}
        ])))]
        
        mock_data_client = MagicMock()
        mock_data_client.chat.completions.create.return_value = mock_data_completion
        mock_data_openai.return_value = mock_data_client
        
        # Mock the Anthropic client for Insights Generation Agent
        mock_insights_content = MagicMock()
        mock_insights_content.text = 'This is a mock insight. The project is progressing well.'
        
        mock_insights_message = MagicMock()
        mock_insights_message.content = [mock_insights_content]
        
        mock_insights_client = MagicMock()
        mock_insights_client.messages.create.return_value = mock_insights_message
        
        mock_insights_anthropic.return_value = mock_insights_client
        
        # Mock the OpenAI client for Risk Management Agent
        mock_risk_completion = MagicMock()
        mock_risk_completion.choices = [MagicMock(message=MagicMock(content=json.dumps([
            {"risk_id": "R1", "description": "Resource availability risk", "category": "Resource"},
            {"risk_id": "R2", "description": "Schedule delay risk", "category": "Schedule"}
        ])))]
        
        mock_risk_client = MagicMock()
        mock_risk_client.chat.completions.create.return_value = mock_risk_completion
        mock_risk_openai.return_value = mock_risk_client
        
        # Step 1: Extract data using the Data Extraction Agent
        data_input = {
            'data_source': 'Project meeting notes from March 10, 2023. Tasks discussed: Task 1 is complete (100%). Task 2 is in progress (50%). Task 3 has not started (0%).',
            'data_type': 'unstructured',
            'extraction_goal': 'Extract task information including status and progress',
            'project_id': 'test_project_integration',
            'store_in_database': True
        }
        
        # Process the data through the Data Extraction Agent
        data_result = self.data_extraction_agent.process(data_input)
        
        # Verify data was extracted
        self.assertIsNotNone(data_result)
        self.assertIn('extracted_data', data_result)
        
        # Verify data was stored in the database
        docs_query = "SELECT * FROM documents WHERE project_id = 'test_project_integration'"
        docs = self.db_manager.query(docs_query)
        self.assertGreater(len(docs), 0)
        
        # Step 2: Generate insights from the extracted data
        df = pd.DataFrame(data_result['extracted_data'])
        
        insights_input = {
            'query_result': df,
            'context': 'project task status',
            'insight_type': 'progress',
            'project_id': 'test_project_integration',
            'use_rag': True,
            'rag_query': 'task progress',
            'store_insights': True
        }
        
        # Process the insights through the Insights Generation Agent
        insights_result = self.insights_agent.process(insights_input)
        
        # Verify insights were generated
        self.assertIsNotNone(insights_result)
        self.assertIn('insights', insights_result)
        self.assertIn('recommendations', insights_result)
        
        # Verify insights were stored in the database
        insights_query = "SELECT * FROM documents WHERE project_id = 'test_project_integration' AND document_type = 'insights'"
        insights_docs = self.db_manager.query(insights_query)
        self.assertGreater(len(insights_docs), 0)
        
        # Step 3: Identify risks based on the same data
        risk_input = {
            'data': df,
            'project_id': 'test_project_integration',
            'risk_categories': ['Schedule', 'Resource', 'Technical', 'Budget'],
            'project_context': 'Integration test project with task tracking',
            'use_rag': True,
            'query': 'task progress risks'
        }
        
        # Process the risks through the Risk Management Agent
        risk_result = self.risk_agent.identify_risks(risk_input)
        
        # Verify risks were identified
        self.assertIsNotNone(risk_result)
        self.assertIn('risks', risk_result)
        
        # Verify risks were stored in the database
        risks_query = "SELECT * FROM risks WHERE project_id = 'test_project_integration'"
        risks = self.db_manager.query(risks_query)
        self.assertGreater(len(risks), 0)
        
        # Step 4: Test SUQL query through Risk Management Agent
        suql_query = """SUQL
            SELECT risk_description, risk_category, probability, impact, mitigation_strategy
            FROM risks
            WHERE project_id = 'test_project_integration'
            AND risk_category IN ('Schedule', 'Resource')
            ANSWER "What are the main schedule and resource risks for this project?"
        """
        
        # Mock the rag_search method for SUQL processing
        with patch.object(self.db_manager, 'perform_rag_search') as mock_rag:
            mock_rag.return_value = [{'content': 'Relevant document content for risks'}]
            
            # Process the SUQL query
            suql_result = self.risk_agent.process_suql_query(suql_query, 'test_project_integration')
            
            # Verify the SUQL result
            self.assertIsNotNone(suql_result)
            self.assertIn('sql_result', suql_result)
            self.assertIn('enriched_result', suql_result)
        
        # Verify the flow of information between agents through the database
        # The Risk Management Agent should have access to insights generated earlier
        insights_for_risk_query = """
            SELECT documents.content 
            FROM documents 
            WHERE project_id = 'test_project_integration' 
            AND document_type = 'insights' 
            ORDER BY id DESC LIMIT 1
        """
        insights_for_risk = self.db_manager.query(insights_for_risk_query)
        
        # Use the insights content for RAG in risk analysis
        with patch.object(self.db_manager, 'perform_rag_search') as mock_rag:
            mock_rag.return_value = [{'content': insights_for_risk[0]['content']}]
            
            # Generate mitigation strategies using insights context
            mitigation_result = self.risk_agent.generate_mitigation_strategies(
                project_id='test_project_integration',
                include_rag=True,
                rag_query='mitigation strategies based on insights'
            )
            
            # Verify mitigation strategies were generated
            self.assertIsNotNone(mitigation_result)
            mock_rag.assert_called_once()

if __name__ == '__main__':
    unittest.main()
