import unittest
import os
import sys
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.insights_generation_agent import InsightsGenerationAgent
from database.db_manager import DatabaseManager

class TestInsightsGenerationAgent(unittest.TestCase):
    
    def setUp(self):
        # Create a test database manager with in-memory SQLite database
        self.db_manager = DatabaseManager(':memory:')
        
        # Set up the database schema
        self.db_manager._initialize_database()
        
        # Insert test data
        self._insert_test_data()
        
        # Create the agent with the test database
        self.agent = InsightsGenerationAgent(db_manager=self.db_manager)
        self.agent.initialize()
    
    def _insert_test_data(self):
        # Insert a test project
        project = {
            'id': 'test_project_1',
            'name': 'Test Project',
            'description': 'This is a test project for insights testing',
            'metadata': json.dumps({
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'budget': 100000,
                'priority': 'High'
            })
        }
        self.db_manager.add_project(project)
        
        # Insert test documents
        for i in range(3):
            document = {
                'project_id': 'test_project_1',
                'title': f'Test Document {i+1}',
                'content': f'This is test document {i+1} with sample content for project insights testing. It contains key information about project progress, timelines, and resources.',
                'document_type': 'report' if i < 2 else 'insights',
                'metadata': json.dumps({
                    'author': f'Test Author {i+1}',
                    'date': f'2023-0{i+1}-15',
                    'version': f'1.{i}'
                })
            }
            self.db_manager.add_document(document)
    
    @patch('agents.insights_generation_agent.Anthropic')
    def test_process_with_database(self, mock_anthropic):
        # Mock the Anthropic client's messages.create method
        mock_content = MagicMock()
        mock_content.text = 'This is a mock insight'
        
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        
        mock_anthropic.return_value = mock_client
        
        # Create test data
        test_data = pd.DataFrame({
            'task_id': [1, 2, 3, 4, 5],
            'task_name': ['Task A', 'Task B', 'Task C', 'Task D', 'Task E'],
            'status': ['Complete', 'In Progress', 'Not Started', 'Complete', 'In Progress'],
            'progress': [100, 50, 0, 100, 75],
            'assigned_to': ['Person A', 'Person B', 'Person C', 'Person A', 'Person D'],
            'due_date': pd.to_datetime(['2023-01-15', '2023-02-01', '2023-02-15', '2023-01-30', '2023-02-28']),
            'priority': ['High', 'Medium', 'Low', 'High', 'Medium']
        })
        
        # Test the process method with database integration
        input_data = {
            'query_result': test_data,
            'context': 'project management tasks',
            'insight_type': 'progress',
            'project_id': 'test_project_1',
            'use_rag': True,
            'rag_query': 'project progress',
            'store_insights': True
        }
        
        result = self.agent.process(input_data)
        
        # Verify that the process method returned the expected structure
        self.assertIn('insights', result)
        self.assertIn('recommendations', result)
        self.assertIn('visualizations', result)
        self.assertIn('summary', result)
        
        # Verify that Anthropic was called
        mock_client.messages.create.assert_called()
        
        # Query the database to verify the insights were stored
        query = """
        SELECT * FROM documents 
        WHERE project_id = 'test_project_1' AND document_type = 'insights'
        ORDER BY id DESC LIMIT 1
        """
        result = self.db_manager.query(query)
        
        # Verify that at least one insight document was found
        self.assertGreater(len(result), 0)
        
        # Verify that the document contains insight information
        self.assertIn('Insights:', result[0]['content'])
    
    @patch('agents.insights_generation_agent.Anthropic')
    def test_fetch_historical_insights(self, mock_anthropic):
        # Mock the Anthropic client's messages.create method
        mock_content = MagicMock()
        mock_content.text = 'This is a mock insight'
        
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        
        mock_anthropic.return_value = mock_client
        
        # Test fetching historical insights
        historical_insights = self.agent.fetch_historical_insights('test_project_1')
        
        # Verify we get at least one insight document
        self.assertGreater(len(historical_insights), 0)
        self.assertEqual(historical_insights[0]['document_type'], 'insights')
    
    @patch('agents.insights_generation_agent.Anthropic')
    def test_compare_with_historical_insights(self, mock_anthropic):
        # Mock the Anthropic client's messages.create method
        mock_content = MagicMock()
        mock_content.text = 'This is a mock comparison analysis'
        
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        
        mock_anthropic.return_value = mock_client
        
        # Create current insights for comparison
        current_insights = [
            {
                'title': 'Current Insight 1',
                'description': 'Description of current insight 1'
            },
            {
                'title': 'Current Insight 2',
                'description': 'Description of current insight 2'
            }
        ]
        
        # Test comparing with historical insights
        comparison = self.agent.compare_with_historical_insights(current_insights, 'test_project_1')
        
        # Verify the comparison result structure
        self.assertIn('comparison', comparison)
        self.assertIn('historical_insight_count', comparison)
        
        # Verify that Anthropic was called for the comparison
        mock_client.messages.create.assert_called()
    
    def test_rag_enhancement(self):
        # Mock the RAG search results
        with patch.object(self.db_manager, 'perform_rag_search') as mock_rag:
            mock_rag.return_value = [
                {'content': 'Relevant document content 1'},
                {'content': 'Relevant document content 2'}
            ]
            
            # Create test data
            test_data = pd.DataFrame({
                'metric': ['A', 'B', 'C'],
                'value': [10, 20, 30]
            })
            
            # Mock the _generate_insights method to avoid calling Anthropic
            with patch.object(self.agent, '_generate_insights') as mock_insights:
                mock_insights.return_value = [{'title': 'Mock Insight', 'description': 'Description'}]
                
                # Mock other methods to avoid generating actual visualizations
                with patch.object(self.agent, '_generate_visualizations') as mock_viz:
                    mock_viz.return_value = [{'title': 'Mock Viz', 'data': 'mock_data'}]
                    
                    with patch.object(self.agent, '_generate_recommendations') as mock_rec:
                        mock_rec.return_value = [{'title': 'Mock Rec', 'steps': ['Step 1']}]
                        
                        with patch.object(self.agent, '_generate_summary') as mock_summary:
                            mock_summary.return_value = 'Mock summary'
                            
                            # Test the process method with RAG
                            input_data = {
                                'query_result': test_data,
                                'context': 'test context',
                                'use_rag': True,
                                'rag_query': 'test query',
                                'project_id': 'test_project_1'
                            }
                            
                            result = self.agent.process(input_data)
                            
                            # Verify RAG search was called
                            mock_rag.assert_called_with('test query', 'test_project_1')
                            
                            # Verify insights were generated with combined context
                            call_args = mock_insights.call_args[0]
                            self.assertIn('Relevant project documents:', call_args[1])

if __name__ == '__main__':
    unittest.main()
