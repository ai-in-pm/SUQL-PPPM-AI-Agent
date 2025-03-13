import os
import sys
import unittest
import pandas as pd
import uuid
import json
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.risk_management_agent import RiskManagementAgent
from database.db_manager import DatabaseManager
from database.suql import SUQLProcessor

class TestRiskManagement(unittest.TestCase):
    """
    Test cases for the Risk Management Agent with database and RAG integrations
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment with a test database
        """
        # Create test database in memory
        cls.db_manager = DatabaseManager(":memory:")
        
        # Create risk management agent with test database
        cls.risk_agent = RiskManagementAgent(cls.db_manager)
        cls.risk_agent.initialize()
        
        # Create sample project
        cls.project_id = f"TEST-{uuid.uuid4().hex[:8]}"
        cls.db_manager.insert_project({
            "id": cls.project_id,
            "name": "Test Construction Project",
            "description": "A test construction project for a commercial building",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "status": "In Progress",
            "budget": 1000000.0
        })
        
        # Add sample documents for RAG
        cls.add_sample_documents()
    
    @classmethod
    def add_sample_documents(cls):
        """
        Add sample documents to test RAG functionality
        """
        # Document 1: Project Charter
        cls.db_manager.add_document({
            "project_id": cls.project_id,
            "title": "Project Charter",
            "content": """This project charter outlines the construction of a 10-story commercial building.
            Key risks include potential material shortages due to supply chain disruptions,
            labor shortages in specialized trades, and potential cost overruns due to design changes.
            Weather conditions might impact the schedule during the foundation phase.
            The project has a tight timeline with penalties for late delivery.""",
            "document_type": "charter",
            "date": "2024-01-01"
        })
        
        # Document 2: Status Report
        cls.db_manager.add_document({
            "project_id": cls.project_id,
            "title": "March Status Report",
            "content": """The project is currently 2 weeks behind schedule due to unexpected ground conditions
            discovered during excavation. The steel supplier has indicated a potential 3-week delay
            in delivery of structural components. Budget is currently tracking 5% over the baseline
            due to additional foundation work. Two key team members have resigned, creating resource gaps.
            Weather forecasts indicate heavy rain next month during the critical roofing phase.""",
            "document_type": "status_report",
            "date": "2024-03-15"
        })
        
        # Document 3: Risk Register
        cls.db_manager.add_document({
            "project_id": cls.project_id,
            "title": "Initial Risk Register",
            "content": """Initial risk assessment identified the following high-priority risks:
            1. Supply chain disruptions affecting material delivery (High)
            2. Labor shortages in electrical and plumbing trades (Medium)
            3. Design changes requested by the client (Medium)
            4. Adverse weather conditions during critical construction phases (High)
            5. Potential permit delays from local regulatory bodies (Medium)""",
            "document_type": "risk_register",
            "date": "2024-01-10"
        })
    
    def test_risk_identification_with_rag(self):
        """
        Test risk identification with RAG enhancement
        """
        # Sample project data
        data = [
            {"task": "Foundation", "status": "Completed", "budget": 200000, "actual_cost": 220000, "variance": -20000},
            {"task": "Structural Steel", "status": "Delayed", "budget": 300000, "actual_cost": 150000, "variance": 150000},
            {"task": "Electrical", "status": "Not Started", "budget": 150000, "actual_cost": 0, "variance": 150000},
            {"task": "Plumbing", "status": "Not Started", "budget": 120000, "actual_cost": 0, "variance": 120000},
            {"task": "Roofing", "status": "Not Started", "budget": 80000, "actual_cost": 0, "variance": 80000},
        ]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Process with RAG
        result = self.risk_agent.process({
            'data': df,
            'project_context': "Commercial building construction project with a tight schedule",
            'risk_categories': ["Technical", "Schedule", "Budget"],
            'project_id': self.project_id,
            'use_rag': True,
            'rag_query': "construction delays and supply chain issues"
        })
        
        # Verify results
        self.assertIn('risks', result)
        self.assertIn('risk_summary', result)
        self.assertGreater(len(result['risks']), 0)
        
        # Check if RAG information influenced risk identification
        supply_chain_found = False
        for risk in result['risks']:
            description = risk.get('description', '').lower()
            if 'supply chain' in description or 'material' in description:
                supply_chain_found = True
                break
        
        self.assertTrue(supply_chain_found, "RAG did not successfully identify supply chain risks")
        
        # Verify risks were stored in database
        risks_in_db = self.db_manager.query("SELECT * FROM risks WHERE project_id = ?", (self.project_id,))
        self.assertGreater(len(risks_in_db), 0)
    
    def test_suql_risk_query(self):
        """
        Test SUQL queries with risk data and RAG
        """
        # Create SUQL processor
        suql_processor = SUQLProcessor(self.db_manager)
        
        # Run SUQL query with RAG
        query = f"""SELECT * FROM risks 
                  WHERE project_id = '{self.project_id}' 
                  AND RAG('weather conditions affecting construction')"""
        
        results = suql_processor.process_query(query)
        
        # Verify results contain both SQL and RAG components
        self.assertIn('sql_results', results)
        self.assertIn('rag_results', results)
        self.assertGreater(len(results['sql_results']), 0)
    
    def test_risk_query_from_db(self):
        """
        Test querying risks from the database
        """
        # Query risks from the database
        risk_df = self.risk_agent.query_risks_from_db(
            project_id=self.project_id,
            min_risk_score=0.5  # Medium-high risks
        )
        
        # Verify results
        self.assertIsInstance(risk_df, pd.DataFrame)
        self.assertGreater(len(risk_df), 0)
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up after tests
        """
        # No need to clean up in-memory database
        pass

if __name__ == "__main__":
    unittest.main()
