import json
import pandas as pd
import numpy as np
import uuid
import datetime
import sqlite3
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path
from anthropic import Anthropic

from agents.base_agent import BaseAgent
from utils.config import ANTHROPIC_API_KEY
from database.db_manager import DatabaseManager

class RiskManagementAgent(BaseAgent):
    """
    Agent responsible for identifying and analyzing project risks using Structured Unstructured Query Language (SUQL).
    
    This agent processes data from SUQL queries to identify potential risks in projects,
    portfolios, and programs, and provides mitigation strategies and risk assessments.
    
    It uses Anthropic's Claude model for risk identification and analysis and integrates
    with SQLite database and RAG functionality for enhanced risk management.
    """
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        super().__init__(
            name="Risk Management Agent",
            description="Identifies and analyzes project risks using Structured Unstructured Query Language (SUQL)"
        )
        self.client = None
        self.output_dir = Path("outputs/risks")
        
        # Initialize database manager
        if db_manager is None:
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = db_manager
    
    def _initialize_resources(self):
        """
        Initialize Anthropic client and create output directory
        """
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query results to identify and analyze risks
        
        Args:
            input_data: Dictionary containing:
                - 'data': DataFrame, JSON data, or dictionary from query results
                - 'project_context': Optional context about the project, portfolio, or program
                - 'risk_categories': Optional list of risk categories to focus on
                - 'project_id': Optional project ID for database operations
                - 'use_rag': Optional boolean to use RAG for risk identification
                - 'rag_query': Optional query for RAG-based risk identification
        
        Returns:
            Dictionary with identified risks and mitigation strategies
        """
        if not self.is_initialized:
            self.initialize()
        
        # Get data from input
        data = input_data.get('data')
        if data is None:
            raise ValueError("No data provided for risk analysis")
        
        # Convert to DataFrame if it's JSON or dict
        if not isinstance(data, pd.DataFrame):
            data = self._convert_to_dataframe(data)
        
        # Get optional parameters
        project_context = input_data.get('project_context', '')
        risk_categories = input_data.get('risk_categories', [])
        project_id = input_data.get('project_id', None)
        use_rag = input_data.get('use_rag', False)
        rag_query = input_data.get('rag_query', None)
        
        # Use RAG to enhance risk identification if requested
        rag_context = None
        if use_rag and rag_query:
            rag_results = self.db_manager.perform_rag_search(rag_query, project_id)
            if rag_results:
                rag_context = "\n\nRelevant project documents:\n"
                for i, result in enumerate(rag_results):
                    rag_context += f"Document {i+1}: {result.get('content', '')}\n"
        
        # Combine project context with RAG context
        if rag_context:
            combined_context = f"{project_context}\n\n{rag_context}"
        else:
            combined_context = project_context
        
        # Identify risks
        risks = self._identify_risks(data, combined_context, risk_categories)
        
        # Analyze risks (severity, probability, impact)
        analyzed_risks = self._analyze_risks(risks, data, combined_context)
        
        # Generate mitigation strategies
        mitigated_risks = self._generate_mitigation_strategies(analyzed_risks, data, combined_context)
        
        # Store risks in database if project_id is provided
        if project_id:
            self._store_risks_in_db(mitigated_risks, project_id)
        
        return {
            "risks": mitigated_risks,
            "risk_summary": self._generate_risk_summary(mitigated_risks),
            "project_id": project_id
        }
    
    def _store_risks_in_db(self, risks: List[Dict[str, Any]], project_id: str) -> None:
        """
        Store identified risks and their mitigation strategies in the database
        
        Args:
            risks: List of risks with mitigation strategies
            project_id: Project ID for database relation
        """
        for risk in risks:
            # Prepare risk data for database
            risk_data = {
                "id": risk.get('id', f"RISK-{uuid.uuid4().hex[:8]}"),
                "project_id": project_id,
                "title": risk.get('title', 'Unknown Risk'),
                "description": risk.get('description', ''),
                "category": risk.get('category', 'Uncategorized'),
                "probability": self._convert_probability_to_numeric(risk.get('probability', 'Medium')),
                "impact": self._convert_impact_to_numeric(risk.get('impact', 'Medium')),
                "risk_score": risk.get('risk_score', 0) if isinstance(risk.get('risk_score'), (int, float)) else 0,
                "status": "Open"
            }
            
            # Insert risk into database
            risk_id = self.db_manager.insert_risk(risk_data)
            
            # Store mitigation strategy if available
            if 'mitigation_strategy' in risk:
                strategy_data = {
                    "risk_id": risk_id,
                    "strategy": risk.get('mitigation_strategy', ''),
                    "action_plan": risk.get('action_plan', ''),
                    "owner": risk.get('owner', 'Unassigned'),
                    "status": "Planned"
                }
                
                self.db_manager.insert_mitigation_strategy(strategy_data)
    
    def _convert_probability_to_numeric(self, probability: str) -> float:
        """
        Convert probability text to numeric value
        
        Args:
            probability: Text probability (High/Medium/Low)
            
        Returns:
            Numeric probability value between 0 and 1
        """
        prob_map = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2,
            'very high': 0.9,
            'very low': 0.1
        }
        
        if isinstance(probability, (int, float)):
            return min(max(float(probability), 0.0), 1.0)
        
        return prob_map.get(probability.lower(), 0.5)
    
    def _convert_impact_to_numeric(self, impact: str) -> float:
        """
        Convert impact text to numeric value
        
        Args:
            impact: Text impact (High/Medium/Low)
            
        Returns:
            Numeric impact value between 0 and 1
        """
        impact_map = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2,
            'very high': 0.9,
            'very low': 0.1
        }
        
        if isinstance(impact, (int, float)):
            return min(max(float(impact), 0.0), 1.0)
        
        return impact_map.get(impact.lower(), 0.5)
        
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert various data formats to a pandas DataFrame
        
        Args:
            data: Input data in various formats (JSON, dict, list)
            
        Returns:
            DataFrame representation of the data
        """
        if isinstance(data, pd.DataFrame):
            return data
        
        try:
            # Handle string JSON
            if isinstance(data, str):
                data = json.loads(data)
            
            # Handle list of dictionaries
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return pd.DataFrame(data)
            
            # Handle flat dictionary
            if isinstance(data, dict) and not any(isinstance(v, (dict, list)) for v in data.values()):
                return pd.DataFrame([data])
            
            # Handle nested dictionary
            if isinstance(data, dict):
                try:
                    return pd.json_normalize(data)
                except Exception:
                    # Fallback - extract what we can
                    flat_data = {}
                    for key, value in data.items():
                        if not isinstance(value, (dict, list)):
                            flat_data[key] = value
                    
                    if flat_data:
                        return pd.DataFrame([flat_data])
            
            # Last resort
            return pd.DataFrame({"raw_data": [json.dumps(data)]})
            
        except Exception as e:
            # Create DataFrame with error info
            return pd.DataFrame({
                "error": [f"Failed to convert data: {str(e)}"],
                "raw_data": [str(data)]
            })
    
    def _identify_risks(self, df: pd.DataFrame, project_context: str, risk_categories: List[str]) -> List[Dict[str, Any]]:
        """Identify potential risks from the data
        
        Args:
            df: DataFrame with the data
            project_context: Context about the project
            risk_categories: Categories of risks to focus on
            
        Returns:
            List of identified risk dictionaries
        """
        # Get DataFrame description
        df_description = self._get_dataframe_description(df)
        
        # Format risk categories for the prompt
        risk_categories_text = "all risk categories"
        if risk_categories:
            risk_categories_text = ", ".join(risk_categories)
        
        # Construct a prompt for the AI
        prompt = f"""
        I have project data from a portfolio, program, or project management system.
        
        Data Description:
        {df_description}
        
        Project Context:
        {project_context}
        
        Based on this data, identify 3-7 key risks that could impact the project or portfolio. 
        Focus on {risk_categories_text}.
        
        For each risk, provide:
        - Risk ID: [a short identifier code]
        - Title: [a concise title for the risk]
        - Description: [a detailed description of the risk]
        - Category: [the category this risk falls under, e.g. technical, financial, schedule, etc.]
        - Source: [what data points or patterns in the provided data suggest this risk]
        
        Format each risk in a structured way with these exact fields.
        """
        
        try:
            # Use Anthropic to identify risks
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.1,  # Low temperature for factual analysis
                system="You are a risk management expert in project, portfolio, and program management. Your role is to identify potential risks based on project data and provide structured risk assessments.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            risk_text = message.content[0].text
            
            # Parse the risks into a structured format
            risks = []
            current_risk = {}
            current_field = None
            
            for line in risk_text.split('\n'):
                line = line.strip()
                
                if line.startswith('- Risk ID:'):
                    if current_risk and 'id' in current_risk:
                        risks.append(current_risk)
                    current_risk = {'id': line.replace('- Risk ID:', '').strip()}
                    current_field = 'id'
                elif line.startswith('- Title:') and current_risk:
                    current_risk['title'] = line.replace('- Title:', '').strip()
                    current_field = 'title'
                elif line.startswith('- Description:') and current_risk:
                    current_risk['description'] = line.replace('- Description:', '').strip()
                    current_field = 'description'
                elif line.startswith('- Category:') and current_risk:
                    current_risk['category'] = line.replace('- Category:', '').strip()
                    current_field = 'category'
                elif line.startswith('- Source:') and current_risk:
                    current_risk['source'] = line.replace('- Source:', '').strip()
                    current_field = 'source'
                elif line and current_field and current_risk and current_field in current_risk:
                    # Append to the current field for multi-line values
                    current_risk[current_field] += ' ' + line
            
            # Add the last risk
            if current_risk and 'id' in current_risk:
                risks.append(current_risk)
            
            return risks
            
        except Exception as e:
            # Return a single risk with the error
            return [{
                "id": "ERR-001",
                "title": "Error Identifying Risks",
                "description": f"Failed to identify risks: {str(e)}",
                "category": "Error",
                "source": "System error"
            }]
    
    def _get_dataframe_description(self, df: pd.DataFrame) -> str:
        """Generate a descriptive summary of the DataFrame
        
        Args:
            df: DataFrame to describe
            
        Returns:
            String description of the DataFrame
        """
        # Basic info
        info = f"DataFrame with {len(df)} rows and {len(df.columns)} columns.\n\n"
        
        # Columns with types
        columns_info = "Columns:\n" + "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns]) + "\n\n"
        
        # Sample data (first few rows)
        try:
            sample_data = "Sample Data:\n" + df.head(3).to_string() + "\n\n"
        except:
            sample_data = "Sample Data: Unable to display\n\n"
        
        # Look for columns that might indicate risks
        risk_relevant_cols = []
        risk_keywords = ['risk', 'issue', 'problem', 'delay', 'cost', 'budget', 'overrun', 'schedule', 
                       'deadline', 'quality', 'resource', 'scope', 'change', 'dependency', 'stakeholder']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in risk_keywords):
                risk_relevant_cols.append(col)
        
        risk_cols_info = ""
        if risk_relevant_cols:
            risk_cols_info = "Risk-relevant columns:\n" + "\n".join([f"- {col}" for col in risk_relevant_cols]) + "\n\n"
        
        return info + columns_info + sample_data + risk_cols_info
    
    def _analyze_risks(self, risks: List[Dict[str, Any]], df: pd.DataFrame, project_context: str) -> List[Dict[str, Any]]:
        """Analyze risks to determine severity, probability, and impact
        
        Args:
            risks: List of identified risks
            df: DataFrame with the data
            project_context: Context about the project
            
        Returns:
            List of risks with analysis added
        """
        if not risks:
            return []
        
        # Format risks for the prompt
        risks_text = "\n\n".join([f"Risk ID: {risk['id']}\nTitle: {risk['title']}\nDescription: {risk['description']}\nCategory: {risk['category']}\nSource: {risk['source']}" for risk in risks])
        
        # Construct a prompt for the AI
        prompt = f"""
        Based on the following identified risks and project data:
        
        {risks_text}
        
        Project Context:
        {project_context}
        
        Analyze each risk and provide the following additional information:
        
        For each risk (use the same Risk IDs as provided), add:
        - Probability: [High/Medium/Low] - Likelihood of the risk occurring
        - Impact: [High/Medium/Low] - Severity of consequences if the risk occurs
        - Risk Score: [1-25] - Numerical score based on probability and impact
        - Timeframe: [Near-term/Medium-term/Long-term] - When the risk might occur
        
        Format your response with the Risk ID followed by only these four pieces of information for each risk.
        """
        
        try:
            # Use Anthropic to analyze risks
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1200,
                temperature=0.1,
                system="You are a risk management expert who specializes in quantitative and qualitative risk analysis for projects, portfolios, and programs.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis_text = message.content[0].text
            
            # Parse the analysis and update risks
            analyzed_risks = risks.copy()
            current_risk_id = None
            
            for line in analysis_text.split('\n'):
                line = line.strip()
                
                if line.startswith('Risk ID:'):
                    risk_id = line.replace('Risk ID:', '').strip()
                    current_risk_id = risk_id
                elif line.startswith('- Probability:') and current_risk_id:
                    # Find the risk with the matching ID
                    for risk in analyzed_risks:
                        if risk['id'] == current_risk_id:
                            risk['probability'] = line.replace('- Probability:', '').strip()
                elif line.startswith('- Impact:') and current_risk_id:
                    for risk in analyzed_risks:
                        if risk['id'] == current_risk_id:
                            risk['impact'] = line.replace('- Impact:', '').strip()
                elif line.startswith('- Risk Score:') and current_risk_id:
                    for risk in analyzed_risks:
                        if risk['id'] == current_risk_id:
                            score_text = line.replace('- Risk Score:', '').strip()
                            try:
                                risk['risk_score'] = int(score_text.split()[0])  # Extract just the number
                            except:
                                risk['risk_score'] = score_text
                elif line.startswith('- Timeframe:') and current_risk_id:
                    for risk in analyzed_risks:
                        if risk['id'] == current_risk_id:
                            risk['timeframe'] = line.replace('- Timeframe:', '').strip()
            
            return analyzed_risks
            
        except Exception as e:
            # Return the original risks with error info
            for risk in risks:
                risk['probability'] = "Unknown"
                risk['impact'] = "Unknown"
                risk['risk_score'] = "Error"
                risk['timeframe'] = "Unknown"
                risk['analysis_error'] = str(e)
            return risks
    
    def _generate_mitigation_strategies(self, analyzed_risks: List[Dict[str, Any]], df: pd.DataFrame, project_context: str) -> List[Dict[str, Any]]:
        """Generate mitigation strategies for identified risks
        
        Args:
            analyzed_risks: List of risks with analysis
            df: DataFrame with the data
            project_context: Context about the project
            
        Returns:
            List of risks with mitigation strategies added
        """
        if not analyzed_risks:
            return []
        
        # Format risks for the prompt
        risks_text = "\n\n".join([f"Risk ID: {risk['id']}\nTitle: {risk['title']}\nDescription: {risk['description']}\nCategory: {risk['category']}\nSource: {risk['source']}\nProbability: {risk['probability']}\nImpact: {risk['impact']}\nRisk Score: {risk['risk_score']}\nTimeframe: {risk['timeframe']}" for risk in analyzed_risks])
        
        # Construct a prompt for the AI
        prompt = f"""
        Based on the following analyzed risks and project data:
        
        {risks_text}
        
        Project Context:
        {project_context}
        
        Generate mitigation strategies for each risk. For each risk (use the same Risk IDs as provided), provide:
        
        - Mitigation Strategy: [a concise description of the strategy]
        - Action Plan: [2-3 specific steps to implement the strategy]
        - Resources Required: [list of resources needed to implement the strategy]
        - Expected Outcome: [description of the expected outcome of the strategy]
        
        Format your response with the Risk ID followed by only these four pieces of information for each risk.
        """
        
        try:
            # Use Anthropic to generate mitigation strategies
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.1,
                system="You are a risk management expert who specializes in developing mitigation strategies for project, portfolio, and program risks.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            strategies_text = message.content[0].text
            
            # Parse the strategies and update risks
            mitigated_risks = analyzed_risks.copy()
            current_risk_id = None
            
            for line in strategies_text.split('\n'):
                line = line.strip()
                
                if line.startswith('Risk ID:'):
                    risk_id = line.replace('Risk ID:', '').strip()
                    current_risk_id = risk_id
                elif line.startswith('- Mitigation Strategy:') and current_risk_id:
                    # Find the risk with the matching ID
                    for risk in mitigated_risks:
                        if risk['id'] == current_risk_id:
                            risk['mitigation_strategy'] = line.replace('- Mitigation Strategy:', '').strip()
                elif line.startswith('- Action Plan:') and current_risk_id:
                    for risk in mitigated_risks:
                        if risk['id'] == current_risk_id:
                            risk['action_plan'] = line.replace('- Action Plan:', '').strip()
                elif line.startswith('- Resources Required:') and current_risk_id:
                    for risk in mitigated_risks:
                        if risk['id'] == current_risk_id:
                            risk['resources_required'] = line.replace('- Resources Required:', '').strip()
                elif line.startswith('- Expected Outcome:') and current_risk_id:
                    for risk in mitigated_risks:
                        if risk['id'] == current_risk_id:
                            risk['expected_outcome'] = line.replace('- Expected Outcome:', '').strip()
            
            return mitigated_risks
            
        except Exception as e:
            # Return the analyzed risks with error info
            for risk in analyzed_risks:
                risk['mitigation_strategy'] = "Not available due to error"
                risk['action_plan'] = "Not available"
                risk['resources_required'] = "Not available"
                risk['expected_outcome'] = "Not available"
                risk['mitigation_error'] = str(e)
            return analyzed_risks
    
    def _generate_risk_summary(self, mitigated_risks: List[Dict[str, Any]]) -> str:
        """Generate a summary of the risks and mitigation strategies
        
        Args:
            mitigated_risks: List of risks with mitigation strategies
            
        Returns:
            Summary of the risks and mitigation strategies
        """
        # Count risks by category
        categories = {}
        for risk in mitigated_risks:
            category = risk.get('category', 'Uncategorized')
            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1
        
        # Count risks by severity (high, medium, low)
        severity = {'High': 0, 'Medium': 0, 'Low': 0}
        for risk in mitigated_risks:
            impact = risk.get('impact', '')
            if isinstance(impact, str):
                if 'high' in impact.lower():
                    severity['High'] += 1
                elif 'medium' in impact.lower():
                    severity['Medium'] += 1
                elif 'low' in impact.lower():
                    severity['Low'] += 1
        
        # Generate summary text
        summary = f"Risk Assessment Summary\n\n"
        summary += f"Total Risks Identified: {len(mitigated_risks)}\n\n"
        
        summary += "Risks by Category:\n"
        for category, count in categories.items():
            summary += f"- {category}: {count}\n"
        
        summary += "\nRisks by Severity:\n"
        for sev, count in severity.items():
            summary += f"- {sev}: {count}\n"
        
        summary += "\nTop Risks (by Risk Score):\n"
        # Sort risks by risk score (highest first)
        sorted_risks = sorted(mitigated_risks, key=lambda x: x.get('risk_score', 0) if isinstance(x.get('risk_score'), (int, float)) else 0, reverse=True)
        for i, risk in enumerate(sorted_risks[:3], 1):
            summary += f"{i}. {risk.get('title', 'Unnamed Risk')} (Score: {risk.get('risk_score', 'N/A')})\n"
            summary += f"   Mitigation: {risk.get('mitigation_strategy', 'Not provided')}\n"
        
        return summary
    
    def query_risks_from_db(self, project_id: str = None, risk_category: str = None, min_risk_score: float = None) -> pd.DataFrame:
        """Query risks from the database with filtering options
        
        Args:
            project_id: Optional project ID to filter by
            risk_category: Optional risk category to filter by
            min_risk_score: Optional minimum risk score to filter by
            
        Returns:
            DataFrame with risks matching the criteria
        """
        # Build SQL query with filters
        query = "SELECT r.*, m.strategy, m.action_plan FROM risks r "
        query += "LEFT JOIN mitigation_strategies m ON r.id = m.risk_id "
        
        where_clauses = []
        params = []
        
        if project_id:
            where_clauses.append("r.project_id = ?")
            params.append(project_id)
        
        if risk_category:
            where_clauses.append("r.category = ?")
            params.append(risk_category)
        
        if min_risk_score is not None:
            where_clauses.append("r.risk_score >= ?")
            params.append(min_risk_score)
        
        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses)
        
        query += " ORDER BY r.risk_score DESC"
        
        # Execute query
        try:
            # Use pandas to execute SQL
            conn = sqlite3.connect(self.db_manager.db_path)
            result = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return result
        except Exception as e:
            print(f"Error querying risks from database: {e}")
            return pd.DataFrame()
    
    def execute_suql_risk_query(self, suql_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a SUQL query specifically for risk analysis
        
        Args:
            suql_query: SUQL query string
            context: Optional context for the query
            
        Returns:
            Query results with risk analysis
        """
        from database.suql import SUQLProcessor
        
        # Create SUQL processor if needed
        suql_processor = SUQLProcessor(self.db_manager)
        
        # Process the query
        results = suql_processor.process_query(suql_query, context)
        
        # Add risk-specific processing here if needed
        
        return results
