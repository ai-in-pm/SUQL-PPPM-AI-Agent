#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structured Unstructured Query Language (SUQL) Module

This module provides implementation of SUQL with RAG capabilities,
allowing agents to query both structured data in the SQLite database
and unstructured data through the RAG system.
"""

import re
import json
import sqlite3
import pandas as pd
import logging
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path

from database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SUQLProcessor:
    """A class to process SUQL queries, integrating SQL with RAG capabilities."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """Initialize the SUQL processor.
        
        Args:
            db_manager: Database manager instance
        """
        if db_manager is None:
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = db_manager
            
        # Special SUQL function patterns
        self.special_functions = {
            'SUMMARY': r'SUMMARY\(([^)]+)\)(?:\s+AS\s+(\w+))?',
            'ANSWER': r'ANSWER\(([^,]+),\s*[\'\"](.*?)[\'\"](,\s*{.*?})?\)(?:\s+AS\s+(\w+))?',
            'INSIGHTS': r'INSIGHTS\(([^,]+),\s*[\'\"](.*?)[\'\"](,\s*{.*?})?\)(?:\s+AS\s+(\w+))?',
            'CALCULATE': r'CALCULATE\(([^)]+)\)(?:\s+AS\s+(\w+))?',
            'RAG': r'RAG\([\'\"](.*?)[\'\"](,\s*([^)]+))?\)(?:\s+AS\s+(\w+))?'
        }
        
        # SQL keywords to identify query parts
        self.sql_keywords = {
            'SELECT': r'\bSELECT\b',
            'FROM': r'\bFROM\b',
            'WHERE': r'\bWHERE\b',
            'GROUP BY': r'\bGROUP\s+BY\b',
            'HAVING': r'\bHAVING\b',
            'ORDER BY': r'\bORDER\s+BY\b',
            'LIMIT': r'\bLIMIT\b'
        }
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a SUQL query and return the results.
        
        Args:
            query: The SUQL query to process
            context: Optional context information for the query
            
        Returns:
            Query results and metadata
        """
        try:
            # Parse the SUQL query
            parsed_query = self._parse_query(query)
            
            # Extract special functions
            special_functions = self._extract_special_functions(query)
            
            # Convert SUQL to SQL
            sql_query, params = self._convert_to_sql(parsed_query, special_functions)
            
            # Execute the SQL query to get base results
            base_results = self.db_manager.execute_query(sql_query)
            
            # Process special functions
            processed_results = self._process_special_functions(base_results, special_functions, context)
            
            return {
                "result": processed_results,
                "metadata": {
                    "original_query": query,
                    "sql_query": sql_query,
                    "special_functions": list(special_functions.keys())
                }
            }
        except Exception as e:
            logger.error(f"Error processing SUQL query: {e}")
            return {
                "result": None,
                "error": str(e),
                "metadata": {"original_query": query}
            }
    
    def _parse_query(self, query: str) -> Dict[str, str]:
        """Parse a SUQL query into its component parts.
        
        Args:
            query: The SUQL query to parse
            
        Returns:
            Dictionary of query parts
        """
        parts = {}
        
        # Find positions of SQL keywords
        positions = {}
        for keyword, pattern in self.sql_keywords.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                positions[keyword] = match.start()
        
        # Sort keywords by position in query
        sorted_keywords = sorted(positions.items(), key=lambda x: x[1])
        
        # Extract parts based on keyword positions
        for i, (keyword, pos) in enumerate(sorted_keywords):
            # Find end position (start of next keyword or end of string)
            end_pos = len(query)
            if i < len(sorted_keywords) - 1:
                end_pos = sorted_keywords[i + 1][1]
            
            # Extract part
            start_pos = pos + len(keyword.split()[0])  # Add length of first word in keyword
            parts[keyword] = query[start_pos:end_pos].strip()
        
        return parts
    
    def _extract_special_functions(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract special SUQL functions from the query.
        
        Args:
            query: The SUQL query
            
        Returns:
            Dictionary of special functions with their parameters
        """
        special_functions = {}
        
        for func_name, pattern in self.special_functions.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            
            func_instances = []
            for match in matches:
                groups = match.groups()
                
                if func_name == 'SUMMARY':
                    # SUMMARY(column) AS alias
                    column = groups[0].strip()
                    alias = groups[1].strip() if groups[1] else f"summary_{column}"
                    func_instances.append({
                        'column': column,
                        'alias': alias,
                        'full_match': match.group(0)
                    })
                
                elif func_name == 'ANSWER':
                    # ANSWER(column, 'question', {options}) AS alias
                    column = groups[0].strip()
                    question = groups[1].strip()
                    options_str = groups[2][1:].strip() if groups[2] else None
                    alias = groups[3].strip() if groups[3] else f"answer_{column}"
                    
                    options = {}
                    if options_str:
                        try:
                            options = json.loads(options_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid options format in ANSWER function: {options_str}")
                    
                    func_instances.append({
                        'column': column,
                        'question': question,
                        'options': options,
                        'alias': alias,
                        'full_match': match.group(0)
                    })
                
                elif func_name == 'INSIGHTS':
                    # INSIGHTS(column, 'question', {options}) AS alias
                    column = groups[0].strip()
                    question = groups[1].strip()
                    options_str = groups[2][1:].strip() if groups[2] else None
                    alias = groups[3].strip() if groups[3] else f"insights_{column}"
                    
                    options = {}
                    if options_str:
                        try:
                            options = json.loads(options_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid options format in INSIGHTS function: {options_str}")
                    
                    func_instances.append({
                        'column': column,
                        'question': question,
                        'options': options,
                        'alias': alias,
                        'full_match': match.group(0)
                    })
                
                elif func_name == 'CALCULATE':
                    # CALCULATE(expression) AS alias
                    expression = groups[0].strip()
                    alias = groups[1].strip() if groups[1] else f"calc_{expression.replace(' ', '_')}"
                    func_instances.append({
                        'expression': expression,
                        'alias': alias,
                        'full_match': match.group(0)
                    })
                
                elif func_name == 'RAG':
                    # RAG('query', filter_expr) AS alias
                    rag_query = groups[0].strip()
                    filter_expr = groups[2].strip() if groups[2] else None
                    alias = groups[3].strip() if groups[3] else f"rag_results"
                    func_instances.append({
                        'query': rag_query,
                        'filter': filter_expr,
                        'alias': alias,
                        'full_match': match.group(0)
                    })
            
            if func_instances:
                special_functions[func_name] = func_instances
        
        return special_functions
    
    def _convert_to_sql(self, parsed_query: Dict[str, str], special_functions: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, List[Any]]:
        """Convert a parsed SUQL query to SQL.
        
        Args:
            parsed_query: The parsed SUQL query
            special_functions: Extracted special functions
            
        Returns:
            Tuple of SQL query string and parameters
        """
        sql_parts = {}
        params = []
        
        # Process SELECT
        if 'SELECT' in parsed_query:
            select_part = parsed_query['SELECT']
            
            # Replace special functions with placeholders in SELECT
            for func_name, instances in special_functions.items():
                for instance in instances:
                    if func_name == 'CALCULATE':
                        # Replace CALCULATE with the expression directly
                        select_part = select_part.replace(
                            instance['full_match'],
                            f"({instance['expression']}) AS {instance['alias']}"
                        )
                    elif func_name == 'RAG':
                        # Replace RAG with NULL placeholder (will be processed later)
                        select_part = select_part.replace(
                            instance['full_match'],
                            f"NULL AS {instance['alias']}"
                        )
                    else:
                        # For other functions, just get the column
                        if func_name in ['SUMMARY', 'ANSWER', 'INSIGHTS']:
                            select_part = select_part.replace(
                                instance['full_match'],
                                f"{instance['column']} AS {instance['alias']}"
                            )
            
            sql_parts['SELECT'] = f"SELECT {select_part}"
        
        # Process other SQL parts directly
        for part in ['FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']:
            if part in parsed_query:
                # Handle RAG filter in WHERE clause
                if part == 'WHERE' and 'RAG' in special_functions:
                    # Process RAG filters here if needed
                    pass
                
                sql_parts[part] = f"{part} {parsed_query[part]}"
        
        # Combine all parts into a SQL query
        sql_query = " ".join([sql_parts.get(part, "") for part in [
            'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT'
        ]]).strip()
        
        return sql_query, params
    
    def _process_special_functions(self, base_results: pd.DataFrame, 
                                  special_functions: Dict[str, List[Dict[str, Any]]], 
                                  context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Process special SUQL functions on the base query results.
        
        Args:
            base_results: DataFrame of base query results
            special_functions: Special functions to process
            context: Optional context for the query
            
        Returns:
            DataFrame with processed special functions
        """
        results = base_results.copy()
        
        if results.empty:
            return results
        
        # Process RAG queries first
        if 'RAG' in special_functions:
            for instance in special_functions['RAG']:
                rag_query = instance['query']
                alias = instance['alias']
                filter_expr = instance['filter']
                
                # Get project_id from context or filter if available
                project_id = None
                if context and 'project_id' in context:
                    project_id = context['project_id']
                elif filter_expr and 'project_id' in filter_expr:
                    # Extract project_id from filter expression (simplified approach)
                    match = re.search(r'project_id\s*=\s*[\'\"](.*?)[\'\"](\W|$)', filter_expr)
                    if match:
                        project_id = match.group(1)
                
                # Perform RAG search
                rag_results = self.db_manager.perform_rag_search(rag_query, project_id)
                
                # Add RAG results as a JSON string in a new column
                results[alias] = json.dumps(rag_results) if rag_results else None
        
        # Process SUMMARY function
        if 'SUMMARY' in special_functions:
            for instance in special_functions['SUMMARY']:
                column = instance['column']
                alias = instance['alias']
                
                if column in results.columns:
                    # This is a placeholder - in a real implementation, 
                    # we would use an AI model to generate summaries
                    results[alias] = results[column].apply(
                        lambda x: f"Summary of {str(x)[:50]}..." if x and len(str(x)) > 50 else str(x)
                    )
        
        # Process ANSWER function
        if 'ANSWER' in special_functions:
            for instance in special_functions['ANSWER']:
                column = instance['column']
                question = instance['question']
                alias = instance['alias']
                
                if column in results.columns:
                    # This is a placeholder - in a real implementation, 
                    # we would use an AI model to answer questions about the content
                    results[alias] = results[column].apply(
                        lambda x: f"Answer to '{question}' based on text" if x else None
                    )
        
        # Process INSIGHTS function
        if 'INSIGHTS' in special_functions:
            for instance in special_functions['INSIGHTS']:
                column = instance['column']
                question = instance['question']
                alias = instance['alias']
                
                if column in results.columns:
                    # This is a placeholder - in a real implementation, 
                    # we would use an AI model to generate insights
                    results[alias] = results[column].apply(
                        lambda x: f"Insights for '{question}' based on data" if x else None
                    )
        
        return results
