import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from anthropic import Anthropic
from agents.base_agent import BaseAgent
from utils.config import ANTHROPIC_API_KEY

class QueryProcessingAgent(BaseAgent):
    """
    Agent responsible for processing Structured Unstructured Query Language (SUQL) queries.
    
    This agent handles queries that combine SQL-like syntax with natural language processing
    functions like SUMMARY() and ANSWER() for text analysis, allowing users to query both
    structured data (like databases and JSON) and unstructured data (like text documents)
    using a unified syntax.
    
    It uses Anthropic's Claude model to handle the unstructured data processing components.
    """
    def __init__(self):
        super().__init__(
            name="Query Processing Agent",
            description="Processes SUQL queries that combine SQL-like syntax with natural language processing"
        )
        self.client = None
        self.data_extraction_agent = None
    
    def _initialize_resources(self):
        """
        Initialize Anthropic client
        """
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    def connect_data_agent(self, data_extraction_agent):
        """
        Connect this agent to the Data Extraction Agent to access data sources
        """
        self.data_extraction_agent = data_extraction_agent
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a SUQL query and return the results
        
        Args:
            input_data: Dictionary containing:
                - 'query': The SUQL query to process
                - 'data_context': Optional context about where to look for data (json, structured, unstructured)
        
        Returns:
            Dictionary with query results
        """
        if not self.is_initialized:
            self.initialize()
            
        if self.data_extraction_agent is None:
            raise ValueError("Data Extraction Agent is not connected. Call connect_data_agent() first.")
        
        suql_query = input_data.get('query')
        if not suql_query:
            raise ValueError("No query provided")
        
        # Determine the data context - default is to look across all data sources
        data_context = input_data.get('data_context', 'all')
        
        # Parse the SUQL query
        parsed_query = self._parse_suql_query(suql_query)
        
        # Execute the query
        result = self._execute_query(parsed_query, data_context)
        
        return {
            "query": suql_query,
            "result": result
        }
    
    def _parse_suql_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a SUQL query into components for execution
        
        Args:
            query: The SUQL query string
            
        Returns:
            Dictionary containing parsed query components
        """
        # This is a simplified parser for demonstration purposes
        # In a real implementation, you would use a proper SQL parser with SUQL extensions
        
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
        select_clause = select_match.group(1).strip() if select_match else "*"
        
        # Extract FROM clause
        from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|\s*;|$)', query, re.IGNORECASE | re.DOTALL)
        from_clause = from_match.group(1).strip() if from_match else ""
        
        # Extract WHERE clause if present
        where_match = re.search(r'WHERE\s+(.+?)(?:;|$)', query, re.IGNORECASE | re.DOTALL)
        where_clause = where_match.group(1).strip() if where_match else ""
        
        # Extract SUMMARY and ANSWER functions from SELECT clause
        summary_functions = self._extract_functions(select_clause, "SUMMARY")
        answer_functions_select = self._extract_functions(select_clause, "ANSWER")
        
        # Extract ANSWER functions from WHERE clause
        answer_functions_where = self._extract_functions(where_clause, "ANSWER") if where_clause else []
        
        return {
            "select": select_clause,
            "from": from_clause,
            "where": where_clause,
            "summary_functions": summary_functions,
            "answer_functions_select": answer_functions_select,
            "answer_functions_where": answer_functions_where
        }
    
    def _extract_functions(self, clause: str, function_name: str) -> List[Dict[str, str]]:
        """
        Extract SUMMARY or ANSWER function calls from a SQL clause
        
        Args:
            clause: The SQL clause to search in
            function_name: The function name to look for ("SUMMARY" or "ANSWER")
            
        Returns:
            List of dictionaries, each containing:
                - field: The field/column the function is applied to
                - alias: The alias for the result (if any)
                - question: The question for ANSWER function (if applicable)
        """
        functions = []
        
        # Find all instances of the function
        pattern = fr'{function_name}\s*\(\s*(.+?)\s*\)(?:\s+AS\s+(\w+))?'
        if function_name == "ANSWER":
            # For ANSWER, extract both the field and the question
            pattern = fr'{function_name}\s*\(\s*(.+?)\s*,\s*[\'\"](.*?)[\'\"]\s*\)(?:\s+AS\s+(\w+))?'
            matches = re.finditer(pattern, clause, re.IGNORECASE)
            for match in matches:
                field = match.group(1).strip()
                question = match.group(2)
                alias = match.group(3) if len(match.groups()) > 2 and match.group(3) else f"{function_name.lower()}_{len(functions)}"
                functions.append({
                    "field": field,
                    "question": question,
                    "alias": alias
                })
        else:
            # For SUMMARY, just extract the field
            matches = re.finditer(pattern, clause, re.IGNORECASE)
            for match in matches:
                field = match.group(1).strip()
                alias = match.group(2) if match.group(2) else f"{function_name.lower()}_{len(functions)}"
                functions.append({
                    "field": field,
                    "alias": alias
                })
        
        return functions
    
    def _execute_query(self, parsed_query: Dict[str, Any], data_context: str = 'all') -> pd.DataFrame:
        """
        Execute the parsed SUQL query
        
        Args:
            parsed_query: Dictionary containing parsed query components
            data_context: Specify which data source to use ('all', 'structured', 'unstructured', 'json')
            
        Returns:
            DataFrame containing the query results
        """
        # Get the table/data name from the FROM clause
        table_name = parsed_query["from"]
        
        # Check different data sources based on data_context
        table_data = None
        source_type = None
        
        if data_context in ['all', 'structured']:
            # Try to get from structured data
            table_data = self.data_extraction_agent.get_structured_data(table_name)
            if table_data is not None:
                source_type = 'structured'
        
        if table_data is None and data_context in ['all', 'json']:
            # Try to get from JSON data
            table_data = self.data_extraction_agent.get_json_data(table_name)
            if table_data is not None:
                source_type = 'json'
                
                # Convert to DataFrame if it's not already
                if not isinstance(table_data, pd.DataFrame):
                    if isinstance(table_data, dict) and not any(isinstance(v, (dict, list)) for v in table_data.values()):
                        # Simple flat dictionary
                        table_data = pd.DataFrame([table_data])
                    elif isinstance(table_data, list) and all(isinstance(item, dict) for item in table_data):
                        # List of dictionaries
                        table_data = pd.DataFrame(table_data)
                    else:
                        # For complex nested JSON, we might need more sophisticated handling
                        # Here we'll just convert what we can and warn if it's too complex
                        try:
                            table_data = pd.json_normalize(table_data)
                        except Exception as e:
                            raise ValueError(f"Cannot query complex JSON data directly: {str(e)}. Try using a more specific table name.")
        
        # If still no data found, check if we need to handle unstructured data differently
        if table_data is None:
            # For an entirely text-based query on unstructured data (e.g., "FROM project_reports")
            if data_context in ['all', 'unstructured']:
                unstructured_data = self.data_extraction_agent.get_unstructured_data(table_name)
                if unstructured_data is not None:
                    # Create a simple DataFrame with the text content for processing
                    table_data = pd.DataFrame({"content": [unstructured_data]})
                    source_type = 'unstructured'
        
        if table_data is None:
            raise ValueError(f"Table or data source '{table_name}' not found in any data source")
        
        # Create a copy of the DataFrame to work with
        result_df = table_data.copy() if isinstance(table_data, pd.DataFrame) else None
        
        # For non-DataFrame JSON data, try to handle it appropriately
        if result_df is None and source_type == 'json':
            # Try to convert complex JSON to a suitable format for querying
            result_df = self._json_to_queryable_format(table_data, table_name)
        
        # Process WHERE clause with traditional SQL conditions and ANSWER functions
        if parsed_query["where"] and result_df is not None:
            result_df = self._process_where_clause(result_df, parsed_query["where"], parsed_query["answer_functions_where"])
        
        # Process SELECT clause
        if parsed_query["select"] != "*" and result_df is not None:
            result_df = self._process_select_clause(result_df, parsed_query["select"], 
                                             parsed_query["summary_functions"], 
                                             parsed_query["answer_functions_select"])
        
        return result_df
    
    def _json_to_queryable_format(self, json_data: Any, table_name: str) -> pd.DataFrame:
        """
        Convert complex JSON data to a format suitable for querying
        
        Args:
            json_data: The JSON data (could be dict, list, or complex nested structure)
            table_name: The name of the table/data being queried
            
        Returns:
            DataFrame representation of the JSON data
        """
        try:
            # For simple structures, use pandas normalization
            if isinstance(json_data, dict) and not any(isinstance(v, (dict, list)) for v in json_data.values()):
                return pd.DataFrame([json_data])
            elif isinstance(json_data, list) and all(isinstance(item, dict) for item in json_data):
                return pd.DataFrame(json_data)
            
            # For more complex structures, use AI to help extract a valid table structure
            json_str = json.dumps(json_data)
            if len(json_str) > 12000:  # Limit size for API call
                json_str = json_str[:12000] + "... [truncated]"
                
            prompt = f"""I have a complex JSON data structure that I want to query with SQL-like syntax. 
            The table name in my query is '{table_name}'. Please help me convert this JSON into a flat 
            table structure that would be most relevant for queries against '{table_name}'.
            
            JSON data (may be truncated):
            {json_str}
            
            Please output ONLY a Python dictionary that represents how to flatten this data into a tabular format 
            suitable for a DataFrame, with column names as keys and lists of values (maintaining original order) as values.
            
            For example, if table_name is 'projects' and the JSON represents project data, extract all relevant project fields.
            If the JSON contains nested objects or arrays that should be part of the table, flatten them appropriately.
            
            Output only the Python dictionary in the format:
            {{
                "column1": [value1, value2, ...],
                "column2": [value1, value2, ...],
                ...
            }}
            """
            
            # Use Anthropic to generate a flattened structure
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0,
                system="You are a data engineer specializing in transforming complex JSON data into tabular formats for SQL queries. You output only valid Python dictionaries that can be used to create a pandas DataFrame.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = message.content[0].text
            
            # Extract just the dictionary portion from the response
            dict_match = re.search(r'\{[\s\S]*\}', result_text)
            if dict_match:
                dict_str = dict_match.group(0)
                # Evaluate the dictionary string safely
                table_dict = eval(dict_str)  # In production, use ast.literal_eval for safety
                
                # Create DataFrame from the dictionary
                return pd.DataFrame(table_dict)
            
            # Fallback: try pandas normalization with error handling
            return pd.json_normalize(json_data)
            
        except Exception as e:
            print(f"Error converting JSON to queryable format: {str(e)}")
            # Last resort: create a single row with the raw JSON as a string
            return pd.DataFrame({"raw_json": [json.dumps(json_data)]})
    
    def _process_where_clause(self, df: pd.DataFrame, where_clause: str, answer_functions: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Process the WHERE clause including both traditional filters and ANSWER functions
        
        Args:
            df: DataFrame to filter
            where_clause: The WHERE clause from the query
            answer_functions: List of ANSWER functions in the WHERE clause
            
        Returns:
            Filtered DataFrame
        """
        # Process ANSWER functions in WHERE clause
        for answer_func in answer_functions:
            field = answer_func["field"]
            question = answer_func["question"]
            alias = answer_func["alias"]
            
            # Get the unstructured data for this field from various sources
            field_data = None
            
            # First check if it's already a column in the DataFrame
            if field in df.columns:
                # Apply the ANSWER function on each text value in the column
                df[alias] = df[field].apply(lambda text: self._process_answer(text, question) if pd.notna(text) else "")
            else:
                # Try to get from unstructured data
                doc_texts = self.data_extraction_agent.get_unstructured_data()
                json_data = self.data_extraction_agent.get_json_data()
                
                if field in doc_texts:
                    # For all rows, use the same unstructured document
                    field_data = doc_texts[field]
                    df[alias] = df.apply(lambda row: self._process_answer(field_data, question), axis=1)
                elif field in json_data:
                    # For JSON data, try to identify the relevant text fields for analysis
                    field_data = json_data[field]
                    if isinstance(field_data, dict):
                        # Find text fields in the dictionary that might contain relevant information
                        text_content = self._extract_text_from_json(field_data)
                        df[alias] = df.apply(lambda row: self._process_answer(text_content, question), axis=1)
                    else:
                        df[alias] = df.apply(lambda row: self._process_answer(str(field_data), question), axis=1)
                else:
                    # If field not found, create empty column
                    df[alias] = ""
            
            # Replace the ANSWER function call in WHERE clause with the column name
            pattern = fr'ANSWER\s*\({re.escape(field)}\s*,\s*[\'\"]{re.escape(question)}[\'\"](\s*\))'
            where_clause = re.sub(pattern, alias, where_clause, flags=re.IGNORECASE)
        
        # Apply the modified WHERE clause
        # This is a simplified implementation for basic conditions
        if "=" in where_clause:
            # Handle simple equality conditions
            try:
                col, val = where_clause.split("=", 1)
                col = col.strip()
                val = val.strip().strip("'").strip('"')
                return df[df[col] == val]
            except Exception as e:
                print(f"Error applying WHERE clause: {str(e)}")
        elif ">" in where_clause:
            # Handle simple greater than conditions
            try:
                col, val = where_clause.split(">", 1)
                col = col.strip()
                val = float(val.strip())
                return df[df[col] > val]
            except Exception as e:
                print(f"Error applying WHERE clause: {str(e)}")
        elif "<" in where_clause:
            # Handle simple less than conditions
            try:
                col, val = where_clause.split("<", 1)
                col = col.strip()
                val = float(val.strip())
                return df[df[col] < val]
            except Exception as e:
                print(f"Error applying WHERE clause: {str(e)}")
        
        # If we couldn't process the WHERE clause or there was an error, return the original DataFrame
        return df
    
    def _process_select_clause(self, df: pd.DataFrame, select_clause: str, 
                           summary_functions: List[Dict[str, str]], 
                           answer_functions: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Process the SELECT clause including both regular columns and SUMMARY/ANSWER functions
        
        Args:
            df: DataFrame to select from
            select_clause: The SELECT clause from the query
            summary_functions: List of SUMMARY functions in the SELECT clause
            answer_functions: List of ANSWER functions in the SELECT clause
            
        Returns:
            DataFrame with selected columns and function results
        """
        # Extract column names from SELECT clause (excluding function calls)
        columns = [col.strip() for col in re.sub(r'SUMMARY\([^)]+\)|ANSWER\([^)]+\)', '', select_clause).split(',')]
        columns = [col for col in columns if col]  # Remove empty strings
        
        # Process SUMMARY functions in SELECT clause
        for summary_func in summary_functions:
            field = summary_func["field"]
            alias = summary_func["alias"]
            
            # Check different data sources for the field
            if field in df.columns:
                # Apply SUMMARY to each value in the column
                df[alias] = df[field].apply(lambda text: self._process_summary(text) if pd.notna(text) else "")
            else:
                # Try to get from unstructured data or JSON
                doc_texts = self.data_extraction_agent.get_unstructured_data()
                json_data = self.data_extraction_agent.get_json_data()
                
                if field in doc_texts:
                    # For all rows, use the same unstructured document
                    field_data = doc_texts[field]
                    df[alias] = df.apply(lambda row: self._process_summary(field_data), axis=1)
                elif field in json_data:
                    # For JSON data, try to summarize relevant text fields
                    field_data = json_data[field]
                    if isinstance(field_data, dict):
                        text_content = self._extract_text_from_json(field_data)
                        df[alias] = df.apply(lambda row: self._process_summary(text_content), axis=1)
                    else:
                        df[alias] = df.apply(lambda row: self._process_summary(str(field_data)), axis=1)
                else:
                    # If field not found, create empty column
                    df[alias] = ""
            
            # Add the alias to the columns list if not already there
            if alias not in columns:
                columns.append(alias)
        
        # Process ANSWER functions in SELECT clause
        for answer_func in answer_functions:
            field = answer_func["field"]
            question = answer_func["question"]
            alias = answer_func["alias"]
            
            # Check different data sources for the field
            if field in df.columns:
                # Apply ANSWER to each value in the column
                df[alias] = df[field].apply(lambda text: self._process_answer(text, question) if pd.notna(text) else "")
            else:
                # Try to get from unstructured data or JSON
                doc_texts = self.data_extraction_agent.get_unstructured_data()
                json_data = self.data_extraction_agent.get_json_data()
                
                if field in doc_texts:
                    # For all rows, use the same unstructured document
                    field_data = doc_texts[field]
                    df[alias] = df.apply(lambda row: self._process_answer(field_data, question), axis=1)
                elif field in json_data:
                    # For JSON data, try to answer questions about relevant text fields
                    field_data = json_data[field]
                    if isinstance(field_data, dict):
                        text_content = self._extract_text_from_json(field_data)
                        df[alias] = df.apply(lambda row: self._process_answer(text_content, question), axis=1)
                    else:
                        df[alias] = df.apply(lambda row: self._process_answer(str(field_data), question), axis=1)
                else:
                    # If field not found, create empty column
                    df[alias] = ""
            
            # Add the alias to the columns list if not already there
            if alias not in columns:
                columns.append(alias)
        
        # Select only the requested columns (handling '*' case)
        if '*' in columns:
            # Keep all original columns plus the new function columns
            all_columns = list(df.columns)
            return df
        else:
            # Filter to only the requested columns (if they exist)
            valid_columns = [col for col in columns if col in df.columns]
            return df[valid_columns]
    
    def _process_summary(self, text: str) -> str:
        """
        Process a SUMMARY function by summarizing the given text
        
        Args:
            text: The text to summarize
            
        Returns:
            A summary of the text
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Use Anthropic's Claude to generate a summary
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                temperature=0,
                system="You are a helpful assistant that summarizes text concisely and accurately.",
                messages=[
                    {"role": "user", "content": f"Please summarize the following text in 1-2 sentences:\n\n{text}"}
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Error generating summary"
    
    def _process_answer(self, text: str, question: str) -> str:
        """
        Process an ANSWER function by answering a question about the given text
        
        Args:
            text: The text to analyze
            question: The question to answer
            
        Returns:
            An answer to the question based on the text
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Use Anthropic's Claude to generate an answer
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                temperature=0,
                system="You are a helpful assistant that answers questions accurately based solely on the provided text. Keep answers very brief.",
                messages=[
                    {"role": "user", "content": f"Text: {text}\n\nQuestion: {question}\n\nPlease answer the question based only on the information in the text above. Keep your answer very brief (preferably just yes/no or a few words)."}
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Error generating answer"
    
    def _extract_text_from_json(self, json_data: Any) -> str:
        """
        Extract text content from potentially complex JSON structures
        
        Args:
            json_data: The JSON data to extract text from
            
        Returns:
            Concatenated text content found in the JSON
        """
        text_content = []
        
        def extract_text(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # Look for fields that likely contain text content
                    if isinstance(value, str) and len(value) > 30 and key.lower() in [
                        'description', 'content', 'text', 'notes', 'summary', 'details', 'comments',
                        'report', 'feedback', 'analysis', 'narrative', 'minutes', 'transcript', 'documentation'
                    ]:
                        text_content.append(f"{key}: {value}")
                    elif isinstance(value, (dict, list)):
                        extract_text(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    if isinstance(item, (dict, list)):
                        extract_text(item, new_path)
                    elif isinstance(item, str) and len(item) > 30:
                        text_content.append(item)
        
        extract_text(json_data)
        return "\n\n".join(text_content)
