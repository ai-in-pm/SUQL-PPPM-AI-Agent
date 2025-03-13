import os
import json
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from openai import OpenAI
from agents.base_agent import BaseAgent
from utils.config import OPENAI_API_KEY
from database.db_manager import DatabaseManager

class DataExtractionAgent(BaseAgent):
    """
    Agent responsible for extracting and preparing both structured and unstructured data
    from various project sources.
    
    This agent uses OpenAI's capabilities to process documents, extract information,
    and prepare data for the query processing agent to support Structured Unstructured
    Query Language (SUQL) operations.
    """
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        super().__init__(
            name="Data Extraction Agent",
            description="Extracts and prepares both structured and unstructured data from various project sources"
        )
        self.client = None
        self.structured_data = {}
        self.unstructured_data = {}
        self.json_data = {}
        
        # Initialize database manager
        if db_manager is None:
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = db_manager
    
    def _initialize_resources(self):
        """
        Initialize OpenAI client and data storage structures
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data sources and extract structured and unstructured data
        
        Args:
            input_data: Dictionary containing data source information
                - 'structured_sources': List of paths to structured data files (CSV, Excel, etc.)
                - 'unstructured_sources': List of paths to unstructured data files (text, PDF, etc.)
                - 'json_sources': List of paths to JSON data files
                - 'json_data': Direct JSON data objects to process
                - 'project_id': Optional project ID for database operations
                - 'store_documents': Whether to store extracted documents in the database
        
        Returns:
            Dictionary with processed structured and unstructured data
        """
        if not self.is_initialized:
            self.initialize()
        
        result = {
            "structured_data": {},
            "unstructured_data": {},
            "json_data": {}
        }
        
        # Get project ID for database operations
        project_id = input_data.get('project_id', None)
        store_documents = input_data.get('store_documents', False)
        
        # Process structured data sources (e.g., CSV, Excel files)
        if 'structured_sources' in input_data:
            for source in input_data['structured_sources']:
                extracted_data = self._extract_structured_data(source)
                result["structured_data"].update(extracted_data)
                
                # Store structured data in database if requested
                if project_id and store_documents:
                    for table_name, df in extracted_data.items():
                        self._store_structured_data(project_id, table_name, df)
        
        # Process unstructured data sources (e.g., documents, text files)
        if 'unstructured_sources' in input_data:
            for source in input_data['unstructured_sources']:
                extracted_data = self._extract_unstructured_data(source)
                result["unstructured_data"].update(extracted_data)
                
                # Store unstructured documents in database if requested
                if project_id and store_documents:
                    for doc_name, content in extracted_data.items():
                        self._store_unstructured_document(project_id, doc_name, content, os.path.basename(source))
        
        # Process JSON data from files
        if 'json_sources' in input_data:
            for source in input_data['json_sources']:
                extracted_data = self._extract_json_data_from_file(source)
                result["json_data"].update(extracted_data)
                
                # Store JSON data in database if requested
                if project_id and store_documents:
                    for data_name, json_content in extracted_data.items():
                        self._store_json_document(project_id, data_name, json_content, os.path.basename(source))
        
        # Process direct JSON data objects
        if 'json_data' in input_data:
            for data_name, json_obj in input_data['json_data'].items():
                processed_json = self._process_json_data(data_name, json_obj)
                result["json_data"].update(processed_json)
                
                # Store direct JSON data in database if requested
                if project_id and store_documents:
                    for json_name, json_content in processed_json.items():
                        self._store_json_document(project_id, json_name, json_content, "direct_input")
        
        self.structured_data.update(result["structured_data"])
        self.unstructured_data.update(result["unstructured_data"])
        self.json_data.update(result["json_data"])
        
        return result

    def _store_structured_data(self, project_id: str, table_name: str, df: pd.DataFrame) -> None:
        """Store structured data in the database
        
        Args:
            project_id: Project ID for database relation
            table_name: Name of the table/dataset
            df: DataFrame with the structured data
        """
        try:
            # Convert DataFrame to a document format
            document = {
                "project_id": project_id,
                "title": f"Structured Data: {table_name}",
                "content": df.to_json(orient="records"),
                "document_type": "structured_data",
                "metadata": json.dumps({
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "table_name": table_name
                })
            }
            
            # Add document to database
            self.db_manager.add_document(document)
            
        except Exception as e:
            print(f"Error storing structured data in database: {e}")
    
    def _store_unstructured_document(self, project_id: str, doc_name: str, content: str, source_filename: str) -> None:
        """Store unstructured document in the database
        
        Args:
            project_id: Project ID for database relation
            doc_name: Name of the document
            content: Text content of the document
            source_filename: Original filename
        """
        try:
            # Create document entry
            document = {
                "project_id": project_id,
                "title": doc_name,
                "content": content,
                "document_type": "unstructured",
                "metadata": json.dumps({
                    "source_file": source_filename,
                    "character_count": len(content)
                })
            }
            
            # Add document to database
            self.db_manager.add_document(document)
            
        except Exception as e:
            print(f"Error storing unstructured document in database: {e}")
    
    def _store_json_document(self, project_id: str, doc_name: str, content: Any, source_filename: str) -> None:
        """Store JSON document in the database
        
        Args:
            project_id: Project ID for database relation
            doc_name: Name of the document
            content: JSON content (will be converted to string)
            source_filename: Original filename or source
        """
        try:
            # Convert JSON to string if it's not already
            if not isinstance(content, str):
                content_str = json.dumps(content)
            else:
                content_str = content
            
            # Create document entry
            document = {
                "project_id": project_id,
                "title": f"JSON Data: {doc_name}",
                "content": content_str,
                "document_type": "json",
                "metadata": json.dumps({
                    "source_file": source_filename,
                    "data_name": doc_name
                })
            }
            
            # Add document to database
            self.db_manager.add_document(document)
            
        except Exception as e:
            print(f"Error storing JSON document in database: {e}")

    def _extract_structured_data(self, source: str) -> Dict[str, pd.DataFrame]:
        """
        Extract structured data from a given source file
        
        Args:
            source: Path to the structured data file
        
        Returns:
            Dictionary with table name as key and pandas DataFrame as value
        """
        result = {}
        
        try:
            file_extension = os.path.splitext(source)[1].lower()
            
            if file_extension == '.csv':
                # Extract table name from filename
                table_name = os.path.basename(source).split('.')[0]
                result[table_name] = pd.read_csv(source)
                
            elif file_extension in ['.xls', '.xlsx']:
                # Read all sheets in Excel file
                excel_data = pd.read_excel(source, sheet_name=None)
                for sheet_name, df in excel_data.items():
                    result[sheet_name] = df
            
            print(f"Extracted structured data from {source}: {list(result.keys())}")
            
        except Exception as e:
            print(f"Error extracting structured data from {source}: {str(e)}")
        
        return result
    
    def _extract_unstructured_data(self, source: str) -> Dict[str, str]:
        """
        Extract unstructured data from a given source file
        
        Args:
            source: Path to the unstructured data file
        
        Returns:
            Dictionary with document name as key and text content as value
        """
        result = {}
        
        try:
            file_extension = os.path.splitext(source)[1].lower()
            doc_name = os.path.basename(source).split('.')[0]
            
            if file_extension in ['.txt', '.md', '.rst']:
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                result[doc_name] = content
                
            elif file_extension == '.pdf':
                # For this implementation, we're assuming a simple text extraction
                # In a real implementation, you would use a PDF parsing library
                # and possibly OCR for scanned documents
                result[doc_name] = f"PDF content from {source} (would be extracted with proper PDF tools)"
                
            elif file_extension in ['.doc', '.docx']:
                # For this implementation, we're assuming a simple placeholder
                # In a real implementation, you would use a Word document parsing library
                result[doc_name] = f"Word document content from {source} (would be extracted with proper tools)"
            
            print(f"Extracted unstructured data from {source}: {doc_name}")
            
        except Exception as e:
            print(f"Error extracting unstructured data from {source}: {str(e)}")
        
        return result
    
    def _extract_json_data_from_file(self, source: str) -> Dict[str, Any]:
        """
        Extract and process JSON data from a file
        
        Args:
            source: Path to the JSON file
        
        Returns:
            Dictionary with dataset name as key and processed JSON as value
        """
        result = {}
        
        try:
            data_name = os.path.basename(source).split('.')[0]
            
            with open(source, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Process the loaded JSON data
            processed_data = self._process_json_data(data_name, json_data)
            result.update(processed_data)
            
            print(f"Extracted JSON data from {source}: {data_name}")
            
        except Exception as e:
            print(f"Error extracting JSON data from {source}: {str(e)}")
        
        return result
    
    def _process_json_data(self, data_name: str, json_data: Any) -> Dict[str, Any]:
        """
        Process JSON data into a format suitable for SUQL querying
        
        Args:
            data_name: Name to identify this JSON dataset
            json_data: The JSON data object (can be dict, list, or other JSON structure)
        
        Returns:
            Dictionary with dataset name as key and processed JSON as value
        """
        result = {}
        
        try:
            # For simple flat dictionaries or lists of flat dictionaries,
            # convert directly to DataFrame for easier SQL-like querying
            if isinstance(json_data, dict) and all(not isinstance(v, (dict, list)) for v in json_data.values()):
                # Simple flat dictionary - convert to single-row DataFrame
                result[data_name] = pd.DataFrame([json_data])
                result[f"{data_name}_raw"] = json_data  # Also keep the raw data
                print(f"Processed {data_name} as flat dictionary")
                
            elif isinstance(json_data, list) and all(isinstance(item, dict) and all(not isinstance(v, (dict, list)) for v in item.values()) for item in json_data):
                # List of flat dictionaries - convert to DataFrame
                result[data_name] = pd.DataFrame(json_data)
                result[f"{data_name}_raw"] = json_data  # Also keep the raw data
                print(f"Processed {data_name} as list of flat dictionaries")
                
            else:
                # Complex nested structure - extract table-like structures and keep raw data
                result[f"{data_name}_raw"] = json_data
                
                # Try to identify and extract table-like structures
                extracted_tables = self._extract_table_structures(data_name, json_data)
                result.update(extracted_tables)
                
                # Use AI to help analyze and extract relevant information
                extracted_info = self._extract_info_with_ai(data_name, json_data)
                result.update(extracted_info)
                
                print(f"Processed {data_name} as complex JSON structure")
        except Exception as e:
            print(f"Error processing JSON data {data_name}: {str(e)}")
            # Store the raw data at minimum
            result[f"{data_name}_raw"] = json_data
        
        return result
    
    def _extract_table_structures(self, base_name: str, json_data: Any, path: str = "") -> Dict[str, pd.DataFrame]:
        """
        Recursively extract table-like structures from complex nested JSON
        
        Args:
            base_name: Base name for the tables
            json_data: The JSON data to process
            path: Current path in the JSON structure (for recursive calls)
        
        Returns:
            Dictionary with table names as keys and DataFrames as values
        """
        tables = {}
        
        # Handle lists of dictionaries (common table-like structure)
        if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
            # Try to normalize the list into a DataFrame
            try:
                table_name = f"{base_name}_{path}" if path else base_name
                tables[table_name] = pd.json_normalize(json_data)
            except Exception as e:
                print(f"Could not normalize {base_name} at {path}: {str(e)}")
        
        # Recursively process nested dictionaries and lists
        elif isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = f"{path}_{key}" if path else key
                if isinstance(value, (dict, list)):
                    nested_tables = self._extract_table_structures(base_name, value, new_path)
                    tables.update(nested_tables)
                    
                    # If this is a dictionary with list values that look like tables, extract them
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        try:
                            table_name = f"{base_name}_{new_path}"
                            tables[table_name] = pd.json_normalize(value)
                        except Exception:
                            pass  # Silently skip if normalization fails
        
        return tables
    
    def _extract_info_with_ai(self, data_name: str, json_data: Any) -> Dict[str, Any]:
        """
        Use AI to extract structured information from complex JSON
        
        Args:
            data_name: Name of the dataset
            json_data: The JSON data to analyze
        
        Returns:
            Dictionary with extracted information
        """
        result = {}
        
        try:
            # Limit the size of the JSON to analyze to avoid token limits
            json_str = json.dumps(json_data)
            if len(json_str) > 10000:  # Simplified size check
                json_str = json_str[:10000] + "... [truncated]"
            
            # Ask the AI to identify key entities and relationships in the JSON
            prompt = f"""Analyze this JSON data structure and identify key entities that could be represented as tables for SQL-like querying.
            For each entity, list the key attributes and how they relate to other entities.
            Also identify any text fields that might contain valuable unstructured information for natural language queries.
            
            JSON data (may be truncated):
            {json_str}
            
            Provide your analysis in a structured format:
            1. Key entities identified (table-like structures)
            2. For each entity, list its attributes and data types
            3. Relationships between entities
            4. Text fields suitable for natural language analysis
            """
            
            # Get analysis from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data analyst specializing in JSON data structures. Your task is to analyze complex JSON and identify structured and unstructured components for SQL and NLP operations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Store the AI analysis as a separate unstructured document
            result[f"{data_name}_analysis"] = response.choices[0].message.content
            
        except Exception as e:
            print(f"Error analyzing JSON with AI for {data_name}: {str(e)}")
        
        return result
    
    def get_structured_data(self, table_name=None):
        """
        Get structured data for a specific table or all tables
        """
        if table_name:
            return self.structured_data.get(table_name)
        return self.structured_data
    
    def get_unstructured_data(self, doc_name=None):
        """
        Get unstructured data for a specific document or all documents
        """
        if doc_name:
            return self.unstructured_data.get(doc_name)
        return self.unstructured_data
        
    def get_json_data(self, data_name=None):
        """
        Get JSON data for a specific dataset or all datasets
        """
        if data_name:
            return self.json_data.get(data_name)
        return self.json_data
