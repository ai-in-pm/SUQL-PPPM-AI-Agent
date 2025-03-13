#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Manager for Structured Unstructured Query Language (SUQL) PPM AI Agent System

This module provides database functionality using SQLite and implements
Retrieval Augmented Generation (RAG) capabilities for all agents in the system.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import datetime
import logging

# For RAG functionality
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """A class to manage SQLite database operations and RAG functionality for SUQL-PPM AI Agent System."""
    
    def __init__(self, db_path: str = None):
        """Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file. If None, a default path will be used.
        """
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if db_path is None:
            self.db_path = self.base_dir / "data" / "suql_ppm.db"
        else:
            self.db_path = Path(db_path)
            
        # Create parent directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the database
        self._initialize_db()
        
        # Initialize RAG components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Vector store for RAG
        self.vector_stores = {}
        
    def _initialize_db(self):
        """Initialize the database with necessary tables."""
        try:
            # Connect to the database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create projects table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    start_date TEXT,
                    planned_end_date TEXT,
                    actual_end_date TEXT,
                    status TEXT,
                    completion_percentage REAL,
                    created_at TEXT,
                    updated_at TEXT
                )
                ''')
                
                # Create risks table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS risks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    probability REAL,
                    impact REAL,
                    risk_score REAL,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
                ''')
                
                # Create mitigation_strategies table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS mitigation_strategies (
                    id TEXT PRIMARY KEY,
                    risk_id TEXT,
                    strategy TEXT NOT NULL,
                    action_plan TEXT,
                    owner TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (risk_id) REFERENCES risks(id)
                )
                ''')
                
                # Create documents table for RAG
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    document_type TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
                ''')
                
                # Create embeddings table for RAG
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    embedding_file TEXT,
                    created_at TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
                ''')
                
                # Create SUQL queries history table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS suql_queries (
                    id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    result_summary TEXT,
                    executed_at TEXT
                )
                ''')
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def insert_project(self, project_data: Dict[str, Any]) -> str:
        """Insert a new project into the database.
        
        Args:
            project_data: Dictionary containing project data
            
        Returns:
            Project ID
        """
        try:
            # Generate a project ID if not provided
            if 'id' not in project_data:
                project_data['id'] = f"PRJ-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                
            # Add timestamps
            current_time = datetime.datetime.now().isoformat()
            project_data['created_at'] = current_time
            project_data['updated_at'] = current_time
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare the query and values
                columns = ', '.join(project_data.keys())
                placeholders = ', '.join(['?'] * len(project_data))
                values = tuple(project_data.values())
                
                # Execute the query
                cursor.execute(f"INSERT INTO projects ({columns}) VALUES ({placeholders})", values)
                conn.commit()
                
                logger.info(f"Project inserted: {project_data['id']}")
                return project_data['id']
        except sqlite3.Error as e:
            logger.error(f"Error inserting project: {e}")
            raise
    
    def insert_risk(self, risk_data: Dict[str, Any]) -> str:
        """Insert a new risk into the database.
        
        Args:
            risk_data: Dictionary containing risk data
            
        Returns:
            Risk ID
        """
        try:
            # Generate a risk ID if not provided
            if 'id' not in risk_data:
                risk_data['id'] = f"RISK-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                
            # Add timestamps
            current_time = datetime.datetime.now().isoformat()
            risk_data['created_at'] = current_time
            risk_data['updated_at'] = current_time
            
            # Calculate risk score if not provided
            if 'risk_score' not in risk_data and 'probability' in risk_data and 'impact' in risk_data:
                risk_data['risk_score'] = float(risk_data['probability']) * float(risk_data['impact'])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare the query and values
                columns = ', '.join(risk_data.keys())
                placeholders = ', '.join(['?'] * len(risk_data))
                values = tuple(risk_data.values())
                
                # Execute the query
                cursor.execute(f"INSERT INTO risks ({columns}) VALUES ({placeholders})", values)
                conn.commit()
                
                logger.info(f"Risk inserted: {risk_data['id']}")
                return risk_data['id']
        except sqlite3.Error as e:
            logger.error(f"Error inserting risk: {e}")
            raise
    
    def insert_mitigation_strategy(self, strategy_data: Dict[str, Any]) -> str:
        """Insert a new mitigation strategy into the database.
        
        Args:
            strategy_data: Dictionary containing mitigation strategy data
            
        Returns:
            Strategy ID
        """
        try:
            # Generate a strategy ID if not provided
            if 'id' not in strategy_data:
                strategy_data['id'] = f"STRAT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                
            # Add timestamps
            current_time = datetime.datetime.now().isoformat()
            strategy_data['created_at'] = current_time
            strategy_data['updated_at'] = current_time
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare the query and values
                columns = ', '.join(strategy_data.keys())
                placeholders = ', '.join(['?'] * len(strategy_data))
                values = tuple(strategy_data.values())
                
                # Execute the query
                cursor.execute(f"INSERT INTO mitigation_strategies ({columns}) VALUES ({placeholders})", values)
                conn.commit()
                
                logger.info(f"Mitigation strategy inserted: {strategy_data['id']}")
                return strategy_data['id']
        except sqlite3.Error as e:
            logger.error(f"Error inserting mitigation strategy: {e}")
            raise
    
    def get_project_by_id(self, project_id: str) -> Dict[str, Any]:
        """Retrieve a project by its ID.
        
        Args:
            project_id: The ID of the project to retrieve
            
        Returns:
            Project data as a dictionary
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert query results to dictionary
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
                result = cursor.fetchone()
                
                if result:
                    return dict(result)
                else:
                    logger.warning(f"Project not found: {project_id}")
                    return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving project: {e}")
            raise
    
    def get_risks_by_project_id(self, project_id: str) -> List[Dict[str, Any]]:
        """Retrieve all risks for a project.
        
        Args:
            project_id: The ID of the project
            
        Returns:
            List of risk dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert query results to dictionary
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM risks WHERE project_id = ?", (project_id,))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving risks: {e}")
            raise
    
    def execute_suql_query(self, query: str) -> Dict[str, Any]:
        """Execute a SUQL query and store it in the history.
        
        Args:
            query: The SUQL query to execute
            
        Returns:
            Query results and metadata
        """
        # Store the query in history
        query_id = f"QUERY-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        executed_at = datetime.datetime.now().isoformat()
        
        try:
            # Store the query without results initially
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO suql_queries (id, query_text, executed_at) VALUES (?, ?, ?)",
                    (query_id, query, executed_at)
                )
                conn.commit()
                
            # Return a placeholder for now - this will be processed by the Query Processing Agent
            return {
                "query_id": query_id,
                "status": "submitted",
                "executed_at": executed_at
            }
        except sqlite3.Error as e:
            logger.error(f"Error storing SUQL query: {e}")
            raise
    
    # ======= RAG Functionality ======= #
    
    def add_document(self, document_data: Dict[str, Any]) -> str:
        """Add a document to the database and process it for RAG.
        
        Args:
            document_data: Document data including content, title, and metadata
            
        Returns:
            Document ID
        """
        try:
            # Generate a document ID if not provided
            if 'id' not in document_data:
                document_data['id'] = f"DOC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                
            # Add timestamps
            current_time = datetime.datetime.now().isoformat()
            document_data['created_at'] = current_time
            document_data['updated_at'] = current_time
            
            # Convert metadata to JSON if it's a dictionary
            if 'metadata' in document_data and isinstance(document_data['metadata'], dict):
                document_data['metadata'] = json.dumps(document_data['metadata'])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare the query and values
                columns = ', '.join(document_data.keys())
                placeholders = ', '.join(['?'] * len(document_data))
                values = tuple(document_data.values())
                
                # Execute the query
                cursor.execute(f"INSERT INTO documents ({columns}) VALUES ({placeholders})", values)
                conn.commit()
                
                logger.info(f"Document inserted: {document_data['id']}")
                
                # Process document for RAG
                self._process_document_for_rag(document_data)
                
                return document_data['id']
        except sqlite3.Error as e:
            logger.error(f"Error inserting document: {e}")
            raise
    
    def _process_document_for_rag(self, document_data: Dict[str, Any]):
        """Process a document for RAG by creating embeddings.
        
        Args:
            document_data: Document data including content and metadata
        """
        try:
            # Split the document into chunks
            text_chunks = self.text_splitter.split_text(document_data['content'])
            
            # Create metadata for each chunk
            metadata = {}
            if 'project_id' in document_data:
                metadata['project_id'] = document_data['project_id']
            if 'title' in document_data:
                metadata['title'] = document_data['title']
            if 'document_type' in document_data:
                metadata['document_type'] = document_data['document_type']
            
            # Create document objects for the vector store
            documents = [Document(page_content=chunk, metadata=metadata) for chunk in text_chunks]
            
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create vector store folder if it doesn't exist
            vector_store_dir = self.base_dir / "data" / "vector_stores"
            vector_store_dir.mkdir(parents=True, exist_ok=True)
            
            # Save vector store with document ID as name
            vector_store_path = vector_store_dir / document_data['id']
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(str(vector_store_path))
            
            # Store embedding information in the database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store each chunk and its embedding file reference
                for i, chunk in enumerate(text_chunks):
                    embedding_id = f"EMB-{document_data['id']}-{i}"
                    cursor.execute(
                        "INSERT INTO embeddings (id, document_id, chunk_index, chunk_text, embedding_file, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (embedding_id, document_data['id'], i, chunk, str(vector_store_path), datetime.datetime.now().isoformat())
                    )
                
                conn.commit()
                
            logger.info(f"Document processed for RAG: {document_data['id']}")
        except Exception as e:
            logger.error(f"Error processing document for RAG: {e}")
            raise
    
    def perform_rag_search(self, query: str, project_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform a RAG search on the document embeddings.
        
        Args:
            query: The search query
            project_id: Optional project ID to filter results
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Get all vector store paths
            vector_store_dir = self.base_dir / "data" / "vector_stores"
            
            # If the directory doesn't exist yet, return empty results
            if not vector_store_dir.exists():
                logger.warning("No vector stores found for RAG search")
                return []
            
            # Filter by project ID if provided
            if project_id:
                # Get document IDs for the project
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM documents WHERE project_id = ?", (project_id,))
                    document_ids = [row['id'] for row in cursor.fetchall()]
                
                # Get vector store paths for these documents
                vector_store_paths = [vector_store_dir / doc_id for doc_id in document_ids if (vector_store_dir / doc_id).exists()]
            else:
                # Get all vector store paths
                vector_store_paths = [p for p in vector_store_dir.iterdir() if p.is_dir()]
            
            if not vector_store_paths:
                logger.warning(f"No vector stores found for project_id: {project_id}")
                return []
            
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            
            all_results = []
            
            # Search each vector store
            for vs_path in vector_store_paths:
                try:
                    # Load the vector store
                    vector_store = FAISS.load_local(str(vs_path), embeddings)
                    
                    # Search
                    results = vector_store.similarity_search_with_score(query, k=top_k)
                    
                    # Format results
                    for doc, score in results:
                        all_results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "relevance_score": float(score),
                            "vector_store": vs_path.name
                        })
                except Exception as e:
                    logger.error(f"Error searching vector store {vs_path}: {e}")
                    continue
            
            # Sort by relevance score (descending)
            all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Return top_k results
            return all_results[:top_k]
        except Exception as e:
            logger.error(f"Error performing RAG search: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a raw SQL query against the database.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results as a pandas DataFrame
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Execute the query and return results as DataFrame
                result = pd.read_sql_query(query, conn)
                return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
