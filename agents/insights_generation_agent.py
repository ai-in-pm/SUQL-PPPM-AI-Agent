import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path
from anthropic import Anthropic
from agents.base_agent import BaseAgent
from utils.config import ANTHROPIC_API_KEY
from database.db_manager import DatabaseManager

class InsightsGenerationAgent(BaseAgent):
    """
    Agent responsible for generating insights and visualizations based on query results.
    
    This agent analyzes data from Structured Unstructured Query Language (SUQL) query results
    and generates actionable insights, recommendations, and visualizations to help with 
    decision making in PPPM contexts.
    
    It uses Anthropic's Claude model to generate natural language insights and
    matplotlib/seaborn for creating visualizations.
    """
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        super().__init__(
            name="Insights Generation Agent",
            description="Generates insights and visualizations from data analysis"
        )
        self.client = None
        self.output_dir = Path("outputs/insights")
        
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
        Process query results to generate insights and visualizations
        
        Args:
            input_data: Dictionary containing:
                - 'query_result': DataFrame or JSON data from query results
                - 'context': Optional context about the data domain (e.g., 'project', 'portfolio')
                - 'insight_type': Optional type of insights to focus on (e.g., 'risks', 'timeline', 'budget')
                - 'project_id': Optional project ID for database operations
                - 'use_rag': Optional boolean to use RAG for enhancing insights
                - 'rag_query': Optional query for RAG-based insight enhancement
                - 'store_insights': Optional boolean to store insights in the database
        
        Returns:
            Dictionary with insights and visualization paths
        """
        if not self.is_initialized:
            self.initialize()
        
        # Get data from input
        data = input_data.get('query_result')
        if data is None:
            raise ValueError("No query results provided for insight generation")
        
        # Convert to DataFrame if it's JSON or dict
        if not isinstance(data, pd.DataFrame):
            data = self._convert_to_dataframe(data)
        
        # Get optional parameters
        context = input_data.get('context', '')
        insight_type = input_data.get('insight_type', 'comprehensive')
        project_id = input_data.get('project_id', None)
        use_rag = input_data.get('use_rag', False)
        rag_query = input_data.get('rag_query', None)
        store_insights = input_data.get('store_insights', False)
        
        # Use RAG to enhance insight generation if requested
        rag_context = None
        if use_rag and rag_query:
            rag_results = self.db_manager.perform_rag_search(rag_query, project_id)
            if rag_results:
                rag_context = "\n\nRelevant project documents:\n"
                for i, result in enumerate(rag_results):
                    rag_context += f"Document {i+1}: {result.get('content', '')}\n"
        
        # Combine context with RAG context
        if rag_context:
            combined_context = f"{context}\n\n{rag_context}"
        else:
            combined_context = context
        
        # Generate insights
        insights = self._generate_insights(data, combined_context, insight_type)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(data, insights)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data, insights, combined_context)
        
        # Store insights in database if requested
        if project_id and store_insights:
            self._store_insights(project_id, insights, recommendations, visualizations)
        
        # Prepare result
        result = {
            "insights": insights,
            "recommendations": recommendations,
            "visualizations": visualizations,
            "summary": self._generate_summary(insights, recommendations)
        }
        
        return result
    
    def _store_insights(self, project_id: str, insights: List[Dict[str, Any]], 
                       recommendations: List[Dict[str, Any]], visualizations: Dict[str, str]) -> None:
        """Store generated insights in the database
        
        Args:
            project_id: Project ID for database relation
            insights: List of generated insights
            recommendations: List of recommendations
            visualizations: Dict of visualization paths
        """
        try:
            # Convert insights to document format
            insights_text = "\n\n".join([f"Insight {i+1}: {insight.get('title', '')}\n{insight.get('description', '')}" 
                                  for i, insight in enumerate(insights)])
            
            # Convert recommendations to text
            recommendations_text = "\n\n".join([f"Recommendation {i+1}: {rec.get('title', '')}\n{rec.get('description', '')}" 
                                        for i, rec in enumerate(recommendations)])
            
            # Combine all insights
            content = f"Insights:\n\n{insights_text}\n\nRecommendations:\n\n{recommendations_text}"
            
            # Create document entry
            document = {
                "project_id": project_id,
                "title": f"Generated Insights - {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                "content": content,
                "document_type": "insights",
                "metadata": json.dumps({
                    "insight_count": len(insights),
                    "recommendation_count": len(recommendations),
                    "visualization_count": len(visualizations),
                    "generation_date": pd.Timestamp.now().isoformat()
                })
            }
            
            # Add document to database
            self.db_manager.add_document(document)
            
        except Exception as e:
            print(f"Error storing insights in database: {e}")
    
    def fetch_historical_insights(self, project_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch historical insights from the database for a project
        
        Args:
            project_id: Project ID to fetch insights for
            limit: Maximum number of insight documents to retrieve
            
        Returns:
            List of insight documents
        """
        try:
            # Query for insight documents
            query = "SELECT * FROM documents WHERE project_id = ? AND document_type = 'insights' ORDER BY id DESC LIMIT ?"
            results = self.db_manager.query(query, (project_id, limit))
            
            return results
        except Exception as e:
            print(f"Error fetching historical insights: {e}")
            return []
    
    def compare_with_historical_insights(self, current_insights: List[Dict[str, Any]], 
                                        project_id: str) -> Dict[str, Any]:
        """Compare current insights with historical insights
        
        Args:
            current_insights: Current list of insights
            project_id: Project ID to fetch historical insights for
            
        Returns:
            Dictionary with comparative analysis
        """
        # Fetch historical insights
        historical_docs = self.fetch_historical_insights(project_id)
        
        if not historical_docs:
            return {"comparison": "No historical insights available for comparison"}
        
        # Extract insights from historical documents
        historical_insights = []
        for doc in historical_docs:
            historical_insights.append({
                "date": doc.get("creation_date", "Unknown"),
                "content": doc.get("content", "")
            })
        
        # Format current insights
        current_text = "\n\n".join([f"Insight: {insight.get('title', '')}\n{insight.get('description', '')}" 
                               for insight in current_insights])
        
        # Use AI to compare insights
        prompt = f"""
        Compare the current insights with historical insights for this project:
        
        Current Insights (Generated now):
        {current_text}
        
        Historical Insights:
        """
        
        for i, hist in enumerate(historical_insights):
            prompt += f"\n\nHistorical Insights from {hist.get('date', 'Unknown')}:\n{hist.get('content', '')}\n"
        
        prompt += """\n\nPlease provide a comparison analysis that highlights:
        1. New insights that weren't present before
        2. Trends or patterns that have evolved over time
        3. Insights that have remained consistent
        4. Any significant changes in recommendations"""
        
        try:
            # Use Anthropic to generate comparison
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.1,
                system="You are an expert data analyst who specializes in comparing current insights with historical data to identify trends, changes, and consistent patterns.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            comparison = message.content[0].text
            
            return {
                "comparison": comparison,
                "historical_insight_count": len(historical_docs)
            }
            
        except Exception as e:
            return {"comparison_error": str(e)}
    
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert various data formats to a pandas DataFrame
        
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
            
            # Handle nested dictionary with simple values at the top level
            if isinstance(data, dict):
                # Try to normalize the JSON structure
                try:
                    return pd.json_normalize(data)
                except Exception:
                    # Fallback for complex structures - extract what we can
                    flat_data = {}
                    for key, value in data.items():
                        if not isinstance(value, (dict, list)):
                            flat_data[key] = value
                    
                    # If we found any simple key-value pairs, return them as a DataFrame
                    if flat_data:
                        return pd.DataFrame([flat_data])
            
            # Last resort - create a single column DataFrame with JSON string
            return pd.DataFrame({"json_data": [json.dumps(data)]})
            
        except Exception as e:
            # Create a DataFrame with error information
            return pd.DataFrame({
                "error": [f"Failed to convert data to DataFrame: {str(e)}"],
                "raw_data": [str(data)]
            })
    
    def _generate_insights(self, df: pd.DataFrame, context: str, insight_type: str) -> List[Dict[str, str]]:
        """
        Generate natural language insights from the data
        
        Args:
            df: DataFrame with the query results
            context: Context about the data domain
            insight_type: Type of insights to focus on
            
        Returns:
            List of insight dictionaries with title and description
        """
        # Get DataFrame summary statistics and info
        df_description = self._get_dataframe_description(df)
        
        # Construct a prompt for the AI
        prompt = f"""
        I have data from a {context} management tool related to {insight_type} insights.
        
        Here is the data description:
        {df_description}
        
        Based on this data, please provide 3-5 key insights that would be valuable for 
        portfolio, program, or project management. Each insight should be provided with 
        a brief title and a more detailed explanation.
        
        Format each insight as:
        - Title: [brief one-line title]
        - Description: [2-3 sentences explaining the insight and its relevance to PPPM]
        
        Focus specifically on {insight_type} aspects and any patterns, trends, or anomalies 
        that would be important for decision-making.
        """
        
        try:
            # Use Anthropic to generate insights
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.2,  # Some creativity but mostly factual
                system="You are an expert data analyst specializing in portfolio, program, and project management (PPPM). Your role is to identify key insights from data that help managers make better decisions.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            insight_text = message.content[0].text
            
            # Parse the insights into a structured format
            insights = []
            current_insight = {}
            for line in insight_text.split('\n'):
                line = line.strip()
                if line.startswith('- Title:'):
                    if current_insight and 'title' in current_insight:
                        insights.append(current_insight)
                    current_insight = {'title': line.replace('- Title:', '').strip()}
                elif line.startswith('- Description:') and 'title' in current_insight:
                    current_insight['description'] = line.replace('- Description:', '').strip()
            
            # Add the last insight
            if current_insight and 'title' in current_insight and 'description' in current_insight:
                insights.append(current_insight)
            
            return insights
            
        except Exception as e:
            # Return a single insight with the error
            return [{
                "title": "Error Generating Insights",
                "description": f"Failed to generate insights: {str(e)}"
            }]
    
    def _get_dataframe_description(self, df: pd.DataFrame) -> str:
        """
        Generate a descriptive summary of the DataFrame
        
        Args:
            df: DataFrame to describe
            
        Returns:
            String description of the DataFrame
        """
        # Capture the output of df.info() as a string
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        # Get descriptive statistics
        try:
            desc_stats = df.describe(include='all').to_string()
        except:
            desc_stats = "Could not generate descriptive statistics."
        
        # Sample data (first few rows)
        try:
            sample_rows = df.head(5).to_string(index=False)
        except:
            sample_rows = "Could not generate sample rows."
        
        # Column list with data types
        columns_str = "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns])
        
        # Check for missing values
        missing_values = df.isnull().sum().to_string()
        
        description = f"""
        DataFrame Summary:
        {info_str}

        Columns:
        {columns_str}

        Sample Data:
        {sample_rows}

        Missing Values:
        {missing_values}

        Description:
        {desc_stats}

        """
        
        return description
    
    def _generate_visualizations(self, df: pd.DataFrame, insights: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Generate visualizations based on the data
        
        Args:
            df: DataFrame with the query results
            insights: List of insights generated
            
        Returns:
            List of visualization dictionaries with title, description, and path
        """
        visualizations = []
        
        # Skip visualization for empty or single-row dataframes
        if df.empty or len(df) < 2:
            return [{
                "title": "Insufficient Data for Visualization",
                "description": "The dataset contains too few rows for meaningful visualization.",
                "format": "text"
            }]
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 
                    (isinstance(df[col].dtype, pd.DatetimeTZDtype))]
        
        # Combination of time series and numerical columns
        time_series_viz = self._create_time_series_visualization(df, date_cols, numerical_cols)
        if time_series_viz:
            visualizations.append(time_series_viz)
        
        # Distribution of numerical columns
        dist_viz = self._create_distribution_visualization(df, numerical_cols)
        if dist_viz:
            visualizations.append(dist_viz)
        
        # Correlation heatmap for numerical columns
        if len(numerical_cols) > 1:
            corr_viz = self._create_correlation_visualization(df, numerical_cols)
            if corr_viz:
                visualizations.append(corr_viz)
        
        # Categorical data visualization
        if categorical_cols:
            cat_viz = self._create_categorical_visualization(df, categorical_cols, numerical_cols)
            if cat_viz:
                visualizations.append(cat_viz)
        
        # If we couldn't generate any visualizations, return a message
        if not visualizations:
            return [{
                "title": "No Suitable Visualizations",
                "description": "The data structure did not allow for automatic visualization generation.",
                "format": "text"
            }]
        
        return visualizations
    
    def _create_time_series_visualization(self, df: pd.DataFrame, date_cols: List[str], 
                                       numerical_cols: List[str]) -> Dict[str, str]:
        """
        Create a time series visualization if date columns are available
        
        Args:
            df: DataFrame with the query results
            date_cols: List of date column names
            numerical_cols: List of numerical column names
            
        Returns:
            Visualization dictionary with title, description, and base64 image
        """
        if not date_cols or not numerical_cols:
            return None
        
        try:
            date_col = date_cols[0]  # Use the first date column
            vis_num_cols = numerical_cols[:3]  # Limit to first 3 numerical columns
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot each numerical column against the date
            for col in vis_num_cols:
                plt.plot(df[date_col], df[col], marker='o', linestyle='-', label=col)
            
            plt.title('Time Series Visualization')
            plt.xlabel(date_col)
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return {
                "title": "Time Series Trends",
                "description": f"Time series visualization of {', '.join(vis_num_cols)} trends over time.",
                "format": "image",
                "data": image_base64
            }
            
        except Exception as e:
            print(f"Error creating time series visualization: {str(e)}")
            return None
    
    def _create_distribution_visualization(self, df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, str]:
        """
        Create distributions visualization for numerical columns
        
        Args:
            df: DataFrame with the query results
            numerical_cols: List of numerical column names
            
        Returns:
            Visualization dictionary with title, description, and base64 image
        """
        if not numerical_cols:
            return None
        
        try:
            # Select up to 4 numerical columns
            vis_num_cols = numerical_cols[:4]
            
            # Create figure with subplots
            fig, axes = plt.subplots(len(vis_num_cols), 1, figsize=(10, 3 * len(vis_num_cols)), sharex=False)
            
            # Handle case of single subplot
            if len(vis_num_cols) == 1:
                axes = [axes]
            
            # Create distribution plots for each column
            for i, col in enumerate(vis_num_cols):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return {
                "title": "Value Distributions",
                "description": f"Distribution visualization showing the spread and central tendencies of {', '.join(vis_num_cols)}.",
                "format": "image",
                "data": image_base64
            }
            
        except Exception as e:
            print(f"Error creating distribution visualization: {str(e)}")
            return None
    
    def _create_correlation_visualization(self, df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, str]:
        """
        Create correlation heatmap for numerical columns
        
        Args:
            df: DataFrame with the query results
            numerical_cols: List of numerical column names
            
        Returns:
            Visualization dictionary with title, description, and base64 image
        """
        if len(numerical_cols) < 2:
            return None
        
        try:
            # Limit to 8 numerical columns maximum for readability
            vis_num_cols = numerical_cols[:8]
            
            # Calculate correlation matrix
            corr_matrix = df[vis_num_cols].corr()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                      square=True, linewidths=.5, cbar_kws={"shrink": .8})
            
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return {
                "title": "Correlation Analysis",
                "description": f"Correlation heatmap showing relationships between {', '.join(vis_num_cols)}. Positive correlations are shown in red, negative in blue.",
                "format": "image",
                "data": image_base64
            }
            
        except Exception as e:
            print(f"Error creating correlation visualization: {str(e)}")
            return None
    
    def _create_categorical_visualization(self, df: pd.DataFrame, categorical_cols: List[str], 
                                        numerical_cols: List[str]) -> Dict[str, str]:
        """
        Create visualization for categorical columns
        
        Args:
            df: DataFrame with the query results
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
            
        Returns:
            Visualization dictionary with title, description, and base64 image
        """
        if not categorical_cols:
            return None
        
        try:
            # Choose one categorical column (preferably with a manageable number of unique values)
            cat_col_candidates = [col for col in categorical_cols if df[col].nunique() <= 10 and df[col].nunique() > 1]
            
            if not cat_col_candidates:
                if categorical_cols[0] and df[categorical_cols[0]].nunique() <= 20:
                    cat_col = categorical_cols[0]
                else:
                    return None  # No suitable categorical column found
            else:
                cat_col = cat_col_candidates[0]
            
            # If numerical columns available, create a relationship visualization
            if numerical_cols:
                num_col = numerical_cols[0]  # Use the first numerical column
                
                plt.figure(figsize=(12, 6))
                
                # Create a barplot of numerical values grouped by categorical column
                sns.barplot(x=cat_col, y=num_col, data=df, errorbar=None)
                
                plt.title(f'{num_col} by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
            else:
                # Create count plot of categorical column
                plt.figure(figsize=(12, 6))
                
                sns.countplot(x=cat_col, data=df)
                
                plt.title(f'Count of {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            
            # Save to buffer and convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            viz_title = "Categorization Analysis"
            viz_desc = f"Visualization showing the distribution of {cat_col}"
            if numerical_cols:
                viz_desc += f" and its relationship with {num_col}"
            viz_desc += "."
            
            return {
                "title": viz_title,
                "description": viz_desc,
                "format": "image",
                "data": image_base64
            }
            
        except Exception as e:
            print(f"Error creating categorical visualization: {str(e)}")
            return None
    
    def _generate_recommendations(self, df: pd.DataFrame, insights: List[Dict[str, str]], 
                               context: str) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations based on insights
        
        Args:
            df: DataFrame with the query results
            insights: List of insights generated
            context: Context about the data domain
            
        Returns:
            List of recommendation dictionaries with title and steps
        """
        # Construct insights text from the insights list
        insights_text = "\n\n".join([f"Insight: {insight['title']}\n{insight['description']}" for insight in insights])
        
        # Get DataFrame summary
        df_description = self._get_dataframe_description(df)
        
        # Construct a prompt for the AI
        prompt = f"""
        I have data from a {context} management tool.
        
        Here is the data description:
        {df_description}
        
        Based on this data, the following insights were identified:
        {insights_text}
        
        Please provide 2-3 actionable recommendations for portfolio, program, or project managers based on these insights. 
        Each recommendation should include:
        - Title: [brief one-line actionable title]
        - Steps: [3-5 specific action steps that managers should take, formatted as a numbered list]
        
        Make sure recommendations are specific, practical, and directly related to the insights and data provided.
        Focus on providing recommendations that would improve decision-making in PPPM.
        """
        
        try:
            # Use Anthropic to generate recommendations
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1200,
                temperature=0.3,  # Some creativity but mostly practical
                system="You are an expert consultant specializing in portfolio, program, and project management (PPPM). Your role is to provide practical, actionable recommendations based on data insights.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            recommendations_text = message.content[0].text
            
            # Parse the recommendations into a structured format
            recommendations = []
            current_recommendation = {}
            steps = []
            parsing_steps = False
            
            for line in recommendations_text.split('\n'):
                line = line.strip()
                
                if line.startswith('- Title:'):
                    # If we were already building a recommendation, save it
                    if current_recommendation and 'title' in current_recommendation:
                        if steps:
                            current_recommendation['steps'] = steps
                        recommendations.append(current_recommendation)
                    
                    # Start a new recommendation
                    current_recommendation = {'title': line.replace('- Title:', '').strip()}
                    steps = []  # Reset steps
                    parsing_steps = False
                    
                elif line.startswith('- Steps:'):
                    parsing_steps = True
                    # Steps will be collected in the next iterations
                
                elif parsing_steps and (line.startswith('1.') or line.startswith('2.') or 
                                       line.startswith('3.') or line.startswith('4.') or 
                                       line.startswith('5.')):
                    # Extract just the step content without the number prefix
                    step_content = line[line.find('.')+1:].strip()
                    steps.append(step_content)
            
            # Add the last recommendation
            if current_recommendation and 'title' in current_recommendation:
                if steps:
                    current_recommendation['steps'] = steps
                recommendations.append(current_recommendation)
            
            return recommendations
            
        except Exception as e:
            # Return a single recommendation with the error
            return [{
                "title": "Error Generating Recommendations",
                "steps": [f"Failed to generate recommendations: {str(e)}"]
            }]
    
    def _generate_summary(self, insights: List[Dict[str, str]], recommendations: List[Dict[str, str]]) -> str:
        """
        Generate a summary of insights and recommendations
        
        Args:
            insights: List of insights generated
            recommendations: List of recommendations
            
        Returns:
            Summary text
        """
        summary = "Summary:\n"
        
        # Summarize insights
        summary += "Insights:\n"
        for insight in insights:
            summary += f"- {insight['title']}: {insight['description']}\n"
        
        # Summarize recommendations
        summary += "\nRecommendations:\n"
        for rec in recommendations:
            summary += f"- {rec['title']}\n"
            for step in rec['steps']:
                summary += f"  - {step}\n"
        
        return summary
