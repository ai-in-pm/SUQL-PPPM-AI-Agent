#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SUQL-PPM AI Agent System

This is the main application file that demonstrates the usage of the SUQL-PPM AI Agent system,
which includes multiple agents for data extraction, query processing, insights generation,
and risk management for Portfolio, Program, and Project Management (PPPM).

SUQL stands for Structured Unstructured Query Language, which enables querying both structured 
and unstructured data sources using a unified syntax.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Import agents
from agents.data_extraction_agent import DataExtractionAgent
from agents.query_processing_agent import QueryProcessingAgent
from agents.insights_generation_agent import InsightsGenerationAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.cost_estimator_agent import CostEstimatorAgent

# Load environment variables
load_dotenv()


def main():
    """
    Main function to demonstrate the SUQL-PPM AI Agent system
    """
    print("=== SUQL-PPM AI Agent System ===\n")
    
    # Initialize agents
    data_extraction_agent = DataExtractionAgent()
    query_processing_agent = QueryProcessingAgent()
    insights_generation_agent = InsightsGenerationAgent()
    risk_management_agent = RiskManagementAgent()
    cost_estimator_agent = CostEstimatorAgent()
    
    # Initialize agents
    data_extraction_agent.initialize()
    query_processing_agent.initialize()
    insights_generation_agent.initialize()
    risk_management_agent.initialize()
    cost_estimator_agent.initialize()
    
    # Connect the Query Processing Agent to the Data Extraction Agent
    query_processing_agent.connect_data_agent(data_extraction_agent)
    
    # Sample data (create sample directory and file)
    sample_data_dir = Path("sample_data")
    sample_data_dir.mkdir(exist_ok=True)
    
    # Create a sample project data file
    sample_project_data = {
        "project_info": {
            "id": "PRJ-2023-001",
            "name": "Enterprise Data Warehouse Modernization",
            "description": "Upgrade the existing data warehouse to a cloud-based solution with enhanced analytics capabilities",
            "start_date": "2023-01-15",
            "planned_end_date": "2023-10-30",
            "actual_end_date": None,
            "status": "In Progress",
            "completion_percentage": 65
        },
        "budget": {
            "total_budget": 850000,
            "spent_to_date": 580000,
            "remaining": 270000,
            "cost_variance": -20000
        },
        "resources": [
            {"name": "John Smith", "role": "Project Manager", "allocation": 100},
            {"name": "Lisa Johnson", "role": "Data Architect", "allocation": 80},
            {"name": "Michael Chen", "role": "ETL Developer", "allocation": 100},
            {"name": "Sarah Williams", "role": "Business Analyst", "allocation": 50},
            {"name": "Robert Davis", "role": "Cloud Engineer", "allocation": 75}
        ],
        "milestones": [
            {"name": "Requirements Gathering", "planned_date": "2023-02-28", "actual_date": "2023-03-10", "status": "Completed"},
            {"name": "Architecture Design", "planned_date": "2023-04-15", "actual_date": "2023-04-20", "status": "Completed"},
            {"name": "Development Phase 1", "planned_date": "2023-06-30", "actual_date": "2023-07-05", "status": "Completed"},
            {"name": "User Acceptance Testing", "planned_date": "2023-08-15", "actual_date": None, "status": "In Progress"},
            {"name": "Go-Live", "planned_date": "2023-10-15", "actual_date": None, "status": "Not Started"}
        ],
        "risks": [
            {
                "id": "RISK-001",
                "name": "Data Migration Delay",
                "description": "Potential delay in migrating historical data due to inconsistencies in source data format",
                "probability": 0.7,
                "impact": 0.8,
                "status": "Open"
            },
            {
                "id": "RISK-002",
                "name": "Resource Constraints",
                "description": "ETL developer might be pulled into another critical project",
                "probability": 0.5,
                "impact": 0.6,
                "status": "Open"
            },
            {
                "id": "RISK-003",
                "name": "Budget Overrun",
                "description": "Cloud infrastructure costs might exceed initial estimates",
                "probability": 0.4,
                "impact": 0.9,
                "status": "Mitigated"
            }
        ],
        "issues": [
            {
                "id": "ISSUE-001",
                "name": "API Integration Failure",
                "description": "The third-party API integration is failing due to undocumented API changes",
                "priority": "High",
                "status": "In Progress",
                "assigned_to": "Lisa Johnson"
            },
            {
                "id": "ISSUE-002",
                "name": "Performance Bottleneck",
                "description": "Query performance degradation observed during large data loads",
                "priority": "Medium",
                "status": "Open",
                "assigned_to": "Michael Chen"
            }
        ],
        "stakeholders": [
            {"name": "Jane Wilson", "role": "Executive Sponsor", "department": "Finance", "influence": "High"},
            {"name": "David Thompson", "role": "Business Owner", "department": "Analytics", "influence": "High"},
            {"name": "Emily Harris", "role": "End User Representative", "department": "Marketing", "influence": "Medium"},
            {"name": "Thomas Moore", "role": "IT Director", "department": "IT", "influence": "Medium"},
            {"name": "Patricia Garcia", "role": "Compliance Officer", "department": "Legal", "influence": "Low"}
        ],
        "dependencies": [
            {"id": "DEP-001", "name": "Cloud Environment Setup", "dependent_on": "External Cloud Team", "status": "Completed"},
            {"id": "DEP-002", "name": "Data Security Approval", "dependent_on": "Security Team", "status": "Delayed"},
            {"id": "DEP-003", "name": "Source System API Access", "dependent_on": "Legacy System Team", "status": "In Progress"}
        ]
    }
    
    # Save the sample project data to a file
    sample_project_file = sample_data_dir / "project_data.json"
    with open(sample_project_file, "w") as f:
        json.dump(sample_project_data, f, indent=4)
    
    print(f"Created sample project data at: {sample_project_file}\n")
    
    # Step 1: Extract data using the Data Extraction Agent
    print("Step 1: Extracting data...")
    extracted_data = data_extraction_agent.process({
        "source_type": "json_file", 
        "source": str(sample_project_file)
    })
    print("Data extraction completed successfully!\n")
    
    # Manually add the project data to the json_data dictionary with the key 'project_data'
    data_extraction_agent.json_data['project_data'] = sample_project_data
    
    # Step 2: Process a SUQL query using the Query Processing Agent
    print("Step 2: Processing SUQL query...")
    suql_query = """SELECT project_info.name, project_info.status, budget.total_budget, budget.spent_to_date, 
                  ANSWER(project_info.description, 'What is the main goal of this project?') AS project_goal,
                  CALCULATE(budget.spent_to_date / budget.total_budget * 100) AS budget_utilization_percentage,
                  INSIGHTS(milestones, 'Are there any schedule delays in the milestones?') AS schedule_insights
              FROM project_data"""
    
    query_result = query_processing_agent.process({
        "query": suql_query,
        "data": data_extraction_agent.json_data,
        "context": extracted_data.get("context")
    })
    print("Query processing completed successfully!")
    print("\nQuery Result:")
    print(query_result.get("result"))
    print("\n")
    
    # Get the query result DataFrame
    result_df = query_result.get("result")
    
    # Convert DataFrame to dictionary for the insights agent
    if isinstance(result_df, pd.DataFrame):
        result_data = result_df.to_dict(orient="records")
    else:
        result_data = result_df
    
    # Step 3: Generate insights using the Insights Generation Agent
    print("Step 3: Generating insights...")
    insights = insights_generation_agent.process({
        "query_result": result_data,
        "context": "Project performance analysis for Enterprise Data Warehouse Modernization",
        "insight_type": "comprehensive"
    })
    print("Insights generation completed successfully!")
    print("\nInsights:")
    print(insights.get("insights"))
    print("\n")
    
    # Step 4: Perform risk management analysis using the Risk Management Agent
    print("Step 4: Performing risk management analysis...")
    risk_analysis = risk_management_agent.process({
        "data": sample_project_data,
        "project_context": "Enterprise Data Warehouse Modernization project with budget concerns and tight timeline",
        "risk_categories": ["technical", "schedule", "resource", "budget"]
    })
    print("Risk management analysis completed successfully!")
    print("\nRisk Summary:")
    print(risk_analysis.get("risk_summary"))
    print("\n")
    
    # Display identified risks with mitigation strategies
    print("Identified Risks with Mitigation Strategies:")
    for i, risk in enumerate(risk_analysis.get("risks"), 1):
        print(f"\n{i}. {risk.get('title')} (ID: {risk.get('id')})")
        print(f"   Category: {risk.get('category')}")
        print(f"   Probability: {risk.get('probability')}")
        print(f"   Impact: {risk.get('impact')}")
        print(f"   Risk Score: {risk.get('risk_score')}")
        print(f"   Mitigation Strategy: {risk.get('mitigation_strategy')}")
    print("\n")
    
    # Step 5: Generate cost estimates using the Cost Estimator Agent
    print("Step 5: Generating cost estimates...")
    
    # Prepare project data for cost estimation
    project_data_for_cost = {
        "duration_months": 10,  # Based on start and end dates
        "team_size": len(sample_project_data["resources"]),
        "complexity": "high",
        "resources": [
            {
                "role": resource["role"].lower().replace(" ", "_"),
                "allocation": resource["allocation"],
                "duration_months": 10
            } for resource in sample_project_data["resources"]
        ],
        "current_cost": sample_project_data["budget"]["spent_to_date"],
        "planned_cost": sample_project_data["budget"]["total_budget"],
        "percent_complete": sample_project_data["project_info"]["completion_percentage"],
        "start_date": sample_project_data["project_info"]["start_date"]
    }
    
    # Sample historical data for comparison
    historical_data = {
        "similar_projects": [
            {"name": "Data Lake Implementation", "total_cost": 750000, "duration_months": 9},
            {"name": "BI Platform Upgrade", "total_cost": 920000, "duration_months": 12},
            {"name": "Data Integration Project", "total_cost": 680000, "duration_months": 8}
        ]
    }
    
    # Generate detailed cost estimate
    detailed_estimate = cost_estimator_agent.process({
        "project_data": project_data_for_cost,
        "historical_data": historical_data,
        "estimation_type": "detailed",
        "cost_categories": ["labor", "overhead", "contingency"],
        "project_type": "software_development",
        "risk_profile": "medium"
    })
    
    print("Cost estimation completed successfully!")
    print("\nDetailed Cost Estimate:")
    print(f"Total Estimated Cost: ${detailed_estimate.get('total_estimated_cost', 0):,.2f}")
    print("\nCost Breakdown:")
    for category, amount in detailed_estimate.get('cost_breakdown', {}).items():
        print(f"  {category.title()}: ${amount:,.2f}")
    
    # Generate cost forecast
    print("\nGenerating cost forecast...")
    forecast = cost_estimator_agent.process({
        "project_data": project_data_for_cost,
        "estimation_type": "forecast",
        "project_type": "software_development",
        "risk_profile": "medium"
    })
    
    print("Cost forecast completed successfully!")
    print("\nCost Forecast:")
    print(f"Current Cost: ${forecast.get('current_cost', 0):,.2f}")
    print(f"Planned Cost: ${forecast.get('planned_cost', 0):,.2f}")
    print(f"Percent Complete: {forecast.get('percent_complete', 0)}%")
    print(f"Cost Performance Index: {forecast.get('cost_performance_index', 0)}")
    print(f"Forecast at Completion: ${forecast.get('forecast_with_risk', 0):,.2f}")
    print(f"Variance from Plan: ${forecast.get('variance_from_plan', 0):,.2f} ({forecast.get('variance_percent', 0)}%)")
    
    # Display cost insights if available
    if 'insights' in detailed_estimate:
        print("\nCost Insights:")
        for i, insight in enumerate(detailed_estimate.get('insights', []), 1):
            print(f"\n{i}. {insight.get('title')}")
            print(f"   {insight.get('description')}")


if __name__ == "__main__":
    main()
