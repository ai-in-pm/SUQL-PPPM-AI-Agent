# SUQL-PPPM AI Agent System

A sophisticated AI agent system for Portfolio, Program, and Project Management (PPPM) using Structured Unstructured Query Language (SUQL).

The development of this repository was inspired by the paper "SUQL: Conversational Search over Structured and Unstructured Data with Large Language Models". To read the full paper, visit https://arxiv.org/pdf/2311.09818

## Overview

The SUQL-PPPM AI Agent System is designed to enhance project management processes through AI-powered data extraction, query processing, insights generation, risk management, and cost estimation. The system uses a custom Structured Unstructured Query Language (SUQL) to enable natural language queries against structured and unstructured project data.

## Features

- **Data Extraction**: Extract structured and unstructured data from various project documents and sources
- **SUQL Query Processing**: Process structured and unstructured queries to retrieve and analyze project data
- **Insights Generation**: Generate actionable insights and visualizations from project data
- **Risk Management**: Identify, analyze, and suggest mitigation strategies for project risks
- **Cost Estimation**: Generate detailed cost estimates and forecasts for projects

## System Architecture

The system consists of five main agent components:

1. **Data Extraction Agent**: Extracts and processes data from various sources
2. **Query Processing Agent**: Processes SUQL queries against the extracted data
3. **Insights Generation Agent**: Generates insights based on query results
4. **Risk Management Agent**: Analyzes project risks and provides mitigation strategies
5. **Cost Estimator Agent**: Provides cost estimates and forecasts for projects

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SUQL-PPPM-AI-Agent.git
cd SUQL-PPPM-AI-Agent

# Create a virtual environment
python -m venv suql-env

# Activate the virtual environment
# On Windows
suql-env\Scripts\activate
# On macOS/Linux
source suql-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Usage

```bash
python main.py
```

This will run a demonstration of the SUQL-PPPM AI Agent system using sample project data.

## Output

```bash
  Labor: $676,000.00
  Overhead: $236,600.00
  Contingency: $182,520.00

Generating cost forecast...
Cost forecast completed successfully!

Cost Forecast:
Current Cost: $580,000.00
Planned Cost: $850,000.00
Percent Complete: 65.0%
Cost Performance Index: 0.95
Forecast at Completion: $965,250.00
Variance from Plan: $115,250.00 (13.6%)

Cost Insights:

1. Comparison to Similar Projects
   This estimate is 39.8% higher than the average cost of similar projects.     
---
## Sample SUQL Queries

The system supports a variety of SUQL queries, such as:

```sql
SELECT project_info.name, project_info.status, budget.total_budget, budget.spent_to_date, 
       ANSWER(project_info.description, 'What is the main goal of this project?') AS project_goal,
       CALCULATE(budget.spent_to_date / budget.total_budget * 100) AS budget_utilization_percentage,
       INSIGHTS(milestones, 'Are there any schedule delays in the milestones?') AS schedule_insights
FROM project_data
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
