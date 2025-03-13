import json
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from openai import OpenAI
from agents.base_agent import BaseAgent
from utils.config import OPENAI_API_KEY, COST_ESTIMATOR_MODEL
from database.db_manager import DatabaseManager

class CostEstimatorAgent(BaseAgent):
    """
    Agent responsible for estimating costs for projects, programs, and portfolios.
    
    This agent uses AI capabilities to analyze project data, historical information,
    and industry benchmarks to provide accurate cost estimates and forecasts. It works
    with data processed through Structured Unstructured Query Language (SUQL) to generate
    comprehensive cost analyses for PPPM contexts.
    """
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        super().__init__(
            name="Cost Estimator Agent",
            description="Estimates and forecasts costs for projects, programs, and portfolios using SUQL data"
        )
        self.client = None
        self.historical_data = {}
        self.cost_models = {}
        
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
        
        # Initialize cost models with industry standard templates
        self._initialize_cost_models()
    
    def _initialize_cost_models(self):
        """
        Initialize cost models with industry standard templates
        """
        # Software development cost model
        self.cost_models["software_development"] = {
            "labor_rates": {
                "project_manager": 120,  # hourly rate
                "business_analyst": 95,
                "software_architect": 150,
                "senior_developer": 130,
                "developer": 100,
                "qa_engineer": 90,
                "devops_engineer": 110
            },
            "overhead_factor": 1.35,  # 35% overhead
            "contingency_factor": 1.15,  # 15% contingency
            "risk_factors": {
                "high_complexity": 1.3,
                "new_technology": 1.25,
                "tight_schedule": 1.2,
                "unclear_requirements": 1.35
            }
        }
        
        # Infrastructure cost model
        self.cost_models["infrastructure"] = {
            "cloud_costs": {
                "compute": 0.10,  # per hour per instance
                "storage": 0.05,  # per GB per month
                "network": 0.01,  # per GB transferred
                "managed_services": 100  # base monthly fee
            },
            "scaling_factor": 0.8,  # economies of scale
            "contingency_factor": 1.2,  # 20% contingency
        }
        
        # Hardware cost model
        self.cost_models["hardware"] = {
            "server": 5000,
            "workstation": 2000,
            "network_equipment": 10000,
            "maintenance_factor": 0.15,  # 15% of hardware cost per year
            "depreciation_years": 5
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and generate cost estimates
        
        Args:
            input_data: Dictionary containing:
                - 'project_data': Project data including resources, timeline, etc.
                - 'historical_data': Optional historical data for reference
                - 'estimation_type': Type of estimation (rough, detailed, forecast)
                - 'cost_categories': Categories to estimate (labor, materials, etc.)
                - 'project_type': Type of project (software, infrastructure, etc.)
                - 'risk_profile': Risk profile for contingency calculation
        
        Returns:
            Dictionary with cost estimates, breakdowns, and confidence levels
        """
        if not self.is_initialized:
            self.initialize()
        
        # Extract parameters
        project_data = input_data.get('project_data', {})
        historical_data = input_data.get('historical_data', {})
        estimation_type = input_data.get('estimation_type', 'detailed')
        cost_categories = input_data.get('cost_categories', ['labor', 'materials', 'services', 'overhead'])
        project_type = input_data.get('project_type', 'software_development')
        risk_profile = input_data.get('risk_profile', 'medium')
        
        # Store historical data for reference
        self.historical_data = historical_data
        
        # Generate cost estimates based on estimation type
        if estimation_type == 'rough':
            cost_estimate = self._generate_rough_estimate(project_data, project_type, risk_profile)
        elif estimation_type == 'forecast':
            cost_estimate = self._generate_cost_forecast(project_data, project_type, risk_profile)
        else:  # detailed
            cost_estimate = self._generate_detailed_estimate(project_data, cost_categories, project_type, risk_profile)
        
        # Add AI-enhanced insights if project data is sufficient
        if len(project_data) > 0:
            cost_insights = self._generate_cost_insights(project_data, cost_estimate, project_type)
            cost_estimate['insights'] = cost_insights
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(project_data, estimation_type, project_type)
        cost_estimate['confidence_level'] = confidence_level
        
        # Add metadata
        cost_estimate['metadata'] = {
            'estimation_type': estimation_type,
            'project_type': project_type,
            'risk_profile': risk_profile,
            'estimation_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'cost_model_version': '1.0'
        }
        
        return cost_estimate
    
    def _generate_rough_estimate(self, project_data: Dict[str, Any], project_type: str, risk_profile: str) -> Dict[str, Any]:
        """
        Generate a rough order of magnitude estimate based on high-level project parameters
        
        Args:
            project_data: High-level project data
            project_type: Type of project
            risk_profile: Risk profile for contingency calculation
            
        Returns:
            Dictionary with rough cost estimate
        """
        # Extract key parameters
        project_duration_months = project_data.get('duration_months', 6)
        team_size = project_data.get('team_size', 5)
        complexity = project_data.get('complexity', 'medium')
        
        # Get cost model for project type
        cost_model = self.cost_models.get(project_type, self.cost_models.get('software_development'))
        
        # Calculate base labor cost
        avg_rate = sum(cost_model.get('labor_rates', {}).values()) / len(cost_model.get('labor_rates', {}))
        monthly_labor_cost = avg_rate * 160 * team_size  # 160 hours per month
        total_labor_cost = monthly_labor_cost * project_duration_months
        
        # Apply complexity factor
        complexity_factors = {'low': 0.8, 'medium': 1.0, 'high': 1.3, 'very_high': 1.6}
        complexity_factor = complexity_factors.get(complexity, 1.0)
        adjusted_labor_cost = total_labor_cost * complexity_factor
        
        # Apply overhead
        overhead_factor = cost_model.get('overhead_factor', 1.35)
        total_cost_with_overhead = adjusted_labor_cost * overhead_factor
        
        # Apply contingency based on risk profile
        risk_contingency = {'low': 1.1, 'medium': 1.2, 'high': 1.35, 'very_high': 1.5}
        contingency_factor = risk_contingency.get(risk_profile, 1.2)
        total_estimated_cost = total_cost_with_overhead * contingency_factor
        
        # Create cost breakdown
        cost_breakdown = {
            'labor': adjusted_labor_cost,
            'overhead': adjusted_labor_cost * (overhead_factor - 1),
            'contingency': total_cost_with_overhead * (contingency_factor - 1)
        }
        
        # Create result
        result = {
            'total_estimated_cost': round(total_estimated_cost, 2),
            'cost_breakdown': cost_breakdown,
            'estimation_basis': {
                'project_duration_months': project_duration_months,
                'team_size': team_size,
                'complexity': complexity,
                'avg_hourly_rate': avg_rate
            }
        }
        
        return result
    
    def _generate_detailed_estimate(self, project_data: Dict[str, Any], cost_categories: List[str], 
                                   project_type: str, risk_profile: str) -> Dict[str, Any]:
        """
        Generate a detailed cost estimate based on project data
        
        Args:
            project_data: Detailed project data including resources, timeline, etc.
            cost_categories: Categories to estimate
            project_type: Type of project
            risk_profile: Risk profile for contingency calculation
            
        Returns:
            Dictionary with detailed cost estimate
        """
        # Get cost model for project type
        cost_model = self.cost_models.get(project_type, self.cost_models.get('software_development'))
        
        # Initialize cost breakdown
        cost_breakdown = {category: 0 for category in cost_categories}
        
        # Calculate labor costs if resources are provided
        if 'resources' in project_data and 'labor' in cost_categories:
            labor_cost = 0
            for resource in project_data.get('resources', []):
                role = resource.get('role', '').lower().replace(' ', '_')
                allocation = resource.get('allocation', 100) / 100  # Convert percentage to decimal
                duration_months = resource.get('duration_months', project_data.get('duration_months', 6))
                
                # Get hourly rate for role or use average
                hourly_rate = cost_model.get('labor_rates', {}).get(role, 100)
                
                # Calculate cost for this resource
                resource_cost = hourly_rate * 160 * allocation * duration_months  # 160 hours per month
                labor_cost += resource_cost
            
            cost_breakdown['labor'] = labor_cost
        
        # Calculate materials costs if provided
        if 'materials' in project_data and 'materials' in cost_categories:
            materials_cost = sum(item.get('cost', 0) * item.get('quantity', 1) 
                               for item in project_data.get('materials', []))
            cost_breakdown['materials'] = materials_cost
        
        # Calculate services costs if provided
        if 'services' in project_data and 'services' in cost_categories:
            services_cost = sum(service.get('cost', 0) for service in project_data.get('services', []))
            cost_breakdown['services'] = services_cost
        
        # Calculate overhead if requested
        if 'overhead' in cost_categories:
            overhead_factor = cost_model.get('overhead_factor', 1.35) - 1  # Convert to percentage
            overhead_base = cost_breakdown.get('labor', 0) + cost_breakdown.get('materials', 0)
            cost_breakdown['overhead'] = overhead_base * overhead_factor
        
        # Calculate total base cost
        total_base_cost = sum(cost_breakdown.values())
        
        # Apply contingency based on risk profile
        risk_contingency = {'low': 1.1, 'medium': 1.2, 'high': 1.35, 'very_high': 1.5}
        contingency_factor = risk_contingency.get(risk_profile, 1.2) - 1  # Convert to percentage
        contingency_amount = total_base_cost * contingency_factor
        
        # Add contingency to breakdown
        cost_breakdown['contingency'] = contingency_amount
        
        # Calculate total estimated cost
        total_estimated_cost = total_base_cost + contingency_amount
        
        # Create result
        result = {
            'total_estimated_cost': round(total_estimated_cost, 2),
            'cost_breakdown': {k: round(v, 2) for k, v in cost_breakdown.items()},
            'cost_timeline': self._generate_cost_timeline(project_data, cost_breakdown)
        }
        
        return result
    
    def _generate_cost_forecast(self, project_data: Dict[str, Any], project_type: str, 
                               risk_profile: str) -> Dict[str, Any]:
        """
        Generate a cost forecast based on current project data and trends
        
        Args:
            project_data: Project data including current costs and progress
            project_type: Type of project
            risk_profile: Risk profile for forecast calculation
            
        Returns:
            Dictionary with cost forecast
        """
        # Extract key parameters
        current_cost = project_data.get('current_cost', 0)
        planned_cost = project_data.get('planned_cost', 0)
        percent_complete = project_data.get('percent_complete', 0) / 100  # Convert to decimal
        
        # Avoid division by zero
        if percent_complete == 0:
            percent_complete = 0.01
        
        # Calculate cost performance index (CPI)
        earned_value = planned_cost * percent_complete
        cpi = earned_value / current_cost if current_cost > 0 else 1.0
        
        # Calculate estimate at completion (EAC)
        if cpi < 0.8 or cpi > 1.2:  # Significant variance
            eac = current_cost + (planned_cost - earned_value) / cpi
        else:  # Minor variance
            eac = current_cost + (planned_cost - earned_value)
        
        # Apply risk factor based on risk profile
        risk_factors = {'low': 1.05, 'medium': 1.1, 'high': 1.2, 'very_high': 1.3}
        risk_factor = risk_factors.get(risk_profile, 1.1)
        
        # Calculate forecast with risk adjustment
        forecast_with_risk = eac * risk_factor
        
        # Calculate variance from plan
        variance = forecast_with_risk - planned_cost
        variance_percent = (variance / planned_cost) * 100 if planned_cost > 0 else 0
        
        # Create result
        result = {
            'current_cost': round(current_cost, 2),
            'planned_cost': round(planned_cost, 2),
            'percent_complete': round(percent_complete * 100, 1),
            'cost_performance_index': round(cpi, 2),
            'estimate_at_completion': round(eac, 2),
            'forecast_with_risk': round(forecast_with_risk, 2),
            'variance_from_plan': round(variance, 2),
            'variance_percent': round(variance_percent, 1),
            'forecast_accuracy': self._calculate_forecast_accuracy(cpi, percent_complete)
        }
        
        return result
    
    def _generate_cost_timeline(self, project_data: Dict[str, Any], cost_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a monthly cost timeline based on project data and cost breakdown
        
        Args:
            project_data: Project data including timeline information
            cost_breakdown: Cost breakdown by category
            
        Returns:
            Dictionary with monthly cost projections
        """
        # Extract timeline parameters
        start_date = project_data.get('start_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
        duration_months = project_data.get('duration_months', 6)
        
        # Create date range
        try:
            start = pd.Timestamp(start_date)
            date_range = [start + pd.DateOffset(months=i) for i in range(duration_months + 1)]
            months = [d.strftime('%Y-%m') for d in date_range]
        except:
            # Fallback if date parsing fails
            months = [f'Month {i+1}' for i in range(duration_months + 1)]
        
        # Create spending curve (S-curve for most projects)
        # This uses a simple approximation of an S-curve
        spending_curve = []
        total_cost = sum(v for k, v in cost_breakdown.items() if k != 'contingency')
        contingency = cost_breakdown.get('contingency', 0)
        
        for i in range(duration_months + 1):
            # Calculate percentage of total to spend in this month
            # This creates an S-curve effect with slower spending at start and end
            x = i / duration_months
            if x <= 0.2:
                # Slow ramp-up phase (0-20% of timeline)
                pct = x * 2.5 * 0.15  # First 15% of budget
            elif x <= 0.8:
                # Main execution phase (20-80% of timeline)
                pct = 0.15 + (x - 0.2) * 1.67 * 0.7  # Next 70% of budget
            else:
                # Wind-down phase (80-100% of timeline)
                pct = 0.85 + (x - 0.8) * 5 * 0.15  # Final 15% of budget
            
            spending_curve.append(pct)
        
        # Calculate monthly and cumulative costs
        monthly_costs = []
        cumulative_costs = []
        cumulative_pct = 0
        
        for i in range(len(months) - 1):  # Exclude the last month in date_range
            month_pct = spending_curve[i+1] - spending_curve[i]
            month_cost = total_cost * month_pct
            cumulative_pct = spending_curve[i+1]
            cumulative_cost = total_cost * cumulative_pct
            
            # Add proportional contingency for this month
            month_contingency = contingency * month_pct
            
            monthly_costs.append({
                'month': months[i],
                'cost': round(month_cost, 2),
                'contingency': round(month_contingency, 2),
                'total': round(month_cost + month_contingency, 2)
            })
            
            cumulative_costs.append({
                'month': months[i],
                'cost': round(cumulative_cost, 2),
                'contingency': round(contingency * cumulative_pct, 2),
                'total': round(cumulative_cost + (contingency * cumulative_pct), 2),
                'percent_of_total': round(cumulative_pct * 100, 1)
            })
        
        return {
            'monthly': monthly_costs,
            'cumulative': cumulative_costs
        }
    
    def _generate_cost_insights(self, project_data: Dict[str, Any], cost_estimate: Dict[str, Any], 
                               project_type: str) -> List[Dict[str, Any]]:
        """
        Generate AI-enhanced insights about the cost estimate
        
        Args:
            project_data: Project data
            cost_estimate: Generated cost estimate
            project_type: Type of project
            
        Returns:
            List of insights about the cost estimate
        """
        insights = []
        
        # Extract key data
        total_cost = cost_estimate.get('total_estimated_cost', 0)
        cost_breakdown = cost_estimate.get('cost_breakdown', {})
        
        # Check if we have historical data to compare
        if self.historical_data and 'similar_projects' in self.historical_data:
            similar_projects = self.historical_data.get('similar_projects', [])
            if similar_projects:
                # Calculate average cost of similar projects
                avg_cost = sum(p.get('total_cost', 0) for p in similar_projects) / len(similar_projects)
                cost_diff_pct = ((total_cost - avg_cost) / avg_cost) * 100 if avg_cost > 0 else 0
                
                # Add comparison insight
                insights.append({
                    'type': 'comparison',
                    'title': 'Comparison to Similar Projects',
                    'description': f"This estimate is {abs(round(cost_diff_pct, 1))}% {'higher' if cost_diff_pct > 0 else 'lower'} than the average cost of similar projects.",
                    'data': {
                        'current_estimate': total_cost,
                        'historical_average': avg_cost,
                        'difference_percent': round(cost_diff_pct, 1)
                    }
                })
        
        # Check labor cost proportion
        if 'labor' in cost_breakdown:
            labor_pct = (cost_breakdown.get('labor', 0) / total_cost) * 100 if total_cost > 0 else 0
            expected_labor_pct = 65 if project_type == 'software_development' else 50
            
            if abs(labor_pct - expected_labor_pct) > 15:
                insights.append({
                    'type': 'cost_structure',
                    'title': 'Unusual Labor Cost Proportion',
                    'description': f"Labor costs represent {round(labor_pct, 1)}% of the total budget, which is {'higher' if labor_pct > expected_labor_pct else 'lower'} than typical for this type of project ({expected_labor_pct}%).",
                    'data': {
                        'labor_percent': round(labor_pct, 1),
                        'expected_percent': expected_labor_pct
                    }
                })
        
        # Check contingency level
        if 'contingency' in cost_breakdown:
            contingency_pct = (cost_breakdown.get('contingency', 0) / total_cost) * 100 if total_cost > 0 else 0
            
            if contingency_pct < 10:
                insights.append({
                    'type': 'risk',
                    'title': 'Low Contingency Reserve',
                    'description': f"The contingency reserve is only {round(contingency_pct, 1)}% of the total budget, which may be insufficient to cover unexpected costs.",
                    'data': {
                        'contingency_percent': round(contingency_pct, 1),
                        'recommended_minimum': 10
                    }
                })
            elif contingency_pct > 25:
                insights.append({
                    'type': 'efficiency',
                    'title': 'High Contingency Reserve',
                    'description': f"The contingency reserve is {round(contingency_pct, 1)}% of the total budget, which may indicate high uncertainty or risk aversion.",
                    'data': {
                        'contingency_percent': round(contingency_pct, 1),
                        'typical_maximum': 25
                    }
                })
        
        return insights
    
    def _calculate_confidence_level(self, project_data: Dict[str, Any], estimation_type: str, 
                                   project_type: str) -> Dict[str, Any]:
        """
        Calculate confidence level for the cost estimate
        
        Args:
            project_data: Project data
            estimation_type: Type of estimation
            project_type: Type of project
            
        Returns:
            Dictionary with confidence level and range
        """
        # Base confidence levels by estimation type
        base_confidence = {
            'rough': 0.6,
            'detailed': 0.8,
            'forecast': 0.75
        }.get(estimation_type, 0.7)
        
        # Adjust based on data completeness
        data_completeness = 0.5  # Default
        
        # Check for key data elements
        required_elements = {
            'rough': ['duration_months', 'team_size'],
            'detailed': ['resources', 'duration_months', 'start_date'],
            'forecast': ['current_cost', 'planned_cost', 'percent_complete']
        }.get(estimation_type, [])
        
        if required_elements:
            present_elements = sum(1 for elem in required_elements if elem in project_data)
            data_completeness = present_elements / len(required_elements)
        
        # Calculate adjusted confidence
        adjusted_confidence = base_confidence * (0.5 + 0.5 * data_completeness)
        
        # Calculate confidence range
        if estimation_type == 'rough':
            lower_range = -30  # -30%
            upper_range = 50   # +50%
        elif estimation_type == 'detailed':
            lower_range = -15  # -15%
            upper_range = 20   # +20%
        else:  # forecast
            # For forecasts, range depends on percent complete
            percent_complete = project_data.get('percent_complete', 0)
            if percent_complete >= 75:
                lower_range = -5   # -5%
                upper_range = 10   # +10%
            elif percent_complete >= 50:
                lower_range = -10  # -10%
                upper_range = 15   # +15%
            elif percent_complete >= 25:
                lower_range = -15  # -15%
                upper_range = 20   # +20%
            else:
                lower_range = -20  # -20%
                upper_range = 30   # +30%
        
        return {
            'confidence_score': round(adjusted_confidence, 2),
            'range_lower_percent': lower_range,
            'range_upper_percent': upper_range
        }
    
    def _calculate_forecast_accuracy(self, cpi: float, percent_complete: float) -> str:
        """
        Calculate the forecast accuracy based on cost performance index and project completion percentage
        
        Args:
            cpi: Cost Performance Index
            percent_complete: Percentage of project completion (0-1)
            
        Returns:
            String indicating forecast accuracy level
        """
        # Higher completion percentage gives more accurate forecasts
        completion_factor = min(percent_complete * 2, 1.0)  # Scale up to 50% completion
        
        # CPI stability affects accuracy (values close to 1.0 are more reliable)
        cpi_stability = 1.0 - min(abs(cpi - 1.0), 0.5) * 2  # Scale CPI variance
        
        # Calculate combined accuracy score (0-1)
        accuracy_score = (completion_factor * 0.7) + (cpi_stability * 0.3)
        
        # Convert to descriptive accuracy level
        if accuracy_score >= 0.8:
            return "High"
        elif accuracy_score >= 0.6:
            return "Medium"
        elif accuracy_score >= 0.4:
            return "Low"
        else:
            return "Very Low"
