"""
Security Report Generator using Groq LLM
Generates intelligent summaries of IDS predictions with detailed analysis
"""
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class ReportGenerator:
    """Generate detailed security reports using LLM"""
    
    def __init__(self, api_key: str):
        """Initialize with Groq API key"""
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        
        self.analysis_prompt = PromptTemplate(
            template="""You are an expert cybersecurity analyst and data scientist. Analyze the intrusion detection results and provide comprehensive insights.

DATASET OVERVIEW:
- Total Flows: {total_flows}
- Features Used: 18 universal network flow features
- Dataset Type: {dataset_type}

GROUND TRUTH (if available):
{ground_truth}

MODEL PERFORMANCE:
{model_results}

BEST PERFORMING MODELS:
{best_models}

WORST PERFORMING MODELS:
{worst_models}

FEATURE STATISTICS:
{feature_stats}

Provide a detailed analysis with:

1. **Executive Summary** (3-4 sentences about overall findings)

2. **Dataset Characteristics** (what type of traffic patterns are present)

3. **Model Performance Analysis**:
   - Why are the best models performing well?
   - Why are the worst models struggling?
   - What patterns are they detecting/missing?

4. **Accuracy Insights** (if ground truth available):
   - Which models are most accurate and why?
   - What types of attacks are being detected/missed?
   - Are there any false positives/negatives patterns?

5. **Feature Importance** (based on the statistics):
   - Which features likely contributed most to detection?
   - Any unusual patterns in the data?

6. **Recommendations**:
   - Which models to trust for this dataset
   - Any concerns or limitations
   - Suggested next steps

Use technical language but keep it clear and actionable. Format with markdown.
""",
            input_variables=["total_flows", "dataset_type", "ground_truth", 
                           "model_results", "best_models", "worst_models", "feature_stats"]
        )
    
    def generate_detailed_report(self, df: pd.DataFrame, all_results: dict, 
                                has_ground_truth: bool = False) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            df: Original dataframe with flow data
            all_results: Dictionary of {model_name: (predictions, confidence)}
            has_ground_truth: Whether True_Label column exists
            
        Returns:
            Detailed markdown report
        """
        total_flows = len(df)
        
        # Ground truth analysis
        if has_ground_truth and 'True_Label' in df.columns:
            true_benign = sum(df['True_Label'] == 0)
            true_attack = sum(df['True_Label'] == 1)
            ground_truth = f"""- True Benign: {true_benign} ({true_benign/total_flows*100:.1f}%)
- True Attack: {true_attack} ({true_attack/total_flows*100:.1f}%)"""
            dataset_type = "AI-Generated with Ground Truth Labels"
        else:
            ground_truth = "No ground truth labels available"
            dataset_type = "Real-world dataset (unlabeled)"
        
        # Model results summary
        model_results_list = []
        accuracy_list = []
        
        for model_name, (predictions, confidence) in all_results.items():
            benign = sum(predictions == 0)
            malicious = sum(predictions == 1)
            avg_conf = np.mean(confidence) * 100
            
            result_str = f"- {model_name}: {benign} benign, {malicious} malicious (avg conf: {avg_conf:.1f}%)"
            model_results_list.append(result_str)
            
            # Calculate accuracy if ground truth available
            if has_ground_truth and 'True_Label' in df.columns:
                accuracy = np.mean(predictions == df['True_Label'].values) * 100
                accuracy_list.append((model_name, accuracy, benign, malicious))
        
        model_results = "\n".join(model_results_list)
        
        # Best and worst models
        if accuracy_list:
            accuracy_list.sort(key=lambda x: x[1], reverse=True)
            best_models = "\n".join([f"- {name}: {acc:.2f}% accuracy" 
                                    for name, acc, _, _ in accuracy_list[:3]])
            worst_models = "\n".join([f"- {name}: {acc:.2f}% accuracy" 
                                     for name, acc, _, _ in accuracy_list[-3:]])
        else:
            best_models = "Ground truth not available"
            worst_models = "Ground truth not available"
        
        # Feature statistics
        feature_cols = [
            'Flow Duration', 'Destination Port', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Packets/s', 'Flow Bytes/s', 'Packet Length Mean', 'Flow IAT Mean',
            'SYN Flag Count', 'ACK Flag Count', 'RST Flag Count'
        ]
        
        feature_stats_list = []
        for feat in feature_cols:
            if feat in df.columns:
                mean_val = df[feat].replace([np.inf, -np.inf], np.nan).mean()
                std_val = df[feat].replace([np.inf, -np.inf], np.nan).std()
                if pd.notna(mean_val):
                    feature_stats_list.append(f"- {feat}: mean={mean_val:.2f}, std={std_val:.2f}")
        
        feature_stats = "\n".join(feature_stats_list[:8])  # Top 8 features
        
        # Generate report
        chain = self.analysis_prompt | self.llm
        
        response = chain.invoke({
            "total_flows": total_flows,
            "dataset_type": dataset_type,
            "ground_truth": ground_truth,
            "model_results": model_results,
            "best_models": best_models,
            "worst_models": worst_models,
            "feature_stats": feature_stats
        })
        
        return response.content