import pandas as pd
import altair as alt

def compare_rag_metrics(baseline_filepath, optimized_filepath, metrics=['answer_correctness', 'faithfulness', 'answer_relevancy', 'context_precision'], chart_title='Baseline vs. Optimized RAG System Metrics', output_filename='rag_system_metrics_comparison.json'):
    """
    Compares the metrics of two RAG systems (baseline and optimized) using boxplots.

    Args:
        baseline_filepath (str): File path to the CSV containing baseline results.
        optimized_filepath (str): File path to the CSV containing optimized results.
        metrics (list, optional): List of metrics to compare. Defaults to the four standard RAG metrics.
        chart_title (str, optional): Title for the generated chart.
        output_filename (str, optional): Filename for saving the chart.

    Returns:
        altair.Chart: The generated Altair chart object.
    """
    
    # Load data
    df_baseline = pd.read_csv(baseline_filepath)
    df_optimized = pd.read_csv(optimized_filepath)

    # Drop Unnamed Columns
    df_baseline.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df_optimized.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    # Melt and prepare data
    df_baseline_melt = df_baseline.melt(value_vars=metrics, var_name='Metric', value_name='Value')
    df_baseline_melt['System'] = 'Baseline'

    df_optimized_melt = df_optimized.melt(value_vars=metrics, var_name='Metric', value_name='Value')
    df_optimized_melt['System'] = 'Optimized'

    df_combined = pd.concat([df_baseline_melt, df_optimized_melt])

    # Create the combined boxplot
    chart = alt.Chart(df_combined).mark_boxplot(ticks=True).encode(
        x=alt.X('System:N', title=None, axis=alt.Axis(labels=False, ticks=False), scale=alt.Scale(padding=1)), 
        y=alt.Y('Value:Q'), 
        color='System:N',
        column=alt.Column('Metric:N', sort=list(metrics), header=alt.Header(orient='bottom'))
    ).properties(
        width=100,
        title=chart_title
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )

    # Save and return the chart
    chart.save(output_filename)
    return chart

# Example usage with your data
# compare_rag_metrics('baseline_ragas_results.csv', 'bm_prompt_engineering_optimization_results.csv')
