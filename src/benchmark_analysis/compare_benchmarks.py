import pandas as pd
import altair as alt

metrics = ['answer_correctness', 'faithfulness', 'answer_relevancy', 'context_precision']


def compare_rag_metrics(
      file_dict, 
      metrics=metrics, 
      chart_title='RAG System Metrics Comparison', 
      output_filename='rag_system_metrics_comparison.json'
):
    """
    Compares the metrics of multiple RAG systems using boxplots.

    Args:
        file_dict (dict): Dictionary with system names as keys and file paths as values.
        metrics (list, optional): List of metrics to compare. Defaults to the four standard RAG metrics.
        chart_title (str, optional): Title for the generated chart.
        output_filename (str, optional): Filename for saving the chart.

    Returns:
        altair.Chart: The generated Altair chart object.
    """
    
    # Initialize an empty list to store DataFrames
    df_list = []

    # Iterate over the dictionary to load data
    for name, filepath in file_dict.items():
        df = pd.read_csv(filepath)
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')  # Drop Unnamed columns if they exist
        df_melt = df.melt(value_vars=metrics, var_name='Metric', value_name='Value')
        df_melt['System'] = name  # Add a column for the system name
        df_list.append(df_melt)

    # Combine all DataFrames into a single DataFrame
    df_combined = pd.concat(df_list, ignore_index=True)

    # Create the combined boxplot
    chart = alt.Chart(df_combined).mark_boxplot(ticks=True).encode(
        x=alt.X('System:N', title=None, axis=alt.Axis(labels=False, ticks=False), scale=alt.Scale(padding=0.5)), 
        y=alt.Y('Value:Q'), 
        color='System:N',
        column=alt.Column('Metric:N', sort=list(metrics), header=alt.Header(orient='bottom'))
    ).properties(
        width=100,
        title=chart_title
    ).configure_facet(
        spacing=0  # Reduce spacing between facets
    ).configure_view(
        stroke=None
    )

    # Save and return the chart
    chart.save(output_filename)
    return chart

# Example usage with
# file_dict = {
#   "Baseline": baseline_filepath,
#   "Optimized": optimized_filepath,
#   '3_large': embedding_3_large
# }

# compare_rag_metrics(file_dict)