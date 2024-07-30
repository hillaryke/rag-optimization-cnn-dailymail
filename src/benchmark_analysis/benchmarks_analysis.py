import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

class BenchmarkAnalysis:
    def __init__(self, baseline_df, prompt_eng_df):
        self.baseline_df = baseline_df.copy()
        self.prompt_eng_df = prompt_eng_df.copy()
        self._clean_data()
        
    def _clean_data(self):
        """Drops unnamed index columns if they exist."""
        self.baseline_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
        self.prompt_eng_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
        
    def calculate_summary_statistics(self):
        """Calculates summary statistics for the specified numeric columns."""
        numeric_columns = ['answer_correctness', 'faithfulness', 'answer_relevancy', 'context_precision']
        summary_stats = {
            'Metric': [],
            'Baseline_Average': [],
            'Prompt_eng_opt_Average': [],
            'Baseline_Highest': [],
            'Prompt_eng_opt_Highest': [],
            'Baseline_Lowest': [],
            'Prompt_eng_opt_Lowest': []
        }
        
        for column in numeric_columns:
            summary_stats['Metric'].append(column)
            summary_stats['Baseline_Average'].append(self.baseline_df[column].mean())
            summary_stats['Prompt_eng_opt_Average'].append(self.prompt_eng_df[column].mean())
            summary_stats['Baseline_Highest'].append(self.baseline_df[column].max())
            summary_stats['Prompt_eng_opt_Highest'].append(self.prompt_eng_df[column].max())
            summary_stats['Baseline_Lowest'].append(self.baseline_df[column].min())
            summary_stats['Prompt_eng_opt_Lowest'].append(self.prompt_eng_df[column].min())

        summary_df = pd.DataFrame(summary_stats)
        return summary_df
    
    def visualize_summary_statistics(self, summary_df):
        """Visualizes the summary statistics using bar plots."""
        plt.figure(figsize=(14, 10))

        # Average comparison
        plt.subplot(3, 1, 1)
        sns.barplot(x='Metric', y='value', hue='variable', data=pd.melt(summary_df, id_vars=['Metric'], value_vars=['Baseline_Average', 'Prompt_eng_opt_Average']))
        plt.title('Average Comparison')
        plt.xticks(rotation=45)

        # Highest value comparison
        plt.subplot(3, 1, 2)
        sns.barplot(x='Metric', y='value', hue='variable', data=pd.melt(summary_df, id_vars=['Metric'], value_vars=['Baseline_Highest', 'Prompt_eng_opt_Highest']))
        plt.title('Highest Value Comparison')
        plt.xticks(rotation=45)

        # Lowest value comparison
        plt.subplot(3, 1, 3)
        sns.barplot(x='Metric', y='value', hue='variable', data=pd.melt(summary_df, id_vars=['Metric'], value_vars=['Baseline_Lowest', 'Prompt_eng_opt_Lowest']))
        plt.title('Lowest Value Comparison')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
    
    def calculate_deviations(self):
        """Calculates deviations between baseline and prompt engineering optimized DataFrames."""
        numeric_columns = ['answer_correctness', 'faithfulness', 'answer_relevancy', 'context_precision']
        deviations = {
            'question': self.baseline_df['question'],
            'answer': self.baseline_df['answer']
        }
        
        for column in numeric_columns:
            deviations[column + '_deviation'] = self.baseline_df[column] - self.prompt_eng_df[column]

        deviations_df = pd.DataFrame(deviations)
        return deviations_df
    
    def visualize_deviations(self, deviations_df):
        """Visualizes the deviations using Altair."""
        numeric_columns = ['answer_correctness', 'faithfulness', 'answer_relevancy', 'context_precision']
        deviation_melted = deviations_df.melt(id_vars=['question', 'answer'], value_vars=[col + '_deviation' for col in numeric_columns], var_name='Metric', value_name='Deviation')

        # Create the Altair plot
        chart = alt.Chart(deviation_melted).mark_bar().encode(
            x=alt.X('question:N', title='Question', sort=None),
            y=alt.Y('Deviation:Q', title='Deviation'),
            color='Metric:N',
            tooltip=['question', 'answer', 'Metric', 'Deviation']
        ).properties(
            width=800,
            height=400,
            title='Deviations for Each Answer Comparing Baseline and RAGAS Scores'
        ).interactive()

        chart.show()

# Example usage:
# baseline_df = pd.read_csv('path_to_baseline.csv')
# prompt_eng_df = pd.read_csv('path_to_prompt_eng.csv')
# benchmark = BenchmarkAnalysis(baseline_df, prompt_eng_df)
# summary_df = benchmark.calculate_summary_statistics()
# print(summary_df)
# benchmark.visualize_summary_statistics(summary_df)
# deviations_df = benchmark.calculate_deviations()
# print(deviations_df)
# benchmark.visualize_deviations(deviations_df)
