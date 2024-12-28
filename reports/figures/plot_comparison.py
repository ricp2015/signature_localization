import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define custom colors
teal = mcolors.to_rgba('teal')
factor = 0.75
dark_teal = (teal[0] * factor, teal[1] * factor, teal[2] * factor, teal[3])
maroon = mcolors.to_rgba('maroon')
grey = mcolors.to_rgba('grey')

# Function to aggregate evaluation metrics from CSV files
def aggregate_evaluation_metrics(folder, suffix=""):
    metrics = []
    method_names = []
    
    # Scan the folder for files matching {method_name}_evaluation_metrics{suffix}.csv
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(f'_evaluation_metrics{suffix}.csv'):
            # Extract the method name (without suffix and extension)
            method_name = filename.replace(f'_evaluation_metrics{suffix}.csv', '')
            method_names.append(method_name)
            
            # Read the CSV file
            df = pd.read_csv(os.path.join(folder, filename))
            
            # Append only precision, recall, and f1-score (excluding accuracy)
            metrics.append(df[['Precision', 'Recall', 'F1-Score']].iloc[0])
    
    # Combine all data into a DataFrame
    metrics_df = pd.DataFrame(metrics, index=method_names)
    return metrics_df.sort_index()  # Ensure the DataFrame is sorted by method names

# Function to calculate the mean IoU
def calculate_mean_iou(folder, suffix=""):
    iou_results = {}
    
    # Scan the folder for files matching {method_name}_iou_results{suffix}.csv
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(f'_iou_results{suffix}.csv'):
            # Extract the method name (without suffix and extension)
            method_name = filename.replace(f'_iou_results{suffix}.csv', '')
            
            # Read the CSV file
            df = pd.read_csv(os.path.join(folder, filename))
            
            # Filter IoU values greater than 0.01
            df_filtered = df[df['IoU'] > 0.01]
            
            # Calculate the mean IoU
            mean_iou = df_filtered['IoU'].mean()
            
            # Store the result
            iou_results[method_name] = mean_iou
    
    # Return a sorted dictionary based on method names
    return dict(sorted(iou_results.items()))

# Function to create a comparison plot
def plot_comparison(metrics_df, iou_results):
    # Ensure metrics_df and iou_results are aligned
    iou_series = pd.Series(iou_results).reindex(metrics_df.index)
    
    # Create a bar plot for precision, recall, and f1-score
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)

    # Plot precision, recall, and f1-score using defined colors
    metrics_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax, color=[maroon, grey, dark_teal])
    
    # Add IoU values to the plot
    ax2 = ax.twinx()
    ax2.plot(iou_series.index, iou_series.values, color='black', marker='o', label='Mean IoU', linestyle='--')
    
    # Labels and title
    ax.set_xlabel('Method')
    ax.set_ylabel('Metric Values')
    ax2.set_ylabel('Mean IoU')
    ax.set_title('Evaluation Metrics and Mean IoU for Each Method')
    
    # Legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Main function to run the pipeline
def main(folder, suffix=""):
    # Aggregate evaluation metrics
    metrics_df = aggregate_evaluation_metrics(folder, suffix)
    
    # Calculate the mean IoU for each method
    iou_results = calculate_mean_iou(folder, suffix)
    
    # Combine metrics and IoU in the plot
    plot_comparison(metrics_df, iou_results)

# Run the script for the specified directories with appropriate suffixes
main('reports/naive/', suffix="")
main('reports/fast/', suffix="_fast")
