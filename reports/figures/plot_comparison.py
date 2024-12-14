import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Definizione dei colori personalizzati
teal = mcolors.to_rgba('teal')
factor = 0.75
dark_teal = (teal[0] * factor, teal[1] * factor, teal[2] * factor, teal[3])
maroon = mcolors.to_rgba('maroon')
grey = mcolors.to_rgba('grey')

# Funzione per aggregare i CSV di valutazione
def aggregate_evaluation_metrics(folder):
    metrics = []
    method_names = []
    
    # Scansione della cartella per trovare i file con nome {method_name}_evaluation_metrics.csv
    for filename in os.listdir(folder):
        if filename.endswith('_evaluation_metrics.csv'):
            # Estrai il nome del metodo (senza estensione)
            method_name = filename.replace('_evaluation_metrics.csv', '')
            method_names.append(method_name)
            
            # Leggi il file CSV
            df = pd.read_csv(os.path.join(folder, filename))
            
            # Aggiungi solo precision, recall, f1-score (escludendo accuracy)
            metrics.append(df[['Precision', 'Recall', 'F1-Score']].iloc[0])
    
    # Combina tutti i dati in un DataFrame
    metrics_df = pd.DataFrame(metrics, index=method_names)
    return metrics_df

# Funzione per calcolare l'IoU medio
def calculate_mean_iou(folder):
    iou_results = {}
    
    # Scansione della cartella per trovare i file con nome {method_name}_iou_results.csv
    for filename in os.listdir(folder):
        if filename.endswith('_iou_results.csv'):
            # Estrai il nome del metodo (senza estensione)
            method_name = filename.replace('_iou_results.csv', '')
            
            # Leggi il file CSV
            df = pd.read_csv(os.path.join(folder, filename))
            
            # Filtra i valori di IoU maggiori di 0.01
            df_filtered = df[df['IoU'] > 0.01]
            
            # Calcola l'IoU medio
            mean_iou = df_filtered['IoU'].mean()
            
            # Memorizza il risultato
            iou_results[method_name] = mean_iou
    
    return iou_results

# Funzione per creare il grafico di confronto
def plot_comparison(metrics_df, iou_results):
    # Crea un grafico per precision, recall, f1-score e IoU
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot precision, recall, f1-score con i colori definiti
    metrics_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax,
        color=[maroon, grey, dark_teal])
    
    # Aggiungi i valori di IoU al grafico
    ax2 = ax.twinx()
    ax2.plot(iou_results.keys(), iou_results.values(), color='black', marker='o', label='Mean IoU', linestyle='--')
    
    # Etichette e titolo
    ax.set_xlabel('Method')
    ax.set_ylabel('Metric Values')
    ax2.set_ylabel('Mean IoU')
    ax.set_title('Evaluation Metrics and Mean IoU for Each Method')
    
    # Legende
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

# Funzione principale per eseguire il tutto
def main(folder):
    # Aggregazione delle metriche di valutazione
    metrics_df = aggregate_evaluation_metrics(folder)
    
    # Calcolo dell'IoU medio per ogni metodo
    iou_results = calculate_mean_iou(folder)
    
    # Unire le metriche e IoU nel grafico
    plot_comparison(metrics_df, iou_results)

# Esegui lo script passando la cartella contenente i CSV
folder = 'reports'  # Sostituisci con il percorso della cartella che contiene i CSV
main(folder)
