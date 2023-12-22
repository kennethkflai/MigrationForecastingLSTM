

import numpy as np
import matplotlib.pyplot as plt

def plot_graph(tr, pr, title, model):
    tr = np.load(truth)
    predictions = []
    for prediction in pr:
        predictions.append(np.load(prediction))
        
    fig = plt.figure(figsize=(12, 4))
#    fig.tight_layout()
    
    for index, m in enumerate(model):
        plt.plot(predictions[index],"--",label=m)
        
    plt.plot(tr,"-",color=[0,0,0],label="Truth")
    
    plt.rc('xtick', labelsize=20)   
    plt.rc('ytick', labelsize=20) 
    plt.rc('legend', fontsize=20) 
    plt.rc('font', size=20) 
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(title)
    
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(title + ".pdf") as pdf:
        pdf.savefig(fig,bbox_inches='tight')
        


for i in ["Employment","Sponsor","Refugee","Economic"]:
    pred = []
    for model in ["TCN", "BiLSTM","LSTM"]:
        truth = r"E:\ken\2023 - Mass Migration\MigrationForecastingLSTM\save\\" + model + "\\"+ i + "_truth_test_frame60.npy"
        pred.append(r"E:\ken\2023 - Mass Migration\MigrationForecastingLSTM\save\\" + model + "\\"+ i + "_predict_test_frame60.npy")
    
    plot_graph(truth,pred, i, ["TCN", "BiLSTM","LSTM"])