truth = r"E:\ken\2023 - Mass Migration\MigrationForecastingLSTM\save\TCN\Total_truth_val.npy"
pred = r"E:\ken\2023 - Mass Migration\MigrationForecastingLSTM\save\TCN\Total_predict_val.npy"

import numpy as np
import matplotlib.pyplot as plt

tr = np.load(truth)
pr = np.load(pred)
fig = plt.figure(figsize=(8, 8))
fig.tight_layout()
plt.plot(pr,"-ro",label="Prediction")
plt.plot(tr,"-bo",label="Truth")

plt.legend()

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("plot.pdf") as pdf:
    pdf.savefig(fig,bbox_inches='tight')