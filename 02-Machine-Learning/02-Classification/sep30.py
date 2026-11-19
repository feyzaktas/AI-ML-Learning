from sklearn import metrics
import matplotlib.pyplot as plt

yTrue=[1,1,0,0,1,0,0,0,1,1]
yPred=[1,1,0,0,0,1,1,0,1,1]

#TP:4 TN:3 FP:2 FN:1

cm=metrics.confusion_matrix(yTrue, yPred)
print(cm)
print("---------------------------")

# cm_display=metrics.ConfusionMatrixDisplay(cm, display_labels=["Kanser Değil", "Kanser"])
# cm_display.plot()
# plt.show()
print("----------------------------------")

print(metrics.accuracy_score(yTrue,yPred))
print(metrics.precision_score(yTrue,yPred)) 
print(metrics.recall_score(yTrue,yPred))
print(metrics.f1_score(yTrue,yPred))
print(metrics.roc_auc_score(yTrue,yPred))
print("-----------------------------------------")
print(metrics.classification_report(yTrue,yPred))



