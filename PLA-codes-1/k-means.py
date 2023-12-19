import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from dataset import *
from main import config
from models.mlp import MLP
from models.siamese import SiameseNet
from models.conv import CNN


# window_size = config['window_size']
# fdt = FeatureDatasetV1(2, f'features-{window_size}')
# x = fdt.data.numpy()
# y = fdt.labels.numpy()

# # cdt = CSIDatasetV5(1)
# # cdt = CSIDatasetV2(2, 'csi')
# # x = cdt.data.numpy()
# # y = cdt.labels.numpy()

cdt = CSIDatasetV2(2, 'csi')
gmm = GaussianMixture(n_components=2).fit(cdt.data.numpy())

# kmeans = KMeans(n_clusters=2, random_state=42).fit(x)
# y = 1-y
# print('KMeans')
acc = metrics.accuracy_score(cdt.labels.numpy(), gmm.predict(cdt.data.numpy()))
print(f'acc: {acc}')

# cdt = CSIDatasetV2(2, 'csi')
# model = CNN(1, 2880)
# total = 0
# for i in range(10000):
#     item= cdt[i][0]
#     start = time.time()
#     model(item.unsqueeze(0).unsqueeze(0).unsqueeze(0))
#     end = time.time()
#     total += (end-start)
# avg_delay = total/10000

# print(avg_delay)

# gmm 0.0026281085014343263s
# ours 0.004534774136543273s
