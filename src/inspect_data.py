import scipy.io
import numpy as np

data = scipy.io.loadmat("data/corn.mat")
propvals = data['propvals']
print("propvals title:", propvals['title'][0, 0])
if 'label' in propvals.dtype.names:
    print("propvals labels:")
    labels = propvals['label'][0, 0]
    for i, l in enumerate(labels):
        print(f"  Col {i}: {l}")

m5spec = data['m5spec']
if 'axisscale' in m5spec.dtype.names:
    wavelengths = m5spec['axisscale'][0, 0][1][0]
    print(f"Wavelength range: {wavelengths[0]} nm to {wavelengths[-1]} nm ({len(wavelengths)} bands)")

Y = propvals['data'][0, 0]
print("Y shape:", Y.shape)
print("Moisture range (Col 0?):", np.min(Y[:, 0]), "to", np.max(Y[:, 0]), "mean:", np.mean(Y[:, 0]))
print("Col 1 range:", np.min(Y[:, 1]), "to", np.max(Y[:, 1]), "mean:", np.mean(Y[:, 1]))
print("Protein range (Col 2?):", np.min(Y[:, 2]), "to", np.max(Y[:, 2]), "mean:", np.mean(Y[:, 2]))
print("Starch range (Col 3?):", np.min(Y[:, 3]), "to", np.max(Y[:, 3]), "mean:", np.mean(Y[:, 3]))
