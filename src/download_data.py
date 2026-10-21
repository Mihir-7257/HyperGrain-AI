import os
import urllib.request
import scipy.io
import numpy as np

def download_corn_dataset(dest_dir="data"):
    os.makedirs(dest_dir, exist_ok=True)
    mat_path = os.path.join(dest_dir, "corn.mat")
    
    url = "https://www.eigenvector.com/data/Corn/corn.mat"
    if not os.path.exists(mat_path):
        print(f"Downloading {url} to {mat_path}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(mat_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete!")
    else:
        print(f"File already exists at {mat_path}")
        
    # Inspect content
    data = scipy.io.loadmat(mat_path)
    print("Keys in .mat file:", [k for k in data.keys() if not k.startswith('__')])
    
    # Check structure
    for key in ['m5spec', 'mp5spec', 'mp6spec', 'propvals']:
        if key in data:
            val = data[key]
            if isinstance(val, np.ndarray) and val.dtype.names:
                print(f"Key {key}: structured array with fields {val.dtype.names}")
                if 'data' in val.dtype.names:
                    print(f"  {key}.data shape: {val['data'][0, 0].shape}")
            else:
                print(f"Key {key}: shape {getattr(val, 'shape', type(val))}")
                
if __name__ == "__main__":
    download_corn_dataset()
