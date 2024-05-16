# Usage (Without GPU acceleration)

Simply run the command

```python
pip install -r requirements.txt
py ./sklearn_model.py
```

# Usage (With GPU acceleration using xgboost and lightgbm libraries)

```python
pip install -r requirements.txt
py ./gpuaccel_model.py
```

# Usage (With GPU acceleration using cuML in Windows 64-bit)

Open a WSL terminal and run the following commands:

```shell
cd ~
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
conda --version
conda create -n rapids-21.12 -c rapidsai -c nvidia -c conda-forge -c defaults cuml=21.12 python=3.8
conda activate rapids-21.12
conda install pandas numpy matplotlib
```

then you can run

```shell
python ./cuML_model.py
```

and you should see your results!
