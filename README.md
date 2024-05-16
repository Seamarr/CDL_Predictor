# Scraping the data

This will take some time, but make sure to gather the data necessary by running the scraping script:

```shell
pip install -r requirements.txt
py ./scrapeData.py
```

After some time, the data will be saved into a file named "preprocessed_player_stats.csv". Rename this to whatever
you would like and make sure to change the data loading in whatever model you are using to this new file.

# Usage (Without GPU acceleration)

Simply run the command

```shell
py ./sklearn_model.py
```

# Usage (With GPU acceleration using xgboost and lightgbm libraries)

```shell
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

## Common bugs/errors

### Feature arrays are not updated

The feature array in the models may be out of date. For ex. a new player has joined the league that we
are not accounting for in the feature array. You will need to get the new players and add them to the feature array.
