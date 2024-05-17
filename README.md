# Scraping the data

This will take some time, but make sure to gather the data necessary by running the scraping script:

```shell
pip install -r requirements.txt
py ./scrapeData.py
```

After some time, the data will be saved into a file named "preprocessed_player_stats.csv". Rename this to whatever
you would like and make sure to change the data loading in whatever model you are using to this new file.

# Usage (With GPU acceleration using pytorch) RECOMMENDED

### Create a Conda Environment

```shell
conda create -n ml_project python=3.9
conda activate ml_project
```

### Install Jupyter, PyTorch, and other libraries

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install jupyter pandas scikit-learn matplotlib
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu{VERSIONNUM}/torch_stable.html
```

Note\* You will have to select the cuda version compatible with yor GPU. You can run the command:

```shell
nvidia-smi
```

in a shell to see what CUDA version your gpu supports. Then, in the pip install replace "{VERSIONNUM}" with your compatible version.
Version 11.8 will be "118" and 12.4 will be "124", etc.

To deactivate, run:

```shell
conda deactivate
```

### Option 1: Combined models

You may use the combined_models python script to get the predictions from all the different models. Not that
this is not a combination of all the models into one, this is just all of the models combined into 1 file for
simplicity of comparison.

```shell
python ./combined_models.py
```

### Option 2: Jupyter Notebook

```shell
jupyter notebook
```

Use "torch_modelv2.ipynb", select the kernel of your jupyter notebook link, and thats it!

# Usage (Without GPU acceleration) SLOW

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
conda deactivate rapids-21.12
```

then you can run

```shell
conda activate rapids-21.12
python ./cuML_model.py
```

## Common bugs/errors

### Feature arrays are not updated

The feature array in the models may be out of date. For ex. a new player has joined the league that we
are not accounting for in the feature array. You will need to get the new players and add them to the feature array.
