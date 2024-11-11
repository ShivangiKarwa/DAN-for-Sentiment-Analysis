Please follow the below instructions to run the file -

1) Create a python virtual environment -

		conda create -n py_env   

2) Activate the virtual environment -

		conda activate py_env

3) Install torch library -
	
		pip3 install torch     ---- on mac
		pip install torch      ---- on windows

4) Install scikit-learn and matplotlib - 

		pip3 install scikit-learn matplotlib

5) Run the DAN model with glove embeddings - 

		python3 main.py --model DAN   

6) Run the DAN model without glove embeddings - 

		python3 main.py --model DANWG

7) Run the DAN Model with BPE implementation - 

		python3 main.py --model SUBWORDDAN 