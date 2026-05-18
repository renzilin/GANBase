# Note

If your sequencing server environment is incompatible with the api, you may need to perform additional environment configuration according to the requirements of read_until_api.
You need to unzip the bonito and read_until_api folders. We've stored the Bonito weights in a separate compressed folder. you can add them to the model as needed.

```bash
python3 -m venv read_until_env
source read_until_env/bin/activate
pip install --upgrade pip
cd read_until_api-3.0.0/
pip install numpy==1.22.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install minknow-api==4.0.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
cd 到bonito

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .

pip install scikit-learn scipy grpcio
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple

```



