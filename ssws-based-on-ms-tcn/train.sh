python3 train.py --config ./result/gtea/ms-tcn/split1/config.yaml
python3 train.py --config ./result/gtea/ms-tcn/split2/config.yaml
python3 train.py --config ./result/gtea/ms-tcn/split3/config.yaml
python3 train.py --config ./result/gtea/ms-tcn/split4/config.yaml

python3 eval.py ./result/gtea/ms-tcn/split1/config.yaml test
python3 eval.py ./result/gtea/ms-tcn/split2/config.yaml test
python3 eval.py ./result/gtea/ms-tcn/split3/config.yaml test
python3 eval.py ./result/gtea/ms-tcn/split4/config.yaml test

python3 train.py --config ./result/50salads/ms-tcn/split1/config.yaml
python3 train.py --config ./result/50salads/ms-tcn/split2/config.yaml
python3 train.py --config ./result/50salads/ms-tcn/split3/config.yaml
python3 train.py --config ./result/50salads/ms-tcn/split4/config.yaml
python3 train.py --config ./result/50salads/ms-tcn/split5/config.yaml


python3 eval.py ./result/50salads/ms-tcn/split1/config.yaml test
python3 eval.py ./result/50salads/ms-tcn/split2/config.yaml test
python3 eval.py ./result/50salads/ms-tcn/split3/config.yaml test
python3 eval.py ./result/50salads/ms-tcn/split4/config.yaml test
python3 eval.py ./result/50salads/ms-tcn/split5/config.yaml test