all: build install

build:
       python3 setup.py build

install:
       python3 setup.py install --user
       cp sbb_predict/sbb_predict.py /home/vahid/bin/sbb_predict
       chmod +x /home/vahid/bin/sbb_predict  
