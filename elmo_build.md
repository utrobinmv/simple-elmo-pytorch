#build
python3 -m build


#upload to pypl
twine upload --repository pypi dist/*

#local install
pip uninstall simple_elmo_pytorch
python setup.py install
