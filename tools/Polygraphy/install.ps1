python.exe -m pip install wheel colored
python.exe setup.py bdist_wheel
$wheel_path = gci -Name dist
python.exe -m pip install --force-reinstall dist\$wheel_path
