python.exe -m pip install build colored
python.exe -m build --wheel
$wheel_path = gci -Name dist
python.exe -m pip install --force-reinstall dist\$wheel_path
