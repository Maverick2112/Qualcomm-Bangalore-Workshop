# File_name : setup_ort.ps1
                    
# Make python environment 
py -3.10 -m venv SDX_ENV
# activate environment
& "SDX_ENV\Scripts\Activate.ps1" 
python -m pip install --upgrade pip 
pip install numpy==1.26.4
pip install onnx==1.16.1
pip install pillow==10.3.0
pip install torchvision==0.18.1