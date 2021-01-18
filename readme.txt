1. Copy "Include" folder to anywhere
2. Delete venv and make new one with same name
3. Move Include folder to original directory
4. copy requirements.txt to \venv\Scripts\

get into cmd
python -m venv venv
cd venv\Scripts\activate.bat
pip install -r requirements.txt
deactivate.bat

5. File-> settings-> python interpretor -> choose the venv's one.
