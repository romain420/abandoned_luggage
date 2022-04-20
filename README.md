# abandoned_luggage
Detection of abandoned luggage in complex environment

How to install and use our abandonned luggage detector:

- first of all, download from our github (https://github.com/romain420/abandoned_luggage) and extract the archive.
- (OPTIONAL), create a new environement
- then, from a powershell (windows) or a terminal (linux):
```bash
pip install -r ./requirements.txt
```
- to run the program(be careful that python is recognized as a command, else try python3, py, py3, or add python to the environement variable 'PATH'):

```bash
python ./detect.py # to run with your camera in real time with 128p (bad quality, depends on your computer), good frame per seconds
```
```bash
python ./detect.py --imgsz 640 # good results, very demanding on your computer
```
```bash
python ./detect.py --source video.mp4 # to analyze your own video, frame per frame, defaut is in 128p
```
```bash
python ./detect.py --imgsz 640 --source video.mp4 # best results by far, 640p, on your own video
```
