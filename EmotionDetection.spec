a = Analysis(
    ['EmotionDetection.py'],
    pathex=[],
    binaries=[],
    datas=[
        (r'C:\hostedtoolcache\windows\Python\3.11.9\x64\Lib\site-packages\cv2\data', 'cv2/data'),
        (r'C:\hostedtoolcache\windows\Python\3.11.9\x64\Lib\site-packages\deepface', 'deepface'),
    ],
    hiddenimports=[
        'deepface',
        'tensorflow',
        'retinaface',
        'cv2',
        'pyttsx3',
        'tf_keras',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    collect_all=['tensorflow', 'deepface', 'retinaface'],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='EmotionDetection',
    console=True,
    onefile=True,
)