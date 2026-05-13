a = Analysis(
    ['EmotionDetection.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'deepface',
        'tensorflow',
        'retinaface',
        'cv2',
        'pyttsx3',
    ],
    hookspath=[],
    collect_all=['tensorflow', 'deepface'],
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