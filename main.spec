# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        #('models/buffalo_l/*.onnx', 'models/buffalo_l'),  # 注意这里改为保留完整路径
        #('models/buffalo_s/*.onnx', 'models/buffalo_s'),
        ('models/buffalo_sc/*.onnx', 'models/buffalo_sc'),
        ('models/yolo/*.onnx', 'models/yolo'),  # 这样会保留yolo子目录
        # 任何其他需要的资源类型
    ],
    hiddenimports=[
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_info_print.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
