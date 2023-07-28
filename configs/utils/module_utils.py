import os
import site

def getSitePackages():
    for path in site.getsitepackages():
        if path.endswith("site-packages"):
            return path
    raise FileNotFoundError("Cannot find 'site-packages'")

def getCTransformersCudaLib_Windows():
    import ctransformers
    finalPath = os.path.normpath(os.path.join(getSitePackages(), "ctransformers/lib/cuda/ctransformers"))
    if not os.path.exists(finalPath + ".dll"):
        raise ModuleNotFoundError("Cannot find cuda lib for ctransformers in: '", finalPath, "'. Maybe you need to install it with 'CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers' or simply download 'ctransformers.dll' from its GitHub")
    return finalPath

def getCTransformersCudaLib_Unix():
    import ctransformers
    finalPath = os.path.normpath(os.path.join(getSitePackages(), "ctransformers/lib/cuda/libctransformers"))
    if not os.path.exists(finalPath + ".so"):
        raise ModuleNotFoundError("Cannot find cuda lib for ctransformers in: '", finalPath, "'. Maybe you need to install it with 'CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers' or simply download 'libctransformers.so' from its GitHub")
    return finalPath
