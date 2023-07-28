def filenameNoExt(nonDirFile: str):
    dotIndex = nonDirFile.rfind(".")
    if dotIndex != -1:
        return nonDirFile[:dotIndex]
    return nonDirFile

def fileExt(nonDirFile: str):
    dotIndex = nonDirFile.rfind(".")
    if dotIndex != -1 and dotIndex+1 < len(nonDirFile):
        return nonDirFile[dotIndex+1:]
    return ""