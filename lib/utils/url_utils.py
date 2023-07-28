from urllib.parse import urlparse

def isUriValid(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False