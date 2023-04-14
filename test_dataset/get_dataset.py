import urllib.request
import tarfile

url = "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz"
file_name = "20news-18828.tar.gz"

urllib.request.urlretrieve(url, file_name)

with tarfile.open(file_name, "r:gz") as tar:
    tar.extractall()
