sudo apt-get install python3-pip\
                     python3-dev\
                     libxml2-dev\
                     libxslt-dev\
                     libjpeg-dev\
                     zlib1g-dev\
                     libpng12-dev\
                     python3-pip
sudo pip3 install virtualenv
virtualenv -p python3 venv

source venv/bin/activate
pip install -r requirements.txt
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python

