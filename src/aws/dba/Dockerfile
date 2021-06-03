FROM opensuse/leap

# Bring system up to date
RUN zypper update -y && \
    zypper install -y vi less which && \
    zypper install -y python3 python3-pip python3-devel \
		      postgresql12 postgresql12-server postgresql12-server-devel \
		      gcc

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Misc config
RUN mkdir /models
RUN mkdir /dba_scripts
RUN mkdir /application
RUN echo 'set -o vi' >> /root/.bashrc
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy local files
COPY dba_scripts /dba_scripts
COPY application /application

# Install dependencies
RUN cd /application && \
    pip install  --no-cache-dir -r ./requirements.txt
RUN cd /dba_scripts && \
    pip install --no-cache-dir -r ./requirements.txt

CMD ["/bin/bash", "/dba_scripts/dba.sh"]
