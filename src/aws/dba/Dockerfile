FROM opensuse/leap

# Bring system up to date
RUN zypper update -y && \
    zypper install -y vi less which && \
    zypper install -y python3 python3-pip python3-devel \
		      postgresql12 postgresql12-server postgresql12-server-devel \
		      gcc

# Misc config
RUN mkdir /models
RUN mkdir /dba_scripts
RUN mkdir /application
RUN echo 'set -o vi' >> /root/.bashrc
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy local files
COPY models /models
COPY dba_scripts /dba_scripts
COPY application /application

# Install dependencies
RUN cd /application && \
    pip install -r ./requirements.txt
RUN cd /dba_scripts && \
    pip install -r ./requirements.txt

CMD ["python", "/application/db_init.py"]
CMD ["/bin/bash", "/dba_scripts/dba.sh"]
