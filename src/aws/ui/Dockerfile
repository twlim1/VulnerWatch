FROM opensuse/leap

# Bring system up to date
RUN zypper update -y && \
    zypper install -y vi less which && \
    zypper install -y python3 python3-pip postgresql12

# Misc config
RUN mkdir /application
RUN echo 'set -o vi' >> /root/.bashrc
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV FLASK_APP=/application/nlp260app.py

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy local files
COPY application /application

# Install dependencies
RUN cd /application && \
    pip install -r ./requirements.txt

# Misc config
RUN echo 'set -o vi' >> /root/.bashrc

WORKDIR /application
# CMD ["/bin/bash", "/application/run.sh"]
