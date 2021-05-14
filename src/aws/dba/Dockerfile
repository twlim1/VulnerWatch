FROM opensuse/leap

# Bring system up to date
RUN zypper update -y && \
    zypper install -y vi less which && \
    zypper install -y python3 python3-pip postgresql12

# Misc config
RUN echo 'set -o vi' >> /root/.bashrc

CMD ["/bin/bash"]
