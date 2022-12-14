# THIS FILE IS AUTOGENERATED FROM .binder/Dockerfile.in


# the docker container for binder needs pymor installed entirely inside
# the container, local dev needs it in path from mounted src
# we trick docker into fulfilling both roles via a conditional ONBUILD
# if you want to use the local dev setup, see docker/docker-compose.yml
ARG BASE=pymor/jupyter_py3.7:06d87fc5395c39231eb05b6df06266dab9a4dbf9
ARG BUILD_ENV=binder

FROM $BASE as image_binder
ONBUILD ADD . /pymor

FROM $BASE as image_dev
ONBUILD RUN echo "dev image uses mounted pymor" && mkdir /pymor
ONBUILD ENV PYTHONPATH=/pymor/src:${PYTHONPATH}

# select "base" image according to build arg
FROM image_${BUILD_ENV}
MAINTAINER rene.fritze@wwu.de

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID
ARG PYMOR_JUPYTER_TOKEN

USER root
RUN useradd -d /home/pymor --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER} && \
    chown -R ${NB_USER} /home/pymor /pymor/

# do local package install before requirements to avoid conflict
COPY .binder/local_packages /pymor/local_packages/
RUN for i in $(find /pymor/local_packages -maxdepth 1 -mindepth 1 -type d | grep -v keepdir) ; do pip install -e $i ; done
RUN bash -c "([[ -e /pymor/setup.py ]] && pip install /pymor[docs,ci,full] || echo 'no install needed') && \
    rm -rf /pymor/local_packages"
USER ${NB_USER}

ENV JUPYTER_TOKEN=${PYMOR_JUPYTER_TOKEN} \
    USER=${NB_USER} \
    HOME=/home/pymor

ENTRYPOINT []
WORKDIR /pymor/notebooks
