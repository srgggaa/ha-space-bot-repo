ARG BUILD_FROM
FROM $BUILD_FROM

# Home Assistant base images are Alpine-based; these are needed for
# opencv-python-headless and general Python build tooling.
RUN apk add --no-cache \
    python3 \
    py3-pip \
    libstdc++ \
    libgomp \
    jpeg \
    libjpeg-turbo \
    tiff \
    libpng \
  && apk add --no-cache --virtual .build-deps \
    python3-dev \
    build-base \
    gcc \
    musl-dev \
    jpeg-dev \
    zlib-dev \
    tiff-dev \
    libpng-dev

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt \
  && apk del .build-deps

COPY space_bot.py /app/space_bot.py
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Persist the "already posted" / "human detected" logs across restarts
VOLUME ["/data"]
WORKDIR /data

CMD [ "/run.sh" ]
