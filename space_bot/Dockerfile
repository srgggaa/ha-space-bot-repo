ARG BUILD_FROM
FROM $BUILD_FROM

# Home Assistant base images are Alpine-based. Building opencv-python from
# source there is unreliable (needs a long chain of native deps), so instead
# we install Alpine's own precompiled OpenCV package plus numpy, and only use
# pip for the pure-Python dependencies.
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-opencv \
    py3-numpy

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

# py3-opencv installs cv2 into Alpine's system site-packages; make sure pip
# packages installed above (in the same site-packages dir) can see it.
RUN python3 -c "import cv2, numpy; print('OpenCV OK:', cv2.__version__)"

COPY space_bot.py /app/space_bot.py
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Persist the "already posted" / "human detected" logs across restarts
VOLUME ["/data"]
WORKDIR /data

CMD [ "/run.sh" ]
