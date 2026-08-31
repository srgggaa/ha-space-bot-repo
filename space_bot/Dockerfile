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
    py3-numpy \
    curl

# Alpine's py3-opencv doesn't include cv2.data (the haarcascade XML files
# that pip's opencv-python bundles), so fetch the two we need directly from
# the OpenCV repo and point the app at them via CASCADE_DIR.
RUN mkdir -p /app/haarcascades \
  && curl -fsSL -o /app/haarcascades/haarcascade_frontalface_default.xml \
     https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml \
  && curl -fsSL -o /app/haarcascades/haarcascade_upperbody.xml \
     https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_upperbody.xml
ENV CASCADE_DIR=/app/haarcascades/

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

# py3-opencv installs cv2 into Alpine's system site-packages; make sure pip
# packages installed above (in the same site-packages dir) can see it, and
# confirm the downloaded cascade files actually load.
RUN python3 -c "\
import cv2; \
print('OpenCV OK:', cv2.__version__); \
c = cv2.CascadeClassifier('/app/haarcascades/haarcascade_frontalface_default.xml'); \
assert not c.empty(), 'face cascade failed to load'; \
print('Cascades OK')"

COPY space_bot.py /app/space_bot.py
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Persist the "already posted" / "human detected" logs across restarts
VOLUME ["/data"]
WORKDIR /data

CMD [ "/run.sh" ]
