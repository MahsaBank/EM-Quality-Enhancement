FROM gcr.io/ris-registry-shared/novnc:ubuntu20.04_cuda11.0
RUN apt-get update -y && apt-get install -y python
RUN apt-get install nano
ADD ./requirements.txt /home/bmahsa/requirements.txt
WORKDIR /home/bmahsa
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade numpy