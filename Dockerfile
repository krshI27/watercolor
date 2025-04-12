FROM --platform=${BUILDPLATFORM} condaforge/miniforge3:latest
WORKDIR /app
COPY . /app
RUN conda env update -n base --file environment.yml
