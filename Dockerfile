FROM --platform=${BUILDPLATFORM} condaforge/miniforge3:23.3.1-1

# Set working directory
WORKDIR /app

# Install ZSH and terminal utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file first for better caching
COPY environment.yml /app/

# Update conda environment
RUN mamba env update -n base --file environment.yml

# Copy application code
COPY . /app

# Set ZSH as default shell
SHELL ["/bin/zsh", "-c"]
CMD ["/bin/zsh"]
