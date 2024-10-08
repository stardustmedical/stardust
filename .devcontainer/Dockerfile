FROM python:3.9-slim-buster

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libopenslide0 \
    gcc \
    python3-dev \
    zsh \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the working directory
WORKDIR /workspaces/stardust

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Switch to the non-root user
USER $USERNAME

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip \
    && pip install --user --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Set zsh as the default shell
SHELL ["/bin/zsh", "-c"]

# Set the default command
CMD ["/bin/zsh"]