FROM osrf/ros:noetic-desktop-full

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    iputils-ping \
    net-tools \
    wget \
    nano \
    python3-catkin-tools \
    python3-pip \
    libopencv-dev \
    git \
    ros-noetic-vision-msgs \
    sysvbanner \
    figlet

# Create a user with the specified UID
ARG USER_ID
ARG USER_NAME
ARG GROUP_ID
ARG GROUP_NAME
ARG WORKSPACE

# Create a new user with the specified UID and GID
RUN groupadd -g $GROUP_ID $GROUP_NAME \
&& useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USER_NAME && \
echo "$USER_NAME:$USER_NAME" | chpasswd && adduser $USER_NAME sudo
    
# Create a new catkin workspace inside the container and set the owner to the new user
WORKDIR /$WORKSPACE
RUN chown -R $USER_NAME:$GROUP_NAME /$WORKSPACE

# Switch to the new user
USER $USER_ID

RUN echo "source /opt/ros/noetic/setup.bash" >> $HOME/.bashrc
ENV PATH=$PATH:/opt/ros/noetic/bin

ENV PATH=$PATH:$HOME/.local/bin
RUN export PATH=$PATH:$HOME/.local/bin && \
    echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc

# Add the figlet command to the bashrc to display a welcome message
RUN echo "figlet -f slant 'Ball Detector!'" >> ~/.bashrc

RUN python3 -m pip install --upgrade pip && \
    pip install ultralytics && \
    pip install onnx && \
    pip install onnxslim && \
    pip install onnxruntime && \
    pip install opencv-python

# Set the default entry point to start the ROS environment
CMD ["tail", "-f", "/dev/null"]
